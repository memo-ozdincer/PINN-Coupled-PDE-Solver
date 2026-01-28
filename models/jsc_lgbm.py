"""
LightGBM model for J_sc prediction with analytical ceiling feature.
Uses GPU acceleration and massive parallelization.
"""
import numpy as np
import lightgbm as lgb
from dataclasses import dataclass, field


@dataclass
class JscLGBMConfig:
    """Configuration for Jsc LightGBM model."""
    # Core parameters
    num_leaves: int = 255
    max_depth: int = 15
    learning_rate: float = 0.05
    n_estimators: int = 2000
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1

    # GPU settings
    device: str = 'gpu'
    gpu_platform_id: int = 0
    gpu_device_id: int = 0

    # Training settings
    early_stopping_rounds: int = 50
    verbose: int = -1
    n_jobs: int = -1
    random_state: int = 42

    # Feature settings
    use_ceiling_feature: bool = True
    use_physics_features: bool = True

    def to_lgb_params(self) -> dict:
        """Convert to LightGBM parameter dict."""
        return {
            'objective': 'regression',
            'metric': ['rmse', 'mae'],
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'device': self.device,
            'gpu_platform_id': self.gpu_platform_id,
            'gpu_device_id': self.gpu_device_id,
            'verbose': self.verbose,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'force_col_wise': True,
        }


class JscLGBM:
    """
    LightGBM model for Jsc with physics-informed features.

    Key insight: Jsc is bounded by the generation limit J_ceiling = q * G * L.
    We predict the ratio Jsc / J_ceiling, which is the collection efficiency.
    This constrains predictions to be physically meaningful.
    """

    def __init__(self, config: JscLGBMConfig):
        self.config = config
        self.model = None
        self.feature_names = None

    def _prepare_features(
        self,
        raw_params: np.ndarray,
        physics_features: np.ndarray,
        jsc_ceiling: np.ndarray
    ) -> np.ndarray:
        """
        Prepare input features including analytical ceiling.

        The ceiling is the theoretical maximum Jsc before any losses:
        J_ceiling = q * Gavg * lP
        """
        features_list = [raw_params]

        if self.config.use_physics_features:
            features_list.append(physics_features)

        if self.config.use_ceiling_feature:
            # Log of ceiling (for scale invariance)
            log_ceiling = np.log10(jsc_ceiling + 1e-30).reshape(-1, 1)
            features_list.append(log_ceiling)

        return np.hstack(features_list)

    def _prepare_target(self, jsc: np.ndarray, jsc_ceiling: np.ndarray) -> np.ndarray:
        """
        Prepare target as collection efficiency (ratio to ceiling).
        This bounds the target to [0, 1] approximately.
        """
        # Collection efficiency: actual / theoretical max
        efficiency = jsc / (jsc_ceiling + 1e-30)
        # Clip for numerical stability
        return np.clip(efficiency, 0, 2)

    def _inverse_target(self, efficiency: np.ndarray, jsc_ceiling: np.ndarray) -> np.ndarray:
        """Convert efficiency back to Jsc."""
        return efficiency * jsc_ceiling

    def fit(
        self,
        X_raw: np.ndarray,
        X_physics: np.ndarray,
        y_jsc: np.ndarray,
        jsc_ceiling: np.ndarray,
        X_raw_val: np.ndarray = None,
        X_physics_val: np.ndarray = None,
        y_jsc_val: np.ndarray = None,
        jsc_ceiling_val: np.ndarray = None
    ) -> dict:
        """
        Train the model.

        Args:
            X_raw: Raw parameters (N, 31)
            X_physics: Physics features (N, n_features)
            y_jsc: Target Jsc values (N,)
            jsc_ceiling: Analytical ceiling values (N,)
            *_val: Validation data (optional)

        Returns:
            Training history dict
        """
        # Prepare features and targets
        X_train = self._prepare_features(X_raw, X_physics, jsc_ceiling)
        y_train = self._prepare_target(y_jsc, jsc_ceiling)

        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)

        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]

        valid_sets = [train_data]
        valid_names = ['train']

        if X_raw_val is not None:
            X_val = self._prepare_features(X_raw_val, X_physics_val, jsc_ceiling_val)
            y_val = self._prepare_target(y_jsc_val, jsc_ceiling_val)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        # Train
        self.model = lgb.train(
            self.config.to_lgb_params(),
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        return {'best_iteration': self.model.best_iteration}

    def predict(
        self,
        X_raw: np.ndarray,
        X_physics: np.ndarray,
        jsc_ceiling: np.ndarray
    ) -> np.ndarray:
        """
        Predict Jsc values.

        Returns actual Jsc (not efficiency).
        """
        X = self._prepare_features(X_raw, X_physics, jsc_ceiling)
        efficiency = self.model.predict(X)
        return self._inverse_target(efficiency, jsc_ceiling)

    def predict_efficiency(
        self,
        X_raw: np.ndarray,
        X_physics: np.ndarray,
        jsc_ceiling: np.ndarray
    ) -> np.ndarray:
        """Predict collection efficiency directly."""
        X = self._prepare_features(X_raw, X_physics, jsc_ceiling)
        return self.model.predict(X)

    def feature_importance(self) -> dict:
        """Get feature importance."""
        return {
            'gain': self.model.feature_importance(importance_type='gain'),
            'split': self.model.feature_importance(importance_type='split')
        }

    def save(self, path: str):
        """Save model to file."""
        self.model.save_model(path)

    def load(self, path: str):
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)


def compute_jsc_ceiling_numpy(params: np.ndarray, col_idx: dict) -> np.ndarray:
    """
    Compute Jsc ceiling from raw parameters (numpy version).
    J_ceiling = q * G_avg * L_P

    NOTE: Handles both log10 and linear scale inputs for Gavg.
    """
    Q_E = 1.602e-19
    lP_m = params[:, col_idx['lP']] * 1e-9  # nm to m
    Gavg_raw = params[:, col_idx['Gavg']]

    # Check if values are in log10 scale (typically 20-30) or linear scale (huge numbers)
    # Use threshold of 100 to distinguish
    if np.abs(Gavg_raw).max() < 100:
        Gavg = 10 ** Gavg_raw  # log scale to linear
    else:
        Gavg = Gavg_raw  # already in linear scale

    return Q_E * Gavg * lP_m


def build_jsc_model(config: JscLGBMConfig) -> JscLGBM:
    """Factory function."""
    return JscLGBM(config)
