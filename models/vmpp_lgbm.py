"""
LightGBM model for V_mpp prediction with physics-informed features.
Uses predicted Voc as a critical input feature.
"""
import numpy as np
import lightgbm as lgb
from dataclasses import dataclass


@dataclass
class VmppLGBMConfig:
    """Configuration for Vmpp LightGBM model."""
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
    use_voc_feature: bool = True  # Use predicted/actual Voc as input
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


class VmppLGBM:
    """
    LightGBM model for Vmpp prediction.

    Key insights:
    1. Vmpp <= Voc always (physical constraint)
    2. Vmpp/Voc ratio (voltage fill factor) depends on:
       - Series resistance (Rs)
       - Extraction efficiency (Theta)
       - Recombination dynamics

    We predict the ratio Vmpp/Voc and multiply by Voc.
    """

    def __init__(self, config: VmppLGBMConfig):
        self.config = config
        self.model = None

    def _prepare_features(
        self,
        raw_params: np.ndarray,
        physics_features: np.ndarray,
        voc: np.ndarray
    ) -> np.ndarray:
        """
        Prepare input features including Voc.
        """
        features_list = [raw_params]

        if self.config.use_physics_features:
            features_list.append(physics_features)

        if self.config.use_voc_feature:
            voc_features = np.column_stack([
                voc,
                np.log10(voc + 1e-30),  # Log scale
            ])
            features_list.append(voc_features)

        return np.hstack(features_list)

    def _prepare_target(self, vmpp: np.ndarray, voc: np.ndarray) -> np.ndarray:
        """
        Prepare target as Vmpp/Voc ratio.
        This typically falls in [0.7, 0.95] for good cells.
        """
        ratio = vmpp / (voc + 1e-30)
        return np.clip(ratio, 0, 1)

    def _inverse_target(self, ratio: np.ndarray, voc: np.ndarray) -> np.ndarray:
        """Convert ratio back to Vmpp."""
        return ratio * voc

    def fit(
        self,
        X_raw: np.ndarray,
        X_physics: np.ndarray,
        y_vmpp: np.ndarray,
        voc: np.ndarray,
        X_raw_val: np.ndarray = None,
        X_physics_val: np.ndarray = None,
        y_vmpp_val: np.ndarray = None,
        voc_val: np.ndarray = None
    ) -> dict:
        """Train the model."""
        X_train = self._prepare_features(X_raw, X_physics, voc)
        y_train = self._prepare_target(y_vmpp, voc)

        train_data = lgb.Dataset(X_train, label=y_train)

        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]

        valid_sets = [train_data]
        valid_names = ['train']

        if X_raw_val is not None:
            X_val = self._prepare_features(X_raw_val, X_physics_val, voc_val)
            y_val = self._prepare_target(y_vmpp_val, voc_val)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

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
        voc: np.ndarray
    ) -> np.ndarray:
        """Predict Vmpp values."""
        X = self._prepare_features(X_raw, X_physics, voc)
        ratio = self.model.predict(X)
        return self._inverse_target(ratio, voc)

    def predict_ratio(
        self,
        X_raw: np.ndarray,
        X_physics: np.ndarray,
        voc: np.ndarray
    ) -> np.ndarray:
        """Predict Vmpp/Voc ratio directly."""
        X = self._prepare_features(X_raw, X_physics, voc)
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


class JmppLGBM:
    """
    LightGBM model for Jmpp prediction.

    Similar approach: predict Jmpp/Jsc ratio.
    """

    def __init__(self, config: VmppLGBMConfig):
        self.config = config
        self.model = None

    def _prepare_features(
        self,
        raw_params: np.ndarray,
        physics_features: np.ndarray,
        jsc: np.ndarray,
        vmpp: np.ndarray
    ) -> np.ndarray:
        """Include both Jsc and Vmpp as features."""
        features_list = [raw_params]

        if self.config.use_physics_features:
            features_list.append(physics_features)

        # Add Jsc and Vmpp features
        extra_features = np.column_stack([
            jsc,
            np.log10(np.abs(jsc) + 1e-30),
            vmpp,
            np.log10(vmpp + 1e-30),
        ])
        features_list.append(extra_features)

        return np.hstack(features_list)

    def _prepare_target(self, jmpp: np.ndarray, jsc: np.ndarray) -> np.ndarray:
        """Prepare target as Jmpp/Jsc ratio."""
        ratio = jmpp / (jsc + 1e-30)
        return np.clip(ratio, 0, 1.5)

    def _inverse_target(self, ratio: np.ndarray, jsc: np.ndarray) -> np.ndarray:
        """Convert ratio back to Jmpp."""
        return ratio * jsc

    def fit(
        self,
        X_raw: np.ndarray,
        X_physics: np.ndarray,
        y_jmpp: np.ndarray,
        jsc: np.ndarray,
        vmpp: np.ndarray,
        X_raw_val: np.ndarray = None,
        X_physics_val: np.ndarray = None,
        y_jmpp_val: np.ndarray = None,
        jsc_val: np.ndarray = None,
        vmpp_val: np.ndarray = None
    ) -> dict:
        """Train the model."""
        X_train = self._prepare_features(X_raw, X_physics, jsc, vmpp)
        y_train = self._prepare_target(y_jmpp, jsc)

        train_data = lgb.Dataset(X_train, label=y_train)

        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]

        valid_sets = [train_data]
        valid_names = ['train']

        if X_raw_val is not None:
            X_val = self._prepare_features(X_raw_val, X_physics_val, jsc_val, vmpp_val)
            y_val = self._prepare_target(y_jmpp_val, jsc_val)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

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
        jsc: np.ndarray,
        vmpp: np.ndarray
    ) -> np.ndarray:
        """Predict Jmpp values."""
        X = self._prepare_features(X_raw, X_physics, jsc, vmpp)
        ratio = self.model.predict(X)
        return self._inverse_target(ratio, jsc)

    def save(self, path: str):
        """Save model to file."""
        self.model.save_model(path)

    def load(self, path: str):
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)


class FFLGBM:
    """
    LightGBM model for Fill Factor prediction.
    FF = Pmpp / (Voc * Jsc)

    Directly predict FF in [0, 1] range.
    """

    def __init__(self, config: VmppLGBMConfig):
        self.config = config
        self.model = None

    def _prepare_features(
        self,
        raw_params: np.ndarray,
        physics_features: np.ndarray,
        voc: np.ndarray,
        jsc: np.ndarray
    ) -> np.ndarray:
        """Include Voc and Jsc as features."""
        features_list = [raw_params]

        if self.config.use_physics_features:
            features_list.append(physics_features)

        extra_features = np.column_stack([
            voc,
            np.log10(voc + 1e-30),
            jsc,
            np.log10(np.abs(jsc) + 1e-30),
            voc * jsc,  # Theoretical max power
        ])
        features_list.append(extra_features)

        return np.hstack(features_list)

    def fit(
        self,
        X_raw: np.ndarray,
        X_physics: np.ndarray,
        y_ff: np.ndarray,
        voc: np.ndarray,
        jsc: np.ndarray,
        X_raw_val: np.ndarray = None,
        X_physics_val: np.ndarray = None,
        y_ff_val: np.ndarray = None,
        voc_val: np.ndarray = None,
        jsc_val: np.ndarray = None
    ) -> dict:
        """Train the model."""
        X_train = self._prepare_features(X_raw, X_physics, voc, jsc)
        y_train = np.clip(y_ff, 0, 1)

        train_data = lgb.Dataset(X_train, label=y_train)

        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]

        valid_sets = [train_data]
        valid_names = ['train']

        if X_raw_val is not None:
            X_val = self._prepare_features(X_raw_val, X_physics_val, voc_val, jsc_val)
            y_val = np.clip(y_ff_val, 0, 1)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

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
        voc: np.ndarray,
        jsc: np.ndarray
    ) -> np.ndarray:
        """Predict FF values."""
        X = self._prepare_features(X_raw, X_physics, voc, jsc)
        ff = self.model.predict(X)
        return np.clip(ff, 0, 1)

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model = lgb.Booster(model_file=path)


def build_vmpp_model(config: VmppLGBMConfig) -> VmppLGBM:
    """Factory function."""
    return VmppLGBM(config)


def build_jmpp_model(config: VmppLGBMConfig) -> JmppLGBM:
    """Factory function."""
    return JmppLGBM(config)


def build_ff_model(config: VmppLGBMConfig) -> FFLGBM:
    """Factory function."""
    return FFLGBM(config)
