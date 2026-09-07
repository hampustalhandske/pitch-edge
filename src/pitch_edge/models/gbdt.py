"""Gradient-boosted trees on the pre-match feature store.

LightGBM is used when it is importable (`uv sync --extra lgbm`; needs an
arm64 libomp on Apple Silicon); otherwise scikit-learn's
HistGradientBoostingClassifier — the same histogram-GBDT algorithm family —
is used so results are reproducible on any machine. The model card records
which backend produced a given run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pitch_edge.features.build import feature_columns
from pitch_edge.models.base import MatchModel, to_frame

try:  # pragma: no cover - environment dependent
    import lightgbm as lgb

    _HAS_LGBM = True
except Exception:  # noqa: BLE001
    lgb = None
    _HAS_LGBM = False


# Feature groups that FAILED the pre-registered ablation criterion (CASE_STUDY.md Result 7: removing them
# improved out-of-sample log-loss by ~0.002). They stay in the feature store and in the ablation, but the
# default production model does not read them. Pass exclude_prefixes=() to include everything.
DEFAULT_EXCLUDED_PREFIXES: tuple[str, ...] = ("pv_", "rot_")

# Feature groups that are new and NOT yet ablated against the pre-registered criterion (CASE_STUDY.md
# Result 8: squad-value / confirmed-lineup signal, Phase 6). Excluded from the default `gbdt`/`gbdt_mkt`
# instances so those two models' inputs — and every historical number reported for them — stay byte-for-
# byte reproducible while the new group is being tested through a separate, explicitly-named candidate
# model (`models/__init__.py::available_models`).
PENDING_EXCLUDED_PREFIXES: tuple[str, ...] = ("sv_",)


class GBDTMatchModel(MatchModel):
    def __init__(
        self,
        include_market: bool = False,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        random_state: int = 42,
        prefer_lightgbm: bool = True,
        exclude_prefixes: tuple[str, ...] | None = None,
        name_suffix: str = "",
    ):
        self.include_market = include_market
        self.uses_market_features = include_market
        self.exclude_prefixes = tuple(
            (*DEFAULT_EXCLUDED_PREFIXES, *PENDING_EXCLUDED_PREFIXES) if exclude_prefixes is None else exclude_prefixes
        )
        self.name = ("gbdt_mkt" if include_market else "gbdt") + name_suffix
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.backend = "lightgbm" if (_HAS_LGBM and prefer_lightgbm) else "sklearn_hgb"
        self._cols: list[str] = []
        self._clf: Any = None
        self._medians: pd.Series | None = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self._cols).astype(float)
        if self._medians is not None:
            X = X.fillna(self._medians)
        return X.to_numpy()

    def fit(self, train: pd.DataFrame) -> GBDTMatchModel:
        candidate = feature_columns(train, include_market=self.include_market)
        if self.exclude_prefixes:
            candidate = [c for c in candidate if not c.startswith(self.exclude_prefixes)]
        X_all = train[candidate].astype(float)
        # sklearn's histogram binner cannot handle all-NaN or constant columns; drop them per fit
        usable = [c for c in candidate if X_all[c].notna().any() and X_all[c].nunique(dropna=True) > 1]
        self._cols = usable
        X_df = X_all[usable]
        self._medians = X_df.median()
        X = X_df.fillna(self._medians).to_numpy()
        y = train["result"].to_numpy().astype(int)
        if self.backend == "lightgbm":
            self._clf = lgb.LGBMClassifier(
                objective="multiclass",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=15,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self._clf = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                l2_regularization=1.0,
                random_state=self.random_state,
                early_stopping=False,
            )
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._clf is None:
            raise RuntimeError("fit() first")
        proba = self._clf.predict_proba(self._matrix(X))
        classes = list(getattr(self._clf, "classes_", [0, 1, 2]))
        ordered = np.zeros((len(X), 3))
        for j, cls in enumerate(classes):
            ordered[:, int(cls)] = proba[:, j]
        return to_frame(ordered, X.index)

    def feature_importance(self) -> pd.Series:
        if self._clf is None:
            return pd.Series(dtype=float)
        if hasattr(self._clf, "feature_importances_"):
            return pd.Series(self._clf.feature_importances_, index=self._cols).sort_values(ascending=False)
        return pd.Series(dtype=float)

    def card(self) -> dict:
        return {
            **super().card(),
            "family": "histogram gradient-boosted decision trees (multiclass)",
            "backend": self.backend,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "features": self._cols,
            "excluded_feature_prefixes": list(self.exclude_prefixes),
            "leakage_checks": [
                "all rolling features shifted by one match",
                "Elo computed sequentially pre-match",
                "closing odds never used; early odds only in *_mkt variant",
            ],
        }
