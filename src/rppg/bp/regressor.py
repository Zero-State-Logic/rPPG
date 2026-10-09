"""Baseline blood-pressure regressor with honest BHS/AAMI reporting.

A deliberately simple, well-validated baseline: gradient-boosted trees mapping a
BVP morphology feature vector (see :mod:`rppg.bp.features`) to systolic and
diastolic pressure. It prefers XGBoost when installed and falls back to
scikit-learn's :class:`GradientBoostingRegressor`. Reporting follows the
British Hypertension Society (BHS) grading and the AAMI / ISO 81060-2 limits so
results are stated against the accepted clinical bars rather than a flattering
in-house metric. A constant-mean predictor is provided as the sanity floor every
learned model must beat. Pure numpy/sklearn, no torch.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# --- BHS cumulative-error grade thresholds (percent within 5/10/15 mmHg). ---
_BHS_GRADES: tuple[tuple[str, float, float, float], ...] = (
    ("A", 60.0, 85.0, 95.0),
    ("B", 50.0, 75.0, 90.0),
    ("C", 40.0, 65.0, 85.0),
)
# --- AAMI / ISO 81060-2 limits on the error distribution (mmHg). ---
_AAMI_MEAN_LIMIT = 5.0
_AAMI_STD_LIMIT = 8.0
_ERR_TOLERANCES = (5.0, 10.0, 15.0)
_DEFAULT_SEED = 42


def _build_estimator(random_state: int, overrides: dict[str, Any]) -> tuple[Any, str]:
    """Return a fresh single-target regressor and a backend label.

    Tries :class:`xgboost.XGBRegressor`; falls back to
    :class:`sklearn.ensemble.GradientBoostingRegressor`. Both are seeded for
    deterministic fits. ``overrides`` is merged last so callers can tune params.
    """
    shared = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.9, random_state=random_state)
    try:
        from xgboost import XGBRegressor  # noqa: PLC0415 (optional dependency)

        params = {**shared, "objective": "reg:squarederror", **overrides}
        return XGBRegressor(**params), "xgboost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415

        params = {**shared, **overrides}
        return GradientBoostingRegressor(**params), "sklearn_gbr"


class BPRegressor:
    """Gradient-boosted baseline predicting systolic and diastolic pressure.

    Two independent single-target regressors share one feature-imputation state.
    NaN/inf features (e.g. interbeat stats on a short, noisy segment) are filled
    with per-column training medians so the non-NaN-aware sklearn backend stays
    robust.
    """

    def __init__(self, random_state: int = _DEFAULT_SEED, **model_kwargs: Any) -> None:
        """Create (but do not fit) the SBP and DBP estimators."""
        self.random_state = int(random_state)
        self._sbp, self.backend = _build_estimator(self.random_state, model_kwargs)
        self._dbp, _ = _build_estimator(self.random_state, model_kwargs)
        self._fill: np.ndarray | None = None
        self._n_features: int = 0
        self._fitted: bool = False

    def _clean(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        """Validate shape and impute non-finite values with training medians."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        arr = np.where(np.isfinite(arr), arr, np.nan)
        if fit:
            med = np.nanmedian(arr, axis=0)
            self._fill = np.where(np.isfinite(med), med, 0.0)
            self._n_features = arr.shape[1]
        elif self._fill is None:
            raise RuntimeError("BPRegressor must be fit before predict.")
        elif arr.shape[1] != self._n_features:
            raise ValueError(
                f"X has {arr.shape[1]} features; model was fit with {self._n_features}.")
        nan_rows, nan_cols = np.where(np.isnan(arr))
        arr[nan_rows, nan_cols] = np.take(self._fill, nan_cols)
        return arr

    @staticmethod
    def _as_targets(y: np.ndarray, n: int, name: str) -> np.ndarray:
        """Coerce a label array to a finite 1-D vector of the expected length."""
        vec = np.asarray(y, dtype=np.float64).ravel()
        if vec.shape[0] != n:
            raise ValueError(f"{name} length {vec.shape[0]} != X samples {n}.")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"{name} contains non-finite values.")
        return vec

    def fit(self, X: np.ndarray, y_sbp: np.ndarray, y_dbp: np.ndarray) -> "BPRegressor":
        """Fit both targets on feature matrix ``X``; returns ``self``."""
        feats = self._clean(X, fit=True)
        n = feats.shape[0]
        self._sbp.fit(feats, self._as_targets(y_sbp, n, "y_sbp"))
        self._dbp.fit(feats, self._as_targets(y_dbp, n, "y_dbp"))
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(sbp_pred, dbp_pred)`` for the rows of ``X``."""
        if not self._fitted:
            raise RuntimeError("BPRegressor must be fit before predict.")
        feats = self._clean(X, fit=False)
        sbp = np.asarray(self._sbp.predict(feats), dtype=np.float64)
        dbp = np.asarray(self._dbp.predict(feats), dtype=np.float64)
        return sbp, dbp


def _bhs_grade(within_5: float, within_10: float, within_15: float) -> str:
    """Return the BHS letter grade; a grade needs all three cumulative bars."""
    for grade, t5, t10, t15 in _BHS_GRADES:
        if within_5 >= t5 and within_10 >= t10 and within_15 >= t15:
            return grade
    return "D"


def bhs_aami_report(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    """Score BP predictions against the BHS grade and AAMI limits.

    Args:
        pred: Predicted pressures (mmHg).
        true: Reference pressures (mmHg), same length as ``pred``.

    Returns:
        Dict with ``mean_err``, ``std_err`` (sample SD, mmHg), ``mae``, the
        cumulative ``within_5``/``within_10``/``within_15`` percentages,
        ``aami_pass`` (|mean_err| <= 5 and std_err <= 8) and ``bhs_grade``.
    """
    pred_arr = np.asarray(pred, dtype=np.float64).ravel()
    true_arr = np.asarray(true, dtype=np.float64).ravel()
    if pred_arr.shape != true_arr.shape:
        raise ValueError(f"pred/true shape mismatch: {pred_arr.shape} vs {true_arr.shape}.")
    if pred_arr.size == 0:
        raise ValueError("pred/true must be non-empty.")

    err = pred_arr - true_arr
    abs_err = np.abs(err)
    mean_err = float(err.mean())
    std_err = float(err.std(ddof=1)) if err.size > 1 else 0.0
    within = {tol: float(np.mean(abs_err <= tol) * 100.0) for tol in _ERR_TOLERANCES}
    aami_pass = bool(abs(mean_err) <= _AAMI_MEAN_LIMIT and std_err <= _AAMI_STD_LIMIT)

    return {
        "mean_err": mean_err,
        "std_err": std_err,
        "mae": float(abs_err.mean()),
        "within_5": within[5.0],
        "within_10": within[10.0],
        "within_15": within[15.0],
        "aami_pass": aami_pass,
        "bhs_grade": _bhs_grade(within[5.0], within[10.0], within[15.0]),
    }


def mean_predictor_mae(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """MAE of the constant baseline that always predicts the training mean.

    This is the floor any learned regressor must beat; a model worse than this
    has learned nothing useful from the features.
    """
    train = np.asarray(y_train, dtype=np.float64).ravel()
    test = np.asarray(y_test, dtype=np.float64).ravel()
    if train.size == 0 or test.size == 0:
        raise ValueError("y_train and y_test must be non-empty.")
    baseline = float(train.mean())
    return float(np.mean(np.abs(test - baseline)))
