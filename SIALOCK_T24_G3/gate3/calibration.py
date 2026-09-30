"""
Calibracao de probabilidades para o ensemble Gate 3.

Metodos implementados:
- Isotonic Regression (nao-parametrico, monotono)
- Platt Scaling (sigmoid parametrico)
- Temperature Scaling (multi-classe)
- Beta Calibration (mais flexivel que Platt)

Metricas:
- Brier Score
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Reliability diagram
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin

log = logging.getLogger("sialock.calibration")


# ============================================================
# Metricas
# ============================================================
def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 15
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 15
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        mce = max(mce, abs(y_true[mask].mean() - p[mask].mean()))
    return float(mce)


def reliability_diagram(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 15
) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        accs.append(float(y_true[mask].mean()))
        confs.append(float(p[mask].mean()))
        counts.append(int(mask.sum()))
    return {"bin_center": centers, "accuracy": accs,
            "confidence": confs, "count": counts}


# ============================================================
# Calibradores
# ============================================================
class PlattCalibrator(BaseEstimator, RegressorMixin):
    """Platt Scaling — sigmoide logistico."""
    def __init__(self):
        self.lr = LogisticRegression(C=1e10, solver="lbfgs")

    def fit(self, p: np.ndarray, y: np.ndarray):
        self.lr.fit(p.reshape(-1, 1), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(p.reshape(-1, 1))[:, 1]


class IsotonicCalibrator(BaseEstimator, RegressorMixin):
    """Isotonic Regression — monotono nao-parametrico."""
    def __init__(self, out_of_bounds: str = "clip"):
        self.iso = IsotonicRegression(out_of_bounds=out_of_bounds)

    def fit(self, p: np.ndarray, y: np.ndarray):
        self.iso.fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.iso.predict(p)


class BetaCalibrator(BaseEstimator, RegressorMixin):
    """Beta Calibration — mais flexivel que Platt."""
    def __init__(self, max_iter: int = 200):
        self.max_iter = max_iter
        self.a_ = self.b_ = self.c_ = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray):
        from scipy.optimize import minimize

        def nll(params):
            a, b, c = params
            eps = 1e-9
            p_ = np.clip(p, eps, 1 - eps)
            logits = a * np.log(p_) + b * np.log(1 - p_) + c
            q = 1 / (1 + np.exp(-logits))
            q = np.clip(q, eps, 1 - eps)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        res = minimize(nll, x0=[1.0, 1.0, 0.0], method="Nelder-Mead",
                       options={"maxiter": self.max_iter})
        self.a_, self.b_, self.c_ = res.x
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        eps = 1e-9
        p_ = np.clip(p, eps, 1 - eps)
        logits = self.a_ * np.log(p_) + self.b_ * np.log(1 - p_) + self.c_
        return 1 / (1 + np.exp(-logits))


class TemperatureScaling(BaseEstimator, RegressorMixin):
    """Temperature Scaling — 1 parametro, otimo para redes/ensembles."""
    def __init__(self, max_iter: int = 100):
        self.T = 1.0
        self.max_iter = max_iter

    def fit(self, p: np.ndarray, y: np.ndarray):
        from scipy.optimize import minimize
        eps = 1e-9

        def nll(T_arr):
            T = max(T_arr[0], 1e-3)
            p_ = np.clip(p, eps, 1 - eps)
            logit = np.log(p_ / (1 - p_)) / T
            q = 1 / (1 + np.exp(-logit))
            q = np.clip(q, eps, 1 - eps)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        res = minimize(nll, x0=[1.0], method="Nelder-Mead",
                       options={"maxiter": self.max_iter})
        self.T = max(res.x[0], 1e-3)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        eps = 1e-9
        p_ = np.clip(p, eps, 1 - eps)
        logit = np.log(p_ / (1 - p_)) / self.T
        return 1 / (1 + np.exp(-logit))


# ============================================================
# Auto-calibracao (escolhe o melhor por ECE)
# ============================================================
@dataclass
class CalibrationReport:
    method: str
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    mce_before: float
    mce_after: float
    temperature: float = 1.0
    params: dict = field(default_factory=dict)


class AutoCalibrator:
    """Testa 4 metodos e escolhe o de menor ECE."""

    METHODS = Literal["platt", "isotonic", "beta", "temperature"]

    def __init__(self, candidates: list[str] | None = None,
                 n_bins: int = 15):
        self.candidates = candidates or ["platt", "isotonic", "beta", "temperature"]
        self.n_bins = n_bins
        self.best_: BaseEstimator | None = None
        self.best_name_: str = ""
        self.report_: CalibrationReport | None = None

    def _build(self, name: str) -> BaseEstimator:
        return {
            "platt": PlattCalibrator(),
            "isotonic": IsotonicCalibrator(),
            "beta": BetaCalibrator(),
            "temperature": TemperatureScaling(),
        }[name]

    def fit(self, p: np.ndarray, y: np.ndarray) -> "AutoCalibrator":
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=int)

        b_brier = brier_score(y, p)
        b_ece = expected_calibration_error(y, p, self.n_bins)
        b_mce = maximum_calibration_error(y, p, self.n_bins)

        best_ece = float("inf")
        for name in self.candidates:
            try:
                cal = self._build(name).fit(p, y)
                p_cal = cal.predict(p)
                ece = expected_calibration_error(y, p_cal, self.n_bins)
                if ece < best_ece:
                    best_ece = ece
                    self.best_ = cal
                    self.best_name_ = name
            except Exception as e:
                log.warning("Calibrador %s falhou: %s", name, e)

        if self.best_ is None:
            raise RuntimeError("Nenhum calibrador convergiu.")

        p_final = self.best_.predict(p)
        self.report_ = CalibrationReport(
            method=self.best_name_,
            brier_before=b_brier,
            brier_after=brier_score(y, p_final),
            ece_before=b_ece,
            ece_after=expected_calibration_error(y, p_final, self.n_bins),
            mce_before=b_mce,
            mce_after=maximum_calibration_error(y, p_final, self.n_bins),
            temperature=getattr(self.best_, "T", 1.0),
        )
        log.info(
            "Calibracao: %s | ECE %.4f -> %.4f | Brier %.4f -> %.4f",
            self.best_name_,
            self.report_.ece_before, self.report_.ece_after,
            self.report_.brier_before, self.report_.brier_after,
        )
        return self

    def calibrate(self, p: np.ndarray) -> np.ndarray:
        if self.best_ is None:
            raise RuntimeError("Chame fit() primeiro.")
        return self.best_.predict(np.asarray(p, dtype=float))

    def reliability(self, p: np.ndarray, y: np.ndarray) -> dict:
        return reliability_diagram(y, self.calibrate(p), self.n_bins)
