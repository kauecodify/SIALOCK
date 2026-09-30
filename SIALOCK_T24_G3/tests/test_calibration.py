import numpy as np
from gate3.calibration import (
    AutoCalibrator, brier_score, expected_calibration_error,
    PlattCalibrator, IsotonicCalibrator, BetaCalibrator, TemperatureScaling,
)


def _dados_mal_calibrados(n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # probabilidades superconfiantes -> mal calibradas
    p = np.clip(0.5 + 0.5 * (y - 0.5) + rng.normal(0, 0.3, n), 0.01, 0.99)
    return p, y


def test_metricas_basicas():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.2])
    assert 0 <= brier_score(y, p) <= 1
    assert 0 <= expected_calibration_error(y, p) <= 1


def test_calibradores_isolados():
    p, y = _dados_mal_calibrados()
    for Cls in (PlattCalibrator, IsotonicCalibrator,
                BetaCalibrator, TemperatureScaling):
        cal = Cls().fit(p, y)
        p_cal = cal.predict(p)
        assert p_cal.shape == p.shape
        assert np.all((p_cal >= 0) & (p_cal <= 1))


def test_auto_calibrator_reduz_ece():
    p, y = _dados_mal_calibrados()
    ece_antes = expected_calibration_error(y, p)
    cal = AutoCalibrator().fit(p, y)
    p_cal = cal.calibrate(p)
    ece_depois = expected_calibration_error(y, p_cal)
    assert ece_depois < ece_antes, f"ECE nao melhorou: {ece_antes}->{ece_depois}"
    assert cal.best_name_ in ("platt", "isotonic", "beta", "temperature")
