#!/usr/bin/env python3
"""
Valida os 3 modulos de ponta a ponta com dados sinteticos controlados.
Execute:  python scripts/validar_modulos.py
"""
import json, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from gate3.duckdb_backend import DuckDBBackend
from gate3.calibration import AutoCalibrator, expected_calibration_error
from gate3.synthetic import gerar_e_validar


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sialock_"))

    # ---- dados fake ----
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "cnpj": [f"{i:014d}" for i in range(n)],
        "ncm": rng.choice(["12345678", "87654321", "11111111"], n),
        "pais_origem": rng.choice(["CN", "US", "DE"], n),
        "uf": rng.choice(["SP", "RJ", "MG"], n),
        "vmle_dolar": rng.normal(5000, 1500, n),
        "peso_liquido": rng.gamma(2, 100, n),
        "score": rng.uniform(0, 1, n),
        "suspeito": rng.integers(0, 2, n),
    })
    p = tmp / "dados.parquet"
    df.to_parquet(p)
    print(f"[i] Parquet: {p} ({n} linhas)")

    # ---- 1) DuckDB ----
    print("\n=== 1) DuckDB ===")
    db = DuckDBBackend(temp_dir=str(tmp / "duck"))
    db.register_parquet("v", p)
    print(db.business_kpis("v").to_string(index=False))
    top = db.topk_por_grupo("v", "ncm", "score", k=5)
    print(f"Top-K grupos: {len(top)} linhas")
    db.close()

    # ---- 2) Calibracao ----
    print("\n=== 2) Calibracao ===")
    y = df["suspeito"].values
    p_mal = np.clip(0.5 + 0.6 * (y - 0.5) + rng.normal(0, 0.4, n), 0.01, 0.99)
    ece0 = expected_calibration_error(y, p_mal)
    cal = AutoCalibrator().fit(p_mal, y)
    ece1 = expected_calibration_error(y, cal.calibrate(p_mal))
    print(f"Metodo: {cal.best_name_} | ECE {ece0:.4f} -> {ece1:.4f}")

    # ---- 3) Synthetic + K-Anon ----
    print("\n=== 3) Synthetic + K-Anonymity ===")
    sint, rep = gerar_e_validar(df, ["ncm", "pais_origem", "uf"], k=5)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n[ok] Tudo funcionando. Tmp: {tmp}")


if __name__ == "__main__":
    main()
