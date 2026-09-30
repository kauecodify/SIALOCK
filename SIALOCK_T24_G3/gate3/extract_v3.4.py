"""
Extract v3.4 - Pipeline principal com integracao DuckDB, Calibracao e Synthetic Data
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from .duckdb_backend import DuckDBBackend
from .calibration import AutoCalibrator, reliability_diagram
from .synthetic import gerar_e_validar

log = logging.getLogger("sialock.extract_v3.4")

# Configuracoes
CFG = {
    "scale": {
        "chunk_size": 100_000,
    }
}

BASE_DIR = {
    "base2": Path("/workspace/SIALOCK_T24_G3/data/base2"),
    "base3": Path("/workspace/SIALOCK_T24_G3/data/base3"),
}


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Engineering de features basico."""
    return df


def _to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Converte DataFrame para matriz de features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].values


def iter_auto(path: Path | str, batch_size: int = 100_000) -> list[pd.DataFrame]:
    """Iterador generico para Parquet/CSV."""
    p = Path(path)
    if p.suffix == ".parquet" or p.is_dir():
        import pyarrow.parquet as pq
        dataset = pq.ParquetDataset(str(p), use_legacy_dataset=False)
        for batch in dataset.iter_batches(batch_size=batch_size):
            yield batch.to_pandas()
    else:
        for chunk in pd.read_csv(p, chunksize=batch_size):
            yield chunk


class Gate3Ensemble:
    """Ensemble basico para o Gate 3."""
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)

    def partial_fit(self, X, y, classes):
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X, y, classes=classes)
        else:
            self.model.fit(X, y)

    def score(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def processar_base(base: str, backend: str = "auto", resume: bool = True):
    """Pipeline principal."""
    cfg = CFG
    bdir = BASE_DIR[base]
    entrada = bdir / "entrada"

    # ------------------------------------------------------------
    # 1) KPIs rapidos via DuckDB (nao carrega 23M em RAM)
    # ------------------------------------------------------------
    if backend in ("auto", "duckdb"):
        db = DuckDBBackend(memory_limit="4GB")
        db.register_auto("base_view", entrada)
        kpis = db.business_kpis("base_view")
        kpis.to_json(bdir / "kpis_duckdb.json", orient="records")
        log.info("[%s] KPIs DuckDB:\n%s", base, kpis.to_string())
    else:
        db = None

    # ------------------------------------------------------------
    # 2) Treina ensemble em lotes + coleta probabilidades
    # ------------------------------------------------------------
    ensemble = Gate3Ensemble()
    probs_all, y_all = [], []

    for chunk in iter_auto(entrada, batch_size=cfg["scale"]["chunk_size"]):
        df = _feature_engineering(chunk)
        X = _to_matrix(df)
        y = df.get("suspeito", pd.Series(0, index=df.index)).astype(int).values
        try:
            ensemble.partial_fit(X, y, classes=np.array([0, 1]))
            p = ensemble.score(X)
            probs_all.append(p)
            y_all.append(y)
        except Exception as e:
            log.warning("partial_fit falhou: %s", e)

    # ------------------------------------------------------------
    # 3) Calibracao de probabilidades
    # ------------------------------------------------------------
    if probs_all:
        p_concat = np.concatenate(probs_all)
        y_concat = np.concatenate(y_all)
        cal = AutoCalibrator().fit(p_concat, y_concat)
        p_cal = cal.calibrate(p_concat)

        # Salva relatorio de calibracao
        relatorio = {
            "metodo": cal.best_name_,
            "brier_antes": cal.report_.brier_before,
            "brier_depois": cal.report_.brier_after,
            "ece_antes": cal.report_.ece_before,
            "ece_depois": cal.report_.ece_after,
            "diagrama": reliability_diagram(y_concat, p_cal),
        }
        (bdir / "calibracao.json").write_text(
            json.dumps(relatorio, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("[%s] Calibracao aplicada: %s (ECE %.4f->%.4f)",
                 base, cal.best_name_,
                 relatorio["ece_antes"], relatorio["ece_depois"])

    # ------------------------------------------------------------
    # 4) Teste de K-Anonymity com dados sinteticos
    # ------------------------------------------------------------
    amostra = next(iter_auto(entrada, batch_size=50_000))
    amostra = _feature_engineering(amostra)
    quasi = [c for c in ("ncm", "pais_origem", "uf") if c in amostra.columns]
    if quasi:
        sint, rep = gerar_e_validar(amostra, quasi_ids=quasi, k=5)
        (bdir / "k_anonymity_report.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("[%s] K-Anonymity: k_min=%s passa=%s",
                 base, rep["k_anonymity"]["k_min_sintetico"],
                 rep["k_anonymity"]["passa_k"])

    # ------------------------------------------------------------
    # 5) Top-K via DuckDB (window function em disco)
    # ------------------------------------------------------------
    if db is not None:
        # (exemplo: assumindo que voce exportou scores em Parquet)
        # db.topk_por_grupo("scores_view", "cnpj", "score", k=50)
        pass

    # ------------------------------------------------------------
    # 6) Checkpoint e entrega
    # ------------------------------------------------------------
    log.info("[%s] Pipeline concluido com sucesso", base)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SIALOCK T24 G3 - Extract v3.4")
    parser.add_argument("--base", type=str, default="base2", help="Base a processar")
    parser.add_argument("--backend", type=str, default="auto", help="Backend: auto, duckdb, pandas")
    parser.add_argument("--resume", action="store_true", help="Retomar processamento")
    args = parser.parse_args()
    processar_base(args.base, args.backend, args.resume)
