"""
DuckDB Backend — agregações out-of-core sobre 23M+ linhas.

Vantagens sobre Pandas:
- Lê Parquet/CSV direto do disco (zero RAM para scan completo)
- SQL com window functions, GROUP BY, JOIN em disco
- Spill-to-disk automático (memory_limit configurável)
- Threads nativas (usa todos os cores)
- Streaming Arrow zero-copy para o ensemble ML
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator, Optional
import duckdb
import pyarrow as pa
import pandas as pd

log = logging.getLogger("sialock.duckdb")


class DuckDBBackend:
    """Wrapper DuckDB para pipelines SIALOCK."""

    def __init__(
        self, db_path: str = ":memory:",
        memory_limit: str = "4GB",
        threads: Optional[int] = None,
        temp_dir: str = "./.duckdb_tmp",
    ):
        self.conn = duckdb.connect(db_path)
        self.conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
        if threads:
            self.conn.execute(f"PRAGMA threads={threads}")
        self.conn.execute(f"PRAGMA temp_directory='{temp_dir}'")
        self.conn.execute("PRAGMA enable_progress_bar=false")
        self.conn.execute("PRAGMA preserve_insertion_order=false")
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        log.info("DuckDB pronto: mem=%s threads=%s tmp=%s",
                 memory_limit, threads or "auto", temp_dir)

    # ---------- Registro de fontes ----------

    def register_parquet(self, view: str, path: Path | str, **opts) -> None:
        """Registra view sobre Parquet (glob suportado)."""
        p = str(path)
        self.conn.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_parquet('{p}', union_by_name=true, "
            f"filename=true)"
        )
        log.info("View '%s' -> %s", view, p)

    def register_csv(self, view: str, path: Path | str,
                     sep: str = ";", header: bool = True) -> None:
        self.conn.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_csv_auto('{path}', delim='{sep}', "
            f"header={str(header).lower()}, sample_size=-1)"
        )

    def register_auto(self, view: str, path: Path | str) -> None:
        p = Path(path)
        if p.is_dir():
            if list(p.glob("*.parquet")):
                self.register_parquet(view, p / "*.parquet")
            else:
                self.register_csv(view, p / "*.csv")
        elif p.suffix == ".parquet":
            self.register_parquet(view, p)
        else:
            self.register_csv(view, p)

    # ---------- KPIs de negocio (SQL) ----------

    def business_kpis(self, view: str) -> pd.DataFrame:
        """KPIs agregados direto no DuckDB (nao carrega em RAM)."""
        sql = f"""
        SELECT
            COUNT(*)                                      AS total_registros,
            COUNT(DISTINCT cnpj)                          AS cnpjs_unicos,
            COUNT(DISTINCT ncm)                           AS ncms_distintos,
            COUNT(DISTINCT pais_origem)                   AS paises_distintos,
            AVG(TRY_CAST(vmle_dolar AS DOUBLE))           AS vmle_medio,
            SUM(TRY_CAST(vmle_dolar AS DOUBLE))           AS vmle_total,
            AVG(TRY_CAST(peso_liquido AS DOUBLE))         AS peso_medio,
            APPROX_QUANTILE(TRY_CAST(vmle_dolar AS DOUBLE), 0.95) AS vmle_p95,
            APPROX_QUANTILE(TRY_CAST(vmle_dolar AS DOUBLE), 0.99) AS vmle_p99
        FROM {view}
        """
        return self.conn.execute(sql).fetchdf()

    def topk_por_grupo(self, view: str, group_col: str = "cnpj",
                       score_col: str = "score", k: int = 50) -> pd.DataFrame:
        """Top-K por grupo via QUALIFY (window function em disco)."""
        sql = f"""
        SELECT * FROM {view}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY {group_col} ORDER BY {score_col} DESC NULLS LAST
        ) <= {k}
        """
        return self.conn.execute(sql).fetchdf()

    def outliers_iqr(self, view: str, col: str) -> pd.DataFrame:
        sql = f"""
        WITH stats AS (
            SELECT
                APPROX_QUANTILE(TRY_CAST({col} AS DOUBLE), 0.25) AS q1,
                APPROX_QUANTILE(TRY_CAST({col} AS DOUBLE), 0.75) AS q3
            FROM {view}
        )
        SELECT t.*, (q3 - q1) AS iqr,
               (q3 + 1.5*(q3-q1)) AS limite_superior
        FROM {view} t, stats
        WHERE TRY_CAST({col} AS DOUBLE) > (q3 + 1.5*(q3-q1))
           OR TRY_CAST({col} AS DOUBLE) < (q1 - 1.5*(q3-q1))
        """
        return self.conn.execute(sql).fetchdf()

    # ---------- Streaming Arrow para ML ----------

    def iter_arrow_batches(
        self, sql: str, batch_size: int = 200_000
    ) -> Iterator[pa.RecordBatch]:
        """Streaming Arrow zero-copy (ideal para partial_fit)."""
        reader = self.conn.execute(sql).to_arrow_reader()
        while True:
            try:
                batch = reader.read_next_batch()
                if batch is None:
                    break
                yield batch
            except StopIteration:
                break

    def iter_pandas_batches(
        self, sql: str, batch_size: int = 200_000
    ) -> Iterator[pd.DataFrame]:
        for batch in self.iter_arrow_batches(sql, batch_size):
            yield batch.to_pandas()

    # ---------- Persistencia ----------

    def export_parquet(self, sql: str, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        self.conn.execute(
            f"COPY ({sql}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        log.info("Exportado: %s", out)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
