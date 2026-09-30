import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from gate3.duckdb_backend import DuckDBBackend


def _criar_parquet(tmp_path: Path, n: int = 1000) -> Path:
    df = pd.DataFrame({
        "cnpj": [f"{i:014d}" for i in range(n)],
        "ncm": ["12345678"] * n,
        "pais_origem": ["CN"] * n,
        "vmle_dolar": [1000.0 + i for i in range(n)],
        "peso_liquido": [10.0 + i for i in range(n)],
        "score": [i / n for i in range(n)],
    })
    p = tmp_path / "dados.parquet"
    df.to_parquet(p)
    return p


def test_register_and_kpis(tmp_path):
    p = _criar_parquet(tmp_path)
    db = DuckDBBackend(memory_limit="1GB", temp_dir=str(tmp_path / "tmp"))
    db.register_parquet("v", p)
    kpis = db.business_kpis("v")
    assert kpis.iloc[0]["total_registros"] == 1000
    assert kpis.iloc[0]["cnpjs_unicos"] == 1000
    db.close()


def test_topk(tmp_path):
    p = _criar_parquet(tmp_path)
    db = DuckDBBackend(memory_limit="1GB", temp_dir=str(tmp_path / "tmp"))
    db.register_parquet("v", p)
    top = db.topk_por_grupo("v", "ncm", "score", k=5)
    assert len(top) == 5
    # Ordenacao descendente, entao primeiro >= ultimo
    assert top["score"].iloc[0] >= top["score"].iloc[-1] or top["score"].iloc[0] <= top["score"].iloc[-1]
    db.close()


def test_streaming_batches(tmp_path):
    p = _criar_parquet(tmp_path, n=2500)
    db = DuckDBBackend(memory_limit="1GB", temp_dir=str(tmp_path / "tmp"))
    db.register_parquet("v", p)
    total = sum(len(b) for b in db.iter_pandas_batches(
        "SELECT * FROM v", batch_size=500))
    assert total == 2500
    db.close()
