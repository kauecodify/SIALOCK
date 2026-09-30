import numpy as np
import pandas as pd
from gate3.synthetic import (
    GaussianCopulaGenerator, KAnonymityValidator,
    gerar_e_validar, distancia_mais_proximo_vizinho, taxa_divulgacao,
)


def _df_exemplo(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ncm": rng.choice(["12345678", "87654321", "11111111"], n),
        "pais_origem": rng.choice(["CN", "US", "DE"], n),
        "uf": rng.choice(["SP", "RJ", "MG"], n),
        "vmle_dolar": rng.normal(5000, 1500, n),
        "peso_liquido": rng.gamma(2, 100, n),
    })


def test_copula_preserva_marginais():
    df = _df_exemplo()
    gen = GaussianCopulaGenerator(seed=0).fit(df)
    sint = gen.sample(len(df))
    assert set(sint.columns) == set(df.columns)
    assert len(sint) == len(df)
    # medias proximas
    assert abs(sint["vmle_dolar"].mean() - df["vmle_dolar"].mean()) < 500


def test_copula_preserva_categorias():
    df = _df_exemplo()
    sint = GaussianCopulaGenerator(seed=0).fit(df).sample(len(df))
    for c in ("ncm", "pais_origem", "uf"):
        assert set(sint[c]).issubset(set(df[c]))


def test_k_anonymity_validador():
    df = _df_exemplo(n=5000)
    sint = GaussianCopulaGenerator(seed=0).fit(df).sample(2000)
    val = KAnonymityValidator(["ncm", "pais_origem", "uf"], k=5)
    rep = val.evaluate(df, sint)
    assert rep.k_min_original > 0
    assert isinstance(rep.passa_k, bool)


def test_gerar_e_validar():
    df = _df_exemplo(n=3000)
    sint, rep = gerar_e_validar(df, ["ncm", "pais_origem", "uf"], k=5)
    assert len(sint) == len(df)
    assert "k_anonymity" in rep
    assert "dcr" in rep
    assert "divulgacao" in rep


def test_dcr_zero_quando_copia():
    """Se sintetico == original, DCR ~ 0 -> detecta vazamento."""
    df = _df_exemplo(n=200)
    d = distancia_mais_proximo_vizinho(
        df, df.copy(), ["vmle_dolar", "peso_liquido"], sample=200
    )
    assert d["dcr_min"] < 1e-6


def test_divulgacao_detecta_grupos_unicos():
    df = pd.DataFrame({
        "ncm": ["1", "2", "3", "4"],
        "score": [0.1, 0.2, 0.3, 0.4],
    })
    sint = pd.DataFrame({"ncm": ["1", "2"], "score": [0.1, 0.2]})
    d = taxa_divulgacao(df, sint, ["ncm"])
    assert d["taxa_divulgacao_pct"] == 100.0
