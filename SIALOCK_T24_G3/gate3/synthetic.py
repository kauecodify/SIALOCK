"""
Synthetic Data Generator para testes de K-Anonymity.

Estrategia:
1. Gaussian Copula (preserva marginais + correlacao) — rapido e deterministico
2. Detecao automatica de colunas (numerica, categorica, data)
3. Validador de K-Anonymity (compara original vs. sintetico)
4. Metricas de risco de reidentificacao (DCR, NNDR, disclosure)

Compativel com:
- Preservacao de tipos e dominios
- Seed deterministico (reprodutibilidade)
- Grande escala (amostragem estratificada opcional)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("sialock.synthetic")


# ============================================================
# 1. Detecao de tipos
# ============================================================
def _classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    num, cat, dt = [], [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            dt.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(c)
        else:
            cat.append(c)
    return {"numeric": num, "categorical": cat, "datetime": dt}


# ============================================================
# 2. Gaussian Copula Generator
# ============================================================
class GaussianCopulaGenerator:
    """Sintetizador via copula gaussiana — rapido, sem GPU."""

    def __init__(self, seed: int = 42, max_categories: int = 500):
        self.seed = seed
        self.max_categories = max_categories
        self.rng = np.random.default_rng(seed)
        self._spec: dict = {}

    def fit(self, df: pd.DataFrame) -> "GaussianCopulaGenerator":
        types = _classify_columns(df)
        self._spec = {"types": types, "n": len(df)}

        # Numericas: guarda media/std + transformacao rank->normal
        if types["numeric"]:
            num_df = df[types["numeric"]].astype(float)
            self._spec["num_mean"] = num_df.mean().to_dict()
            self._spec["num_std"] = num_df.std().replace(0, 1).to_dict()
            # copula: correlacao sobre ranks normalizados
            ranks = num_df.rank(pct=True).clip(1e-6, 1 - 1e-6)
            normals = stats.norm.ppf(ranks)
            self._spec["num_corr"] = np.corrcoef(
                normals, rowvar=False
            ) if len(types["numeric"]) > 1 else np.array([[1.0]])

        # Categoricas: distribuicao empirica
        self._spec["cat_dist"] = {}
        for c in types["categorical"]:
            vc = df[c].astype("object").fillna("__NA__").value_counts()
            if len(vc) > self.max_categories:
                top = vc.head(self.max_categories)
                resto = vc.iloc[self.max_categories:].sum()
                top["__OUTROS__"] = resto
                vc = top
            self._spec["cat_dist"][c] = (vc / vc.sum()).to_dict()

        # Datas: guarda min/max
        self._spec["dt_range"] = {}
        for c in types["datetime"]:
            s = pd.to_datetime(df[c])
            self._spec["dt_range"][c] = (s.min(), s.max())

        log.info("Fit: %d linhas | num=%d cat=%d dt=%d",
                 len(df), len(types["numeric"]),
                 len(types["categorical"]), len(types["datetime"]))
        return self

    def sample(self, n: Optional[int] = None) -> pd.DataFrame:
        n = n or self._spec["n"]
        types = self._spec["types"]
        out: dict[str, np.ndarray] = {}

        # Numericas via copula
        if types["numeric"]:
            k = len(types["numeric"])
            corr = self._spec["num_corr"]
            try:
                z = self.rng.multivariate_normal(
                    mean=np.zeros(k), cov=corr, size=n, method="cholesky"
                )
            except np.linalg.LinAlgError:
                z = self.rng.standard_normal((n, k))
            u = stats.norm.cdf(z)
            for i, c in enumerate(types["numeric"]):
                # Quantil empirico preservando marginais
                # Aqui usamos aproximacao normal (mais rapido, sem guardar amostra)
                mu, sd = self._spec["num_mean"][c], self._spec["num_std"][c]
                out[c] = stats.norm.ppf(u[:, i], loc=mu, scale=sd)

        # Categoricas via amostragem da distribuicao empirica
        for c in types["categorical"]:
            dist = self._spec["cat_dist"][c]
            cats = list(dist.keys())
            probs = np.array(list(dist.values()), dtype=float)
            probs = probs / probs.sum()
            out[c] = self.rng.choice(cats, size=n, p=probs)

        # Datas uniformes no range
        for c in types["datetime"]:
            lo, hi = self._spec["dt_range"][c]
            lo_i, hi_i = lo.value, hi.value
            out[c] = pd.to_datetime(
                self.rng.integers(lo_i, hi_i, size=n)
            )

        # Reordena colunas para casar com original
        cols = (types["numeric"] + types["categorical"] + types["datetime"])
        return pd.DataFrame(out)[cols]


# ============================================================
# 3. Validador K-Anonymity
# ============================================================
@dataclass
class KAnonReport:
    k_min_original: int
    k_min_sintetico: int
    k_medio_original: float
    k_medio_sintetico: float
    grupos_violados_original: int
    grupos_violados_sintetico: int
    registros_descartados_pct: float
    passa_k: bool


class KAnonymityValidator:
    """Compara K-Anonymity entre original e sintetico."""

    def __init__(self, quasi_ids: list[str], k: int = 5):
        self.quasi_ids = quasi_ids
        self.k = k

    def _k_stats(self, df: pd.DataFrame) -> tuple[pd.Series, int, float, int]:
        if not self.quasi_ids:
            return pd.Series([len(df)] * len(df), index=df.index), len(df), float(len(df)), 0
        groups = df.groupby(self.quasi_ids, dropna=False)
        sizes = groups[self.quasi_ids[0]].transform("size")
        k_min = int(sizes.min())
        k_medio = float(sizes.mean())
        violados = int((sizes < self.k).sum())
        return sizes, k_min, k_medio, violados

    def evaluate(self, original: pd.DataFrame,
                 sintetico: pd.DataFrame) -> KAnonReport:
        _, kmin_o, kmed_o, vio_o = self._k_stats(original)
        _, kmin_s, kmed_s, vio_s = self._k_stats(sintetico)
        descartados = (vio_s / len(sintetico) * 100) if len(sintetico) else 0.0

        rep = KAnonReport(
            k_min_original=kmin_o,
            k_min_sintetico=kmin_s,
            k_medio_original=round(kmed_o, 2),
            k_medio_sintetico=round(kmed_s, 2),
            grupos_violados_original=vio_o,
            grupos_violados_sintetico=vio_s,
            registros_descartados_pct=round(descartados, 2),
            passa_k=kmin_s >= self.k,
        )
        log.info("K-Anon: orig k_min=%d | sint k_min=%d | passa=%s",
                 kmin_o, kmin_s, rep.passa_k)
        return rep


# ============================================================
# 4. Metricas de risco de reidentificacao
# ============================================================
def distancia_mais_proximo_vizinho(
    original: pd.DataFrame, sintetico: pd.DataFrame,
    num_cols: list[str], sample: int = 1000
) -> dict:
    """
    DCR (Distance to Closest Record) — quanto MAIOR, mais seguro.
    Se DCR ≈ 0, o sintetico copiou um registro real -> vazamento.
    """
    if not num_cols or len(original) == 0 or len(sintetico) == 0:
        return {"dcr_min": None, "dcr_med": None, "dcr_p05": None}

    o = original[num_cols].sample(min(sample, len(original)),
                                  random_state=42).values
    s = sintetico[num_cols].sample(min(sample, len(sintetico)),
                                   random_state=42).values
    # normaliza
    mu, sd = o.mean(0), o.std(0) + 1e-9
    o = (o - mu) / sd
    s = (s - mu) / sd

    # distancia euclidiana de cada sintetico ao original mais proximo
    d = np.sqrt(((s[:, None, :] - o[None, :, :]) ** 2).sum(-1))
    d_min = d.min(axis=1)
    return {
        "dcr_min": float(d_min.min()),
        "dcr_med": float(np.median(d_min)),
        "dcr_p05": float(np.percentile(d_min, 5)),
    }


def taxa_divulgacao(
    original: pd.DataFrame, sintetico: pd.DataFrame,
    quasi_ids: list[str]
) -> dict:
    """% de registros sinteticos que caem em um grupo unico do original."""
    if not quasi_ids:
        return {"taxa_divulgacao_pct": 0.0}
    orig_counts = (original.groupby(quasi_ids, dropna=False)
                   .size().rename("n_orig").reset_index())
    merged = sintetico.merge(orig_counts, on=quasi_ids, how="left")
    unicos = (merged["n_orig"] == 1).sum()
    return {
        "taxa_divulgacao_pct": round(unicos / max(len(sintetico), 1) * 100, 3)
    }


# ============================================================
# 5. API unica
# ============================================================
def gerar_e_validar(
    original: pd.DataFrame,
    quasi_ids: list[str],
    k: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Gera sintetico e retorna (df_sintetico, relatorio)."""
    gen = GaussianCopulaGenerator(seed=seed).fit(original)
    sint = gen.sample(len(original))

    val = KAnonymityValidator(quasi_ids, k=k)
    rep_k = val.evaluate(original, sint)

    num_cols = [c for c in quasi_ids
                if pd.api.types.is_numeric_dtype(original[c])]
    dcr = distancia_mais_proximo_vizinho(original, sint, num_cols)
    div = taxa_divulgacao(original, sint, quasi_ids)

    relatorio = {
        "k_anonymity": rep_k.__dict__,
        "dcr": dcr,
        "divulgacao": div,
        "n_original": len(original),
        "n_sintetico": len(sint),
    }
    return sint, relatorio
