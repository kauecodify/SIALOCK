"""
Pipeline com Enriquecimento Externo + Treinamento Supervisionado
=================================================================
Fluxo:
  1. Lê XLSX e Parquet locais (dados reais do cliente)
  2. Enriquece CNPJs via APIs externas:
       • BrasilAPI        → dados cadastrais da Receita Federal
       • Minha Receita    → CNAE, sócios, situação cadastral
       • Comex Stat       → histórico de importação/exportação
       • IBGE             → município, UF, região
  3. Cacheia localmente (Parquet) para evitar re-consultas
  4. Calcula KPIs de qualidade + KPIs de negócio
  5. Score de risco colorido
  6. Treina ML (label = CNPJ) com features locais + externas
  7. Exporta XLSX colorido + Parquet + JSON
  8. Interface PyAutoGUI para consulta interativa
"""

import os
import json
import time
import hashlib
import warnings
import numpy as np
import pandas as pd
import joblib
import requests
import pyautogui

from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    top_k_accuracy_score, log_loss,
)
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DADOS      = "./dados"
PASTA_CACHE      = "./cache_externo"
PASTA_SAIDA      = "./saida"
MODELO_PATH      = "./modelo_importacao_externo.joblib"

CAMPO_CNPJ       = "CNPJ"

# Colunas locais
COLUNAS_TEXTO    = ["RazaoSocial", "Produto", "DescricaoNCM", "Fornecedor"]
COLUNAS_CAT      = ["CNAE", "UF", "PaisOrigem", "SituacaoCadastral", "HS_Code", "Importador"]
COLUNAS_NUM      = ["CapitalSocial", "ValorImportado", "PesoKg", "ValorFrete"]

# Colunas EXTERNAS (enriquecimento)
COLUNAS_EXT_TEXTO = ["EXT_RazaoSocial", "EXT_NomeFantasia", "EXT_CNAE_Descricao"]
COLUNAS_EXT_CAT   = ["EXT_UF", "EXT_Municipio", "EXT_Situacao", "EXT_Porte",
                     "EXT_NaturezaJuridica", "EXT_Regiao"]
COLUNAS_EXT_NUM   = ["EXT_CapitalSocial", "EXT_IdadeEmpresa",
                     "EXT_QtdSocios", "EXT_QtdCNAESecundarios",
                     "EXT_TotalImportacoes", "EXT_ValorTotalImportado"]

# KPIs (features)
COLUNAS_KPI = [
    "KPI_Subvalorizacao", "KPI_DisparidadeHS", "KPI_SurgeVolume",
    "KPI_DesvioETA", "KPI_Completude", "KPI_Outlier",
    "KPI_Ext_ConsistenciaCadastral", "KPI_Ext_IdadeEmpresa",
    "RiskScore",
]

COLUNAS_ALVO = (
    COLUNAS_TEXTO + COLUNAS_CAT + COLUNAS_NUM +
    COLUNAS_EXT_TEXTO + COLUNAS_EXT_CAT + COLUNAS_EXT_NUM +
    COLUNAS_KPI
)

# Auxiliares
COL_HS_CODE      = "HS_Code"
COL_ETA          = "ETA"
COL_ETA_REAL     = "ETAReal"
COL_IMPORTADOR   = "Importador"

# Limiares de risco
LIMIAR_SUBVALORIZACAO = 0.60
LIMIAR_SURGE          = 3.0
LIMIAR_ETA_DIAS       = 7

PESO_KPIS = {
    "subvalorizacao":    0.25,
    "disparidade_hs":    0.15,
    "surge_volume":      0.15,
    "desvio_eta":        0.10,
    "completude":        0.05,
    "outlier":           0.05,
    "consist_cadastral": 0.15,
    "idade_empresa":     0.10,
}

# APIs externas
API_BRASILAPI     = "https://brasilapi.com.br/api/cnpj/v1/{cnpj}"
API_MINHA_RECEITA = "https://minhareceita.org/{cnpj}"
API_IBGE_MUN      = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios/{codigo}"
API_COMEX         = "https://api-comexstat.mdic.gov.br/general"  # requer POST

TIMEOUT_API      = 15
MAX_CNPJ_EXTERNO = 500      # limite para não estourar rate limit

BATCH_SIZE       = 64
TOP_K            = [1, 3, 5, 10]
RANDOM_STATE     = 42
N_TOP_RISCO_XLSX = 2000


# ============================================================
# 1. INTERFACE PYATOGUI
# ============================================================

def menu_inicial():
    """Menu principal com PyAutoGUI."""
    resp = pyautogui.confirm(
        text="Deseja enriquecer os dados locais com APIs externas?\n\n"
             "• SIM: consulta BrasilAPI / Minha Receita / IBGE\n"
             "• NÃO: treina apenas com dados locais",
        title="🌐 Enriquecimento Externo",
        buttons=["Sim", "Não", "Cancelar"]
    )
    return resp


def confirmar(msg):
    return pyautogui.confirm(msg, "Confirmação", ["Sim", "Não"]) == "Sim"


def alertar(msg, titulo="Resultado"):
    pyautogui.alert(msg, titulo, "OK")


def coletar_valores_alvo(colunas):
    valores = {}
    for col in colunas:
        v = pyautogui.prompt(f"Valor para '{col}':", f"🔎 {col}", "")
        if v is None:
            return None
        valores[col] = v.strip()
    return valores


# ============================================================
# 2. CARREGAMENTO LOCAL (XLSX + PARQUET)
# ============================================================

def hash_cabecalho(df):
    return hashlib.md5(",".join(map(str, df.columns)).encode()).hexdigest()[:10]


def carregar_arquivos_reais(pasta):
    dfs, relatorios = [], []

    for arq in sorted(Path(pasta).glob("*")):
        try:
            if arq.suffix.lower() == ".xlsx":
                df = pd.read_excel(arq, dtype=str)
                tipo = "xlsx"
            elif arq.suffix.lower() == ".parquet":
                df = pd.read_parquet(arq, engine="pyarrow")
                tipo = "parquet"
            else:
                continue

            df.columns = [str(c).strip() for c in df.columns]
            if CAMPO_CNPJ not in df.columns:
                print(f"⚠️ Ignorado (sem '{CAMPO_CNPJ}'): {arq.name}")
                continue

            df[CAMPO_CNPJ] = (
                df[CAMPO_CNPJ].astype(str)
                .str.replace(r"\D", "", regex=True)
                .str.zfill(14)
            )

            relatorios.append({
                "arquivo":     arq.name,
                "tipo":        tipo,
                "linhas":      len(df),
                "colunas":     len(df.columns),
                "hash_header": hash_cabecalho(df),
                "completude":  round(float(df.notna().mean().mean()), 4),
            })
            dfs.append(df)
            print(f"✅ [{tipo:>7}] {arq.name} — {len(df)} linhas")
        except Exception as e:
            print(f"❌ {arq.name}: {e}")

    if not dfs:
        alertar(f"Nenhum XLSX/Parquet com '{CAMPO_CNPJ}' em {pasta}", "⚠️")
        return None, None

    # Merge progressivo
    df_final = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        comuns = [c for c in df.columns if c in df_final.columns and c != CAMPO_CNPJ]
        if comuns:
            df = df.drop(columns=comuns)
        df_final = df_final.merge(df, on=CAMPO_CNPJ, how="outer", suffixes=("", f"_{i}"))

    print(f"\n📊 Base unificada: {len(df_final)} linhas | {len(df_final.columns)} colunas")
    return df_final, pd.DataFrame(relatorios)


# ============================================================
# 3. ENRIQUECIMENTO EXTERNO
# ============================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _get_json(url, timeout=TIMEOUT_API):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "ML-Pipeline/1.0"})
    if r.status_code == 200:
        return r.json()
    if r.status_code in (404, 400):
        return None
    r.raise_for_status()


def consultar_brasilapi(cnpj):
    """Consulta dados cadastrais via BrasilAPI."""
    try:
        data = _get_json(API_BRASILAPI.format(cnpj=cnpj))
        if not data:
            return None
        return {
            "EXT_RazaoSocial":       data.get("razao_social"),
            "EXT_NomeFantasia":      data.get("nome_fantasia"),
            "EXT_CNAE_Codigo":       str(data.get("cnae_fiscal") or ""),
            "EXT_CNAE_Descricao":    data.get("cnae_fiscal_descricao"),
            "EXT_NaturezaJuridica":  data.get("natureza_juridica"),
            "EXT_UF":                data.get("uf"),
            "EXT_Municipio":         data.get("municipio"),
            "EXT_Situacao":          data.get("descricao_situacao_cadastral"),
            "EXT_Porte":             data.get("porte"),
            "EXT_CapitalSocial":     data.get("capital_social"),
            "EXT_DataAbertura":      data.get("data_inicio_atividade"),
            "EXT_QtdSocios":         len(data.get("qsa") or []),
            "EXT_QtdCNAESecundarios": len(data.get("cnaes_secundarios") or []),
        }
    except Exception:
        return None


def consultar_minha_receita(cnpj):
    """Consulta complementar via Minha Receita."""
    try:
        data = _get_json(API_MINHA_RECEITA.format(cnpj=cnpj))
        if not data:
            return None
        return {
            "EXT_MR_Simples":         data.get("opcao_pelo_simples"),
            "EXT_MR_MEI":             data.get("opcao_pelo_mei"),
            "EXT_MR_Email":           data.get("email"),
            "EXT_MR_Telefone":        data.get("ddd_telefone_1"),
        }
    except Exception:
        return None


def consultar_ibge_regiao(municipio):
    """Mapeia município → região (via IBGE). Placeholder para cache local."""
    if not municipio:
        return None
    regioes = {
        "norte":        ["AM", "RR", "AP", "PA", "TO", "RO", "AC"],
        "nordeste":     ["MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"],
        "centro-oeste": ["MT", "MS", "GO", "DF"],
        "sudeste":      ["SP", "RJ", "MG", "ES"],
        "sul":          ["PR", "SC", "RS"],
    }
    return None  # resolvido depois via UF


def uf_para_regiao(uf):
    if not uf:
        return "Desconhecida"
    uf = str(uf).upper().strip()
    mapa = {
        "AM": "Norte", "RR": "Norte", "AP": "Norte", "PA": "Norte",
        "TO": "Norte", "RO": "Norte", "AC": "Norte",
        "MA": "Nordeste", "PI": "Nordeste", "CE": "Nordeste", "RN": "Nordeste",
        "PB": "Nordeste", "PE": "Nordeste", "AL": "Nordeste", "SE": "Nordeste",
        "BA": "Nordeste",
        "MT": "Centro-Oeste", "MS": "Centro-Oeste", "GO": "Centro-Oeste", "DF": "Centro-Oeste",
        "SP": "Sudeste", "RJ": "Sudeste", "MG": "Sudeste", "ES": "Sudeste",
        "PR": "Sul", "SC": "Sul", "RS": "Sul",
    }
    return mapa.get(uf, "Desconhecida")


def enriquecer_cnpjs(df, cache_dir=PASTA_CACHE, max_cnpjs=MAX_CNPJ_EXTERNO):
    """
    Enriquece cada CNPJ único com dados externos.
    Cache local em Parquet para não repetir consultas.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "cnpj_cache.parquet")

    # Cache existente
    cache_df = pd.DataFrame()
    if os.path.exists(cache_file):
        try:
            cache_df = pd.read_parquet(cache_file, engine="pyarrow")
            print(f"💾 Cache existente: {len(cache_df)} CNPJs")
        except Exception:
            cache_df = pd.DataFrame()

    cnpjs_unicos = df[CAMPO_CNPJ].dropna().unique().tolist()
    cnpjs_cache  = set(cache_df[CAMPO_CNPJ]) if not cache_df.empty else set()
    cnpjs_faltantes = [c for c in cnpjs_unicos if c not in cnpjs_cache]

    # Limita quantidade
    if len(cnpjs_faltantes) > max_cnpjs:
        print(f"⚠️ Limitando consulta externa a {max_cnpjs} CNPJs "
              f"(de {len(cnpjs_faltantes)} faltantes)")
        cnpjs_faltantes = cnpjs_faltantes[:max_cnpjs]

    print(f"\n🌐 Enriquecendo {len(cnpjs_faltantes)} CNPJs via APIs externas...")
    novos = []
    for i, cnpj in enumerate(cnpjs_faltantes, 1):
        registro = {CAMPO_CNPJ: cnpj}

        br = consultar_brasilapi(cnpj)
        if br:
            registro.update(br)

        mr = consultar_minha_receita(cnpj)
        if mr:
            registro.update(mr)

        # Região via UF
        uf = registro.get("EXT_UF")
        registro["EXT_Regiao"] = uf_para_regiao(uf)

        # Idade da empresa (anos)
        data_abertura = registro.get("EXT_DataAbertura")
        if data_abertura:
            try:
                dt = pd.to_datetime(data_abertura, errors="coerce")
                if pd.notna(dt):
                    registro["EXT_IdadeEmpresa"] = round(
                        (pd.Timestamp.now() - dt).days / 365.25, 2
                    )
            except Exception:
                pass

        # Placeholders de comércio exterior (seriam preenchidos via Comex Stat)
        registro.setdefault("EXT_TotalImportacoes", np.nan)
        registro.setdefault("EXT_ValorTotalImportado", np.nan)

        novos.append(registro)

        if i % 25 == 0:
            print(f"   ... {i}/{len(cnpjs_faltantes)}")

        time.sleep(0.15)  # respeita rate limit

    # Atualiza cache
    if novos:
        cache_novo = pd.DataFrame(novos)
        cache_df = pd.concat([cache_df, cache_novo], ignore_index=True) \
            if not cache_df.empty else cache_novo
        cache_df = cache_df.drop_duplicates(subset=[CAMPO_CNPJ], keep="last")
        cache_df.to_parquet(cache_file, engine="pyarrow", index=False)
        print(f"💾 Cache atualizado: {len(cache_df)} CNPJs")

    # Merge com base principal
    if not cache_df.empty:
        df = df.merge(cache_df, on=CAMPO_CNPJ, how="left")
        print(f"✅ Base enriquecida: {df.shape[1]} colunas "
              f"(+{df.shape[1] - len(df.columns) + len(cache_df.columns)} externas)")
    return df


# ============================================================
# 4. KPIs DE QUALIDADE
# ============================================================

def calcular_kpis_qualidade(df, relatorios):
    kpis = {
        "total_linhas":       len(df),
        "total_colunas":      df.shape[1],
        "completude_media":   round(float(df.notna().mean().mean()), 4),
        "linhas_duplicadas":  int(df.duplicated().sum()),
        "cnpj_unicos":        int(df[CAMPO_CNPJ].nunique()),
        "taxa_duplicacao":    round(float(df[CAMPO_CNPJ].duplicated().mean()), 4),
    }
    if relatorios is not None and not relatorios.empty:
        kpis["arquivos_xlsx"]       = int((relatorios["tipo"] == "xlsx").sum())
        kpis["arquivos_parquet"]    = int((relatorios["tipo"] == "parquet").sum())
        kpis["hash_headers_unicos"] = int(relatorios["hash_header"].nunique())
    return kpis


# ============================================================
# 5. KPIs DE NEGÓCIO (incluindo externos)
# ============================================================

def calcular_kpis_negocio(df):
    df = df.copy()

    # Subvalorização
    if "ValorImportado" in df.columns and "PesoKg" in df.columns:
        val  = pd.to_numeric(df["ValorImportado"], errors="coerce")
        peso = pd.to_numeric(df["PesoKg"], errors="coerce").replace(0, np.nan)
        vk = val / peso
        mediana = vk.median()
        df["KPI_Subvalorizacao"] = np.clip(
            (mediana - vk) / (mediana if mediana else 1), 0, 1
        ).fillna(0).values
        df["ValorKg_Calculado"] = vk.values
    else:
        df["KPI_Subvalorizacao"] = 0.0

    # Disparidade HS
    if COL_HS_CODE in df.columns and "DescricaoNCM" in df.columns:
        hs = df[COL_HS_CODE].astype(str).str.replace(r"\D", "", regex=True)
        zeros_finais = hs.str.count(r"0+$") / hs.str.len().replace(0, 1)
        desc_len = df["DescricaoNCM"].astype(str).str.len()
        df["KPI_DisparidadeHS"] = np.clip(
            zeros_finais * (desc_len / (desc_len.max() or 1)), 0, 1
        ).fillna(0).values
    else:
        df["KPI_DisparidadeHS"] = 0.0

    # Surge
    if COL_IMPORTADOR in df.columns and "ValorImportado" in df.columns:
        val = pd.to_numeric(df["ValorImportado"], errors="coerce")
        df["_val_tmp"] = val
        stats = df.groupby(COL_IMPORTADOR)["_val_tmp"].agg(["mean", "std"]).reset_index()
        stats.columns = [COL_IMPORTADOR, "media", "std"]
        df = df.merge(stats, on=COL_IMPORTADOR, how="left")
        df["KPI_SurgeVolume"] = np.clip(
            (val - df["media"]) / (df["std"].replace(0, np.nan) * LIMIAR_SURGE), 0, 1
        ).fillna(0).values
        df = df.drop(columns=["_val_tmp", "media", "std"])
    else:
        df["KPI_SurgeVolume"] = 0.0

    # Desvio ETA
    if COL_ETA in df.columns and COL_ETA_REAL in df.columns:
        eta  = pd.to_datetime(df[COL_ETA], errors="coerce")
        real = pd.to_datetime(df[COL_ETA_REAL], errors="coerce")
        desvio = (real - eta).dt.days.abs()
        df["KPI_DesvioETA"] = np.clip(desvio / LIMIAR_ETA_DIAS, 0, 1).fillna(0).values
    else:
        df["KPI_DesvioETA"] = 0.0

    # Completude
    df["KPI_Completude"] = 1 - df.notna().mean(axis=1)

    # ---- KPIs EXTERNOS ----

    # Consistência cadastral (situação ativa + CNAE compatível)
    if "EXT_Situacao" in df.columns:
        ativa = df["EXT_Situacao"].astype(str).str.lower().str.contains("ativa")
        df["KPI_Ext_ConsistenciaCadastral"] = (~ativa).astype(float)
    else:
        df["KPI_Ext_ConsistenciaCadastral"] = 0.0

    # Idade da empresa (novas são mais arriscadas)
    if "EXT_IdadeEmpresa" in df.columns:
        idade = pd.to_numeric(df["EXT_IdadeEmpresa"], errors="coerce")
        df["KPI_Ext_IdadeEmpresa"] = np.clip(1 - idade / 10, 0, 1).fillna(0.5).values
    else:
        df["KPI_Ext_IdadeEmpresa"] = 0.0

    return df


def calcular_outliers(df):
    num_cols = [c for c in COLUNAS_NUM if c in df.columns]
    if not num_cols or len(df) < 10:
        return np.zeros(len(df))
    X_num = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, len(df) - 1), contamination=0.1)
        pred = lof.fit_predict(X_num)
        scores = -lof.negative_outlier_factor_
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return np.where(pred == -1, scores, scores * 0.3)
    except Exception:
        return np.zeros(len(df))


# ============================================================
# 6. SCORE DE RISCO
# ============================================================

def calcular_risco(df):
    df = df.copy()
    df["KPI_Outlier"] = calcular_outliers(df)

    mapping = {
        "subvalorizacao":    "KPI_Subvalorizacao",
        "disparidade_hs":    "KPI_DisparidadeHS",
        "surge_volume":      "KPI_SurgeVolume",
        "desvio_eta":        "KPI_DesvioETA",
        "completude":        "KPI_Completude",
        "outlier":           "KPI_Outlier",
        "consist_cadastral": "KPI_Ext_ConsistenciaCadastral",
        "idade_empresa":     "KPI_Ext_IdadeEmpresa",
    }
    score = np.zeros(len(df))
    for kpi, peso in PESO_KPIS.items():
        col = mapping[kpi]
        if col in df.columns:
            score += peso * df[col].fillna(0).values

    df["RiskScore"] = np.clip(score * 100, 0, 100).round(2)

    bins   = [0, 25, 50, 75, 100.01]
    labels = ["BAIXO", "MEDIO", "ALTO", "CRITICO"]
    cores  = {"BAIXO": "#C6EFCE", "MEDIO": "#FFEB9C",
              "ALTO": "#FFC7CE", "CRITICO": "#C00000"}

    df["RiskLevel"] = pd.cut(df["RiskScore"], bins=bins, labels=labels, right=False)
    df["RiskColor"] = df["RiskLevel"].map(cores).fillna("#FFFFFF")
    return df


# ============================================================
# 7. PREPROCESSADOR + ENSEMBLE SGD
# ============================================================

class TextConcat(BaseEstimator):
    def __init__(self, colunas):
        self.colunas = colunas
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.colunas].fillna("").astype(str).agg(" ".join, axis=1).values
        return np.array([" ".join(map(str, r)) for r in X])


def construir_preprocessador(df):
    txt = [c for c in (COLUNAS_TEXTO + COLUNAS_EXT_TEXTO) if c in df.columns]
    cat = [c for c in (COLUNAS_CAT + COLUNAS_EXT_CAT)     if c in df.columns]
    num = [c for c in (COLUNAS_NUM + COLUNAS_EXT_NUM + COLUNAS_KPI) if c in df.columns]

    transformers = []
    if txt:
        transformers.append((
            "tfidf",
            Pipeline([
                ("concat", TextConcat(txt)),
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2), min_df=1, max_features=5000,
                    sublinear_tf=True, strip_accents="unicode",
                )),
            ]),
            txt,
        ))
    if cat:
        transformers.append((
            "onehot",
            OneHotEncoder(handle_unknown="ignore", min_frequency=1),
            cat,
        ))
    if num:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]),
            num,
        ))

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


class SGDEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1e-4, random_state=42):
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        base = dict(
            alpha=self.alpha, max_iter=2000, tol=1e-4,
            learning_rate="adaptive", eta0=0.01,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, class_weight="balanced",
            average=True, random_state=self.random_state,
        )
        self.models_ = {
            "log":   SGDClassifier(loss="log_loss",       **base),
            "huber": SGDClassifier(loss="modified_huber", **base),
        }
        self._trained = {}
        for nome, m in self.models_.items():
            try:
                m.fit(X, y)
                self._trained[nome] = m
                print(f"   ✓ SGD[{nome}] treinado")
            except Exception as e:
                print(f"   ✗ SGD[{nome}]: {e}")
        if not self._trained:
            raise RuntimeError("Nenhum SGD conseguiu treinar.")
        return self

    def predict_proba(self, X):
        pesos = {"log": 1.0, "huber": 0.8}
        probs = []
        for nome, m in self._trained.items():
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X)
                if not np.array_equal(m.classes_, self.classes_):
                    p_full = np.zeros((X.shape[0], len(self.classes_)))
                    for i, c in enumerate(m.classes_):
                        idx = np.where(self.classes_ == c)[0]
                        if len(idx):
                            p_full[:, idx[0]] = p[:, i]
                    p = p_full
                probs.append(p * pesos[nome])
        probs = np.sum(probs, axis=0)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-12
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
        if not hasattr(self, "_trained") or not self._trained:
            base = dict(average=True, class_weight="balanced",
                        random_state=self.random_state)
            self.models_ = {
                "log":   SGDClassifier(loss="log_loss",       **base),
                "huber": SGDClassifier(loss="modified_huber", **base),
            }
            self._trained = dict(self.models_)
        for nome, m in self._trained.items():
            try:
                if not hasattr(m, "classes_"):
                    m.partial_fit(X, y, classes=classes)
                else:
                    m.partial_fit(X, y)
            except Exception:
                pass
        return self


# ============================================================
# 8. TREINAMENTO INCREMENTAL
# ============================================================

def treinar_incremental(modelo, X, y, batch_size=BATCH_SIZE):
    classes = np.unique(y)
    n = X.shape[0]
    historico = []
    for i in range(0, n, batch_size):
        Xb, yb = X[i:i+batch_size], y[i:i+batch_size]
        if len(Xb) == 0:
            continue
        try:
            modelo.partial_fit(Xb, yb, classes=classes)
            y_pred = modelo.predict(Xb)
            acc = accuracy_score(yb, y_pred)
            f1  = f1_score(yb, y_pred, average="macro", zero_division=0)
            historico.append({"batch": i // batch_size + 1, "acc": acc, "f1": f1})
            print(f"   Batch {historico[-1]['batch']:>3} | acc={acc:.4f} | f1={f1:.4f}")
        except Exception:
            continue
    return historico


# ============================================================
# 9. AVALIAÇÃO
# ============================================================

def avaliar_modelo(modelo, X, y):
    y_pred = modelo.predict(X)
    metricas = {
        "accuracy":  round(float(accuracy_score(y, y_pred)), 4),
        "f1_macro":  round(float(f1_score(y, y_pred, average="macro", zero_division=0)), 4),
        "precision": round(float(precision_score(y, y_pred, average="macro", zero_division=0)), 4),
        "recall":    round(float(recall_score(y, y_pred, average="macro", zero_division=0)), 4),
    }
    try:
        y_proba = modelo.predict_proba(X)
        n_classes = len(np.unique(y))
        for k in TOP_K:
            if k >= n_classes:
                continue
            try:
                metricas[f"top_{k}"] = round(float(
                    top_k_accuracy_score(y, y_proba, k=k, labels=modelo.classes_)
                ), 4)
            except Exception:
                pass
        try:
            metricas["log_loss"] = round(float(
                log_loss(y, y_proba, labels=modelo.classes_)
            ), 4)
        except Exception:
            pass
    except Exception:
        pass
    return metricas


# ============================================================
# 10. EXPORTAÇÃO XLSX COLORIDO
# ============================================================

def exportar_xlsx_colorido(df, caminho):
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows

    wb = Workbook()
    ws = wb.active
    ws.title = "Auditoria Importacoes"

    cores = {"BAIXO": "C6EFCE", "MEDIO": "FFEB9C",
             "ALTO": "FFC7CE", "CRITICO": "C00000"}

    cols = list(df.columns)
    ws.append(cols)
    for c in range(1, len(cols) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center")

    col_risk  = cols.index("RiskLevel") + 1 if "RiskLevel" in cols else None
    col_score = cols.index("RiskScore") + 1 if "RiskScore" in cols else None

    for r in dataframe_to_rows(df, index=False, header=False):
        ws.append(r)
        row_idx = ws.max_row
        if col_risk:
            val = str(ws.cell(row=row_idx, column=col_risk).value or "").upper()
            hex_cor = cores.get(val, "FFFFFF")
            fill = PatternFill("solid", fgColor=hex_cor)
            font_color = "FFFFFF" if val == "CRITICO" else "000000"
            for c in range(1, len(cols) + 1):
                ws.cell(row=row_idx, column=c).fill = fill
            ws.cell(row=row_idx, column=col_risk).font = Font(bold=True, color=font_color)
        if col_score:
            ws.cell(row=row_idx, column=col_score).number_format = "0.00"

    for i, col in enumerate(cols, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
            max(12, min(30, len(col) + 4))

    wb.save(caminho)
    print(f"💾 XLSX colorido: {caminho}")


# ============================================================
# 11. PIPELINE PRINCIPAL
# ============================================================

def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    os.makedirs(PASTA_CACHE, exist_ok=True)

    print("=" * 74)
    print("  PIPELINE EXTERNO — APIs + XLSX + PARQUET | KPIs | RISCO | CNPJ")
    print("=" * 74)

    t0 = time.time()

    # ---------- 0. Menu ----------
    resp = menu_inicial()
    if resp == "Cancelar" or resp is None:
        print("Cancelado pelo usuário.")
        return
    usar_externo = (resp == "Sim")

    # ---------- 1. Carregar local ----------
    print("\n📂 Carregando XLSX e Parquet locais...")
    df, relatorios = carregar_arquivos_reais(PASTA_DADOS)
    if df is None or df.empty:
        return

    # ---------- 2. Enriquecimento externo ----------
    if usar_externo:
        df = enriquecer_cnpjs(df)
    else:
        print("\n⏭️ Enriquecimento externo pulado (modo local).")

    # ---------- 3. KPIs de qualidade ----------
    print("\n📋 KPIs de qualidade:")
    kpis_q = calcular_kpis_qualidade(df, relatorios)
    for k, v in kpis_q.items():
        print(f"   • {k}: {v}")

    # ---------- 4. KPIs de negócio ----------
    print("\n🎯 KPIs de negócio (locais + externos)...")
    df = calcular_kpis_negocio(df)

    # ---------- 5. Risco ----------
    print("\n🚨 Score de risco...")
    df = calcular_risco(df)
    print(df["RiskLevel"].value_counts().to_string())

    # ---------- 6. Features ----------
    colunas_disponiveis = [c for c in COLUNAS_ALVO if c in df.columns]
    print(f"\n🎯 Features ({len(colunas_disponiveis)}): {colunas_disponiveis[:10]}...")

    preproc = construir_preprocessador(df)
    X = preproc.fit_transform(df)
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    print(f"   Dimensão: {X_dense.shape}")

    # ---------- 7. Label = CNPJ ----------
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df[CAMPO_CNPJ].astype(str).values)
    n_classes = len(label_enc.classes_)
    print(f"   Classes (CNPJs): {n_classes}")

    if n_classes < 2:
        alertar("É necessário pelo menos 2 CNPJs distintos.", "⚠️")
        return

    # ---------- 8. Split ----------
    strat = y if np.min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_dense, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat,
    )
    print(f"\n   Treino: {len(X_tr)} | Teste: {len(X_te)}")

    # ---------- 9. Ensemble SGD ----------
    print("\n🧩 Treinando ensemble SGD...")
    ens = SGDEnsemble(alpha=1e-4).fit(X_tr, y_tr)

    # ---------- 10. Calibração ----------
    print("\n🎯 Calibrando probabilidades...")
    try:
        cal = CalibratedClassifierCV(
            estimator=SGDClassifier(
                loss="log_loss", alpha=1e-4, average=True,
                max_iter=2000, early_stopping=True, class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            method="sigmoid", cv=3,
        )
        cal.fit(X_tr, y_tr)
        usar_cal = True
        print("   ✓ Calibração OK")
    except Exception as e:
        print(f"   ⚠️ Calibração falhou: {e}")
        usar_cal = False

    # ---------- 11. Incremental ----------
    print("\n🧠 Treinamento incremental (partial_fit)...")
    inc = SGDEnsemble(alpha=1e-4, random_state=RANDOM_STATE)
    inc._trained = {}
    hist = treinar_incremental(inc, X_tr, y_tr)
    if hist:
        print(f"   ✅ {len(hist)} batches | acc final = {hist[-1]['acc']:.4f}")

    # ---------- 12. Avaliação ----------
    print("\n📊 Avaliação:")
    m_ens = avaliar_modelo(ens, X_te, y_te)
    m_cal = avaliar_modelo(cal, X_te, y_te) if usar_cal else {}
    m_inc = avaliar_modelo(inc, X_te, y_te) if hist else {}

    def _print(m, nome):
        if not m: return
        print(f"\n── {nome} ──")
        for k, v in m.items():
            print(f"   • {k:>10}: {v}")

    _print(m_ens, "Ensemble SGD")
    _print(m_cal, "SGD Calibrado")
    _print(m_inc, "SGD Incremental")

    candidatos = {"ensemble": (ens, m_ens)}
    if m_cal: candidatos["calibrado"]   = (cal, m_cal)
    if m_inc: candidatos["incremental"] = (inc, m_inc)

    melhor_nome = max(
        candidatos,
        key=lambda k: candidatos[k][1].get("top_5", candidatos[k][1].get("accuracy", 0))
    )
    melhor_modelo, melhor_metricas = candidatos[melhor_nome]
    print(f"\n🏆 Melhor modelo: {melhor_nome}")

    # ---------- 13. Persistência ----------
    joblib.dump({
        "modelo":         melhor_modelo,
        "preprocessador": preproc,
        "label_encoder":  label_enc,
        "metricas":       melhor_metricas,
        "colunas_alvo":   colunas_disponiveis,
        "tipo":           melhor_nome,
        "usa_externo":    usar_externo,
    }, MODELO_PATH, compress=3)
    print(f"\n💾 Modelo: {MODELO_PATH}")

    # ---------- 14. Anexa previsões ----------
    try:
        y_proba_full = melhor_modelo.predict_proba(X_dense)
        top_idx_full = np.argsort(y_proba_full, axis=1)[:, ::-1][:, :5]
        df["CNPJ_Previsto"]  = label_enc.inverse_transform(y_proba_full.argmax(axis=1))
        df["CNPJ_Confianca"] = np.round(y_proba_full.max(axis=1), 4)
        for r in range(5):
            df[f"CNPJ_Top{r+1}"]   = label_enc.inverse_transform(top_idx_full[:, r])
            df[f"CNPJ_Top{r+1}_p"] = np.round(
                np.take_along_axis(y_proba_full, top_idx_full[:, r:r+1], axis=1).ravel(), 4
            )
    except Exception as e:
        print(f"⚠️ Não anexou previsões: {e}")

    # ---------- 15. Exportação ----------
    ts = time.strftime("%Y%m%d_%H%M%S")

    parquet_path = os.path.join(PASTA_SAIDA, f"auditoria_externo_{ts}.parquet")
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    print(f"💾 Parquet: {parquet_path}")

    top_risco = df.sort_values("RiskScore", ascending=False).head(N_TOP_RISCO_XLSX)
    xlsx_path = os.path.join(PASTA_SAIDA, f"risco_colorido_externo_{ts}.xlsx")
    exportar_xlsx_colorido(top_risco, xlsx_path)

    kpi_path = os.path.join(PASTA_SAIDA, f"kpis_externo_{ts}.json")
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump({
            "qualidade":           kpis_q,
            "risco_distribuicao":  df["RiskLevel"].value_counts().to_dict(),
            "metricas_ml":         melhor_metricas,
            "tipo_melhor_modelo":  melhor_nome,
            "classes_cnpj":        int(n_classes),
            "usa_externo":         bool(usar_externo),
            "colunas_externas":    int(sum(1 for c in df.columns if c.startswith("EXT_"))),
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"💾 KPIs JSON: {kpi_path}")

    # ---------- 16. Resumo ----------
    dist = df["RiskLevel"].value_counts().to_dict()
    alertar(
        f"✅ Pipeline externo concluído!\n\n"
        f"Linhas:  {len(df)}\n"
        f"CNPJs (classes): {n_classes}\n"
        f"Enriquecimento externo: {'SIM' if usar_externo else 'NÃO'}\n"
        f"Colunas EXT_: {sum(1 for c in df.columns if c.startswith('EXT_'))}\n\n"
        f"Risco:\n" + "\n".join(f"  • {k}: {v}" for k, v in dist.items()) +
        f"\n\nMelhor modelo: {melhor_nome}\n" +
        "\n".join(f"  • {k}: {v}" for k, v in melhor_metricas.items()),
        "🎯 Finalizado"
    )

    # ---------- 17. Consulta interativa ----------
    while True:
        if not confirmar("Consultar CNPJ de uma operação?"):
            break
        valores = coletar_valores_alvo(COLUNAS_ALVO)
        if valores is None:
            break
        try:
            Xq = preproc.transform(pd.DataFrame([valores]))
            Xq = Xq.toarray() if hasattr(Xq, "toarray") else Xq
            probs = melhor_modelo.predict_proba(Xq)[0]
            top = np.argsort(probs)[::-1][:5]
            linhas = "\n".join(
                f"{r+1}º  CNPJ {label_enc.inverse_transform([i])[0]}  (p={probs[i]:.4f})"
                for r, i in enumerate(top)
            )
            alertar(f"🎯 Top-5 CNPJs previstos:\n\n{linhas}", "Captura de CNPJ")
        except Exception as e:
            alertar(f"Erro na consulta: {e}", "⚠️")

    print(f"\n⏱️ Tempo total: {time.time() - t0:.2f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
