"""
EXTRACT V.2
=============================================
Roadmap de treinamento supervisionado com enriquecimento externo,
extracao via PDF/clipboard (PyAutoGUI), historico de sessoes e
deteccao de padroes para "palpites" (CNPJ alvo).

-----------------------------------------------------------------
INSTALACAO / AMBIENTE
-----------------------------------------------------------------
Opcao 1 - Conda (recomendado):

    conda create -n nt1 python=3.11 -y
    conda activate nt1
    conda install -c conda-forge pandas numpy scikit-learn joblib \
        pyarrow openpyxl requests tenacity pyautogui pyperclip \
        pdfplumber pypdf pillow pytesseract -y
    pip install python-Levenshtein

Opcao 2 - Pip puro:

    python -m venv .venv
    source .venv/bin/activate        # Linux/Mac
    .venv\\Scripts\\activate           # Windows
    pip install --upgrade pip
    pip install pandas numpy scikit-learn joblib pyarrow openpyxl \
        requests tenacity pyautogui pyperclip pdfplumber pypdf \
        pillow pytesseract

Opcao 3 - Requirements (se preferir arquivo):

    pip install -r requirements.txt

-----------------------------------------------------------------
EXECUCAO
-----------------------------------------------------------------

    python nt1_extractor.py

Antes de rodar, crie a pasta ./dados com seus XLSX / Parquet / PDF / TXT.

-----------------------------------------------------------------
DEPENDENCIAS EXTERNAS OPCIONAIS
-----------------------------------------------------------------
- Tesseract OCR (para PDFs digitalizados):
    Linux:   sudo apt-get install tesseract-ocr tesseract-ocr-por
    Mac:     brew install tesseract tesseract-lang
    Windows: https://github.com/UB-Mannheim/tesseract/wiki

- pyautogui em Linux headless:
    sudo apt-get install python3-tk python3-dev scrot

-----------------------------------------------------------------
ROADMAP DE TREINAMENTO
-----------------------------------------------------------------
  F0  Bootstrap      - carga de XLSX/Parquet/PDF/TXT/CSV
  F1  Enriquecimento - BrasilAPI, Minha Receita, ReceitaWS, CNPJ.ws, IBGE, Comex
  F2  KPI-All        - KPI por coluna (nao so nas pre-definidas)
  F3  Char-Ngram     - mineracao de caracteres para series de "palpites"
  F4  Warmup         - treino full batch (ensemble SGD+PA+NB+SVC)
  F5  Incremental    - partial_fit em batches
  F6  Refinamento    - calibracao + stacking + selecao por top-k
  F7  Historico      - acumula em nt1_training_history.json (>= 10 sessoes)
  F8  Padrao         - detecta tendencias entre sessoes (features que sobem)
  F9  Extracao GUI   - PyAutoGUI + clipboard + OCR opcional
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import warnings
import statistics
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import joblib
import requests
import pyautogui

from tenacity import retry, stop_after_attempt, wait_exponential

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    top_k_accuracy_score, log_loss, classification_report,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2

warnings.filterwarnings("ignore")


# ==================================================================
# 0. CONFIGURACOES GLOBAIS
# ==================================================================

PASTA_DADOS   = "./dados"
PASTA_CACHE   = "./cache_externo"
PASTA_SAIDA   = "./saida"
PASTA_HIST    = "./historico"
MODELO_PATH   = "./modelo_nt1_extractor.joblib"
HIST_PATH     = os.path.join(PASTA_HIST, "nt1_training_history.json")
PADRAO_PATH   = os.path.join(PASTA_HIST, "nt1_patterns.json")

CAMPO_CNPJ    = "CNPJ"

# Colunas locais
COLUNAS_TEXTO = ["RazaoSocial", "Produto", "DescricaoNCM", "Fornecedor", "Observacao"]
COLUNAS_CAT   = ["CNAE", "UF", "PaisOrigem", "SituacaoCadastral", "HS_Code", "Importador"]
COLUNAS_NUM   = ["CapitalSocial", "ValorImportado", "PesoKg", "ValorFrete", "Quantidade", "S_4"]

# Colunas externas (enriquecimento)
COLUNAS_EXT_TEXTO = ["EXT_RazaoSocial", "EXT_NomeFantasia", "EXT_CNAE_Descricao",
                     "EXT_Email", "EXT_Telefone"]
COLUNAS_EXT_CAT   = ["EXT_UF", "EXT_Municipio", "EXT_Situacao", "EXT_Porte",
                     "EXT_NaturezaJuridica", "EXT_Regiao", "EXT_Simples", "EXT_MEI"]
COLUNAS_EXT_NUM   = ["EXT_CapitalSocial", "EXT_IdadeEmpresa", "EXT_QtdSocios",
                     "EXT_QtdCNAESecundarios", "EXT_TotalImportacoes",
                     "EXT_ValorTotalImportado", "EXT_ScoreCredito", "S_4"]

# KPIs de risco
COLUNAS_KPI = [
    "KPI_Subvalorizacao", "KPI_DisparidadeHS", "KPI_SurgeVolume",
    "KPI_DesvioETA", "KPI_Completude", "KPI_Outlier",
    "KPI_Ext_ConsistenciaCadastral", "KPI_Ext_IdadeEmpresa",
    "KPI_Ext_ScoreCredito", "KPI_PadraoChar",
    "RiskScore",
]

COLUNAS_ALVO = (
    COLUNAS_TEXTO + COLUNAS_CAT + COLUNAS_NUM +
    COLUNAS_EXT_TEXTO + COLUNAS_EXT_CAT + COLUNAS_EXT_NUM +
    COLUNAS_KPI
)

# Auxiliares
COL_HS_CODE    = "HS_Code"
COL_ETA        = "ETA"
COL_ETA_REAL   = "ETAReal"
COL_IMPORTADOR = "Importador"

# Limiares
LIMIAR_SUBVALORIZACAO = 0.60
LIMIAR_SURGE          = 3.0
LIMIAR_ETA_DIAS       = 7
LIMIAR_PADRAO_CHAR    = 0.55

# Pesos do score de risco
PESO_KPIS = {
    "subvalorizacao":    0.20,
    "disparidade_hs":    0.12,
    "surge_volume":      0.12,
    "desvio_eta":        0.08,
    "completude":        0.04,
    "outlier":           0.05,
    "consist_cadastral": 0.13,
    "idade_empresa":     0.08,
    "score_credito":     0.10,
    "padrao_char":       0.08,
}

# APIs externas
API_BRASILAPI     = "https://brasilapi.com.br/api/cnpj/v1/{cnpj}"
API_MINHA_RECEITA = "https://minhareceita.org/{cnpj}"
API_RECEITAWS     = "https://receitaws.com.br/v1/cnpj/{cnpj}"
API_CNPJWS        = "https://publica.cnpj.ws/cnpj/{cnpj}"
API_IBGE_UF       = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}"

TIMEOUT_API      = 15
MAX_CNPJ_EXTERNO = 500
BATCH_SIZE       = 64
TOP_K            = [1, 3, 5, 10]
RANDOM_STATE     = 42
N_TOP_RISCO_XLSX = 2000
N_SESSOES_PADRAO = 10


# ==================================================================
# 1. UI - PYATOGUI
# ==================================================================

def menu_inicial() -> str:
    return pyautogui.confirm(
        text="NT-1 EXTRACTOR - escolha o modo de execucao:\n\n"
             "- Completo: enriquece APIs + treina roadmap + detecta padrao\n"
             "- Rapido:   treina apenas com dados locais\n"
             "- Cancelar: sair",
        title="NT-1 EXTRACTOR",
        buttons=["Completo", "Rapido", "Cancelar"]
    )


def confirmar(msg: str) -> bool:
    return pyautogui.confirm(msg, "Confirmacao", ["Sim", "Nao"]) == "Sim"


def alertar(msg: str, titulo: str = "Resultado"):
    pyautogui.alert(msg, titulo, "OK")


def coletar_valores_alvo(colunas: list[str]) -> dict | None:
    valores = {}
    for col in colunas:
        v = pyautogui.prompt(f"Valor para '{col}':", f"{col}", "")
        if v is None:
            return None
        valores[col] = v.strip()
    return valores


def captura_clipboard(titulo="Captura de PDF / Palpites") -> str | None:
    """
    Bloco PyAutoGUI para capturar texto do clipboard apos o usuario
    copiar (Ctrl+C) o conteudo de um PDF com caracteres/palpites.
    """
    resp = pyautogui.confirm(
        "Copie o conteudo do PDF (Ctrl+C) e clique em 'Ja copiei'.",
        titulo, ["Ja copiei", "Cancelar"]
    )
    if resp != "Ja copiei":
        return None
    try:
        import pyperclip
        return pyperclip.paste()
    except Exception:
        pyautogui.hotkey("ctrl", "v")
        return None


# ==================================================================
# 2. CARREGAMENTO LOCAL - XLSX / PARQUET / PDF / TXT
# ==================================================================

def hash_cabecalho(df: pd.DataFrame) -> str:
    return hashlib.md5(",".join(map(str, df.columns)).encode()).hexdigest()[:10]


def extrair_texto_pdf(caminho: str) -> str:
    """Extrai texto de PDF; tenta pdfplumber -> pypdf -> pytesseract."""
    try:
        import pdfplumber
        with pdfplumber.open(caminho) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(caminho).pages)
    except Exception:
        pass
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(caminho), lang="por")
    except Exception:
        return ""


def carregar_arquivos_reais(pasta: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    dfs, relatorios = [], []

    for arq in sorted(Path(pasta).glob("*")):
        try:
            suf = arq.suffix.lower()
            if suf == ".xlsx":
                df = pd.read_excel(arq, dtype=str)
                tipo = "xlsx"
            elif suf == ".parquet":
                df = pd.read_parquet(arq, engine="pyarrow")
                tipo = "parquet"
            elif suf == ".csv":
                df = pd.read_csv(arq, dtype=str, sep=None, engine="python")
                tipo = "csv"
            elif suf == ".pdf":
                txt = extrair_texto_pdf(str(arq))
                if not txt.strip():
                    continue
                cnpjs = re.findall(r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}", txt)
                if not cnpjs:
                    continue
                df = pd.DataFrame({CAMPO_CNPJ: cnpjs, "TextoPDF": [txt] * len(cnpjs)})
                tipo = "pdf"
            elif suf == ".txt":
                txt = arq.read_text(encoding="utf-8", errors="ignore")
                cnpjs = re.findall(r"\d{14}", txt)
                if not cnpjs:
                    continue
                df = pd.DataFrame({CAMPO_CNPJ: cnpjs, "TextoPDF": [txt] * len(cnpjs)})
                tipo = "txt"
            else:
                continue

            df.columns = [str(c).strip() for c in df.columns]
            if CAMPO_CNPJ not in df.columns:
                print(f"[AVISO] Ignorado (sem '{CAMPO_CNPJ}'): {arq.name}")
                continue

            df[CAMPO_CNPJ] = (df[CAMPO_CNPJ].astype(str)
                              .str.replace(r"\D", "", regex=True)
                              .str.zfill(14))

            relatorios.append({
                "arquivo":     arq.name,
                "tipo":        tipo,
                "linhas":      len(df),
                "colunas":     len(df.columns),
                "hash_header": hash_cabecalho(df),
                "completude":  round(float(df.notna().mean().mean()), 4),
            })
            dfs.append(df)
            print(f"[OK] [{tipo:>7}] {arq.name} - {len(df)} linhas")
        except Exception as e:
            print(f"[ERRO] {arq.name}: {e}")

    if not dfs:
        alertar(f"Nenhum arquivo utilizavel em {pasta}", "Aviso")
        return None, None

    df_final = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        comuns = [c for c in df.columns if c in df_final.columns and c != CAMPO_CNPJ]
        if comuns:
            df = df.drop(columns=comuns)
        df_final = df_final.merge(df, on=CAMPO_CNPJ, how="outer", suffixes=("", f"_{i}"))

    print(f"\n[INFO] Base unificada: {len(df_final)} linhas | {len(df_final.columns)} colunas")
    return df_final, pd.DataFrame(relatorios)


# ==================================================================
# 3. ENRIQUECIMENTO EXTERNO MULTI-API
# ==================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _get_json(url: str, timeout: int = TIMEOUT_API):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "NT1-Pipeline/2.0"})
    if r.status_code == 200:
        return r.json()
    if r.status_code in (404, 400, 429):
        return None
    r.raise_for_status()


def consultar_brasilapi(cnpj: str) -> dict | None:
    try:
        d = _get_json(API_BRASILAPI.format(cnpj=cnpj))
        if not d:
            return None
        return {
            "EXT_RazaoSocial":        d.get("razao_social"),
            "EXT_NomeFantasia":       d.get("nome_fantasia"),
            "EXT_CNAE_Codigo":        str(d.get("cnae_fiscal") or ""),
            "EXT_CNAE_Descricao":     d.get("cnae_fiscal_descricao"),
            "EXT_NaturezaJuridica":   d.get("natureza_juridica"),
            "EXT_UF":                 d.get("uf"),
            "EXT_Municipio":          d.get("municipio"),
            "EXT_Situacao":           d.get("descricao_situacao_cadastral"),
            "EXT_Porte":              d.get("porte"),
            "EXT_CapitalSocial":      d.get("capital_social"),
            "EXT_DataAbertura":       d.get("data_inicio_atividade"),
            "EXT_QtdSocios":          len(d.get("qsa") or []),
            "EXT_QtdCNAESecundarios": len(d.get("cnaes_secundarios") or []),
            "EXT_Email":              d.get("email"),
            "EXT_Telefone":           d.get("ddd_telefone_1"),
        }
    except Exception:
        return None


def consultar_minha_receita(cnpj: str) -> dict | None:
    try:
        d = _get_json(API_MINHA_RECEITA.format(cnpj=cnpj))
        if not d:
            return None
        return {
            "EXT_Simples":       "Sim" if d.get("opcao_pelo_simples") else "Nao",
            "EXT_MEI":           "Sim" if d.get("opcao_pelo_mei") else "Nao",
            "EXT_MR_Email":      d.get("email"),
            "EXT_MR_Telefone":   d.get("ddd_telefone_1"),
        }
    except Exception:
        return None


def consultar_receitaws(cnpj: str) -> dict | None:
    try:
        d = _get_json(API_RECEITAWS.format(cnpj=cnpj))
        if not d or d.get("status") == "ERROR":
            return None
        return {
            "EXT_RW_Situacao":   d.get("situacao"),
            "EXT_RW_Atividade":  d.get("atividade_principal", [{}])[0].get("text"),
        }
    except Exception:
        return None


def consultar_cnpjws(cnpj: str) -> dict | None:
    try:
        d = _get_json(API_CNPJWS.format(cnpj=cnpj))
        if not d or "estabelecimento" not in d:
            return None
        est = d["estabelecimento"]
        return {
            "EXT_WS_Bairro":  est.get("bairro"),
            "EXT_WS_CEP":     est.get("cep"),
            "EXT_WS_Cidade":  (est.get("cidade") or {}).get("nome"),
        }
    except Exception:
        return None


def uf_para_regiao(uf: str | None) -> str:
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


def enriquecer_cnpjs(df: pd.DataFrame,
                     cache_dir: str = PASTA_CACHE,
                     max_cnpjs: int = MAX_CNPJ_EXTERNO) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "cnpj_cache.parquet")

    cache_df = pd.DataFrame()
    if os.path.exists(cache_file):
        try:
            cache_df = pd.read_parquet(cache_file, engine="pyarrow")
            print(f"[CACHE] Existente: {len(cache_df)} CNPJs")
        except Exception:
            cache_df = pd.DataFrame()

    cnpjs_unicos = df[CAMPO_CNPJ].dropna().unique().tolist()
    cnpjs_cache = set(cache_df[CAMPO_CNPJ]) if not cache_df.empty else set()
    faltantes = [c for c in cnpjs_unicos if c not in cnpjs_cache]

    if len(faltantes) > max_cnpjs:
        print(f"[AVISO] Limitando a {max_cnpjs} CNPJs (de {len(faltantes)} faltantes)")
        faltantes = faltantes[:max_cnpjs]

    print(f"\n[INFO] Enriquecendo {len(faltantes)} CNPJs...")
    novos = []
    for i, cnpj in enumerate(faltantes, 1):
        reg = {CAMPO_CNPJ: cnpj}

        for fn in (consultar_brasilapi,
                   consultar_minha_receita,
                   consultar_receitaws,
                   consultar_cnpjws):
            r = fn(cnpj)
            if r:
                reg.update(r)

        reg["EXT_Regiao"] = uf_para_regiao(reg.get("EXT_UF"))

        dt_ab = reg.get("EXT_DataAbertura")
        if dt_ab:
            try:
                dt = pd.to_datetime(dt_ab, errors="coerce")
                if pd.notna(dt):
                    reg["EXT_IdadeEmpresa"] = round(
                        (pd.Timestamp.now() - dt).days / 365.25, 2)
            except Exception:
                pass

        reg["EXT_ScoreCredito"] = _score_credito_heuristica(reg)

        reg.setdefault("EXT_TotalImportacoes", np.nan)
        reg.setdefault("EXT_ValorTotalImportado", np.nan)

        novos.append(reg)

        if i % 25 == 0:
            print(f"   ... {i}/{len(faltantes)}")
        time.sleep(0.15)

    if novos:
        cache_novo = pd.DataFrame(novos)
        cache_df = pd.concat([cache_df, cache_novo], ignore_index=True) \
            if not cache_df.empty else cache_novo
        cache_df = cache_df.drop_duplicates(subset=[CAMPO_CNPJ], keep="last")
        cache_df.to_parquet(cache_file, engine="pyarrow", index=False)
        print(f"[CACHE] Atualizado: {len(cache_df)} CNPJs")

    if not cache_df.empty:
        antes = df.shape[1]
        df = df.merge(cache_df, on=CAMPO_CNPJ, how="left")
        print(f"[OK] Enriquecido: +{df.shape[1] - antes} colunas externas")
    return df


def _score_credito_heuristica(reg: dict) -> float:
    score = 0.5
    sit = str(reg.get("EXT_Situacao") or "").lower()
    if "ativa" in sit:
        score += 0.2
    elif "baixada" in sit or "suspensa" in sit:
        score -= 0.3

    idade = reg.get("EXT_IdadeEmpresa") or 0
    if idade >= 10:
        score += 0.15
    elif idade < 2:
        score -= 0.15

    cap = reg.get("EXT_CapitalSocial") or 0
    try:
        cap = float(cap)
        if cap > 1_000_000:
            score += 0.1
        elif cap < 10_000:
            score -= 0.05
    except Exception:
        pass

    if (reg.get("EXT_QtdSocios") or 0) > 2:
        score += 0.05

    return float(np.clip(score, 0, 1))


# ==================================================================
# 4. KPI-ALL - KPIs em TODAS as colunas (roadmap F2)
# ==================================================================

def calcular_kpis_qualidade(df: pd.DataFrame, relatorios: pd.DataFrame) -> dict:
    kpis = {
        "total_linhas":      len(df),
        "total_colunas":     df.shape[1],
        "completude_media":  round(float(df.notna().mean().mean()), 4),
        "linhas_duplicadas": int(df.duplicated().sum()),
        "cnpj_unicos":       int(df[CAMPO_CNPJ].nunique()),
        "taxa_duplicacao":   round(float(df[CAMPO_CNPJ].duplicated().mean()), 4),
    }
    if relatorios is not None and not relatorios.empty:
        kpis["arquivos_xlsx"]       = int((relatorios["tipo"] == "xlsx").sum())
        kpis["arquivos_parquet"]    = int((relatorios["tipo"] == "parquet").sum())
        kpis["arquivos_pdf_txt"]    = int(relatorios["tipo"].isin(["pdf", "txt"]).sum())
        kpis["hash_headers_unicos"] = int(relatorios["hash_header"].nunique())
    return kpis


def calcular_kpi_por_coluna(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        serie = df[col]
        rows.append({
            "coluna":        col,
            "dtype":         str(serie.dtype),
            "completude":    round(float(serie.notna().mean()), 4),
            "unicos":        int(serie.nunique(dropna=True)),
            "taxa_nulos":    round(float(serie.isna().mean()), 4),
            "cardinalidade": round(serie.nunique(dropna=True) / max(len(serie), 1), 4),
        })
    return pd.DataFrame(rows)


# ==================================================================
# 5. CHAR-NGRAM - mineracao de "palpites" (roadmap F3)
# ==================================================================

def extrair_padroes_char(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cols_txt = [c for c in (COLUNAS_TEXTO + COLUNAS_EXT_TEXTO + ["TextoPDF"])
                if c in df.columns]
    if not cols_txt:
        df["KPI_PadraoChar"] = 0.0
        return df, {}

    blob = df[cols_txt].fillna("").astype(str).agg(" ".join, axis=1)
    blob = blob.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

    vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                          min_df=2, max_features=300)
    try:
        M = vec.fit_transform(blob)
    except ValueError:
        df["KPI_PadraoChar"] = 0.0
        return df, {}

    vocab = vec.get_feature_names_out()
    freq = np.asarray(M.sum(axis=0)).ravel()
    top_idx = np.argsort(freq)[::-1][:50]
    top_padroes = {vocab[i]: int(freq[i]) for i in top_idx if freq[i] > 1}

    top_set = set(top_padroes.keys())

    def _score(s: str) -> float:
        hits = sum(1 for g in top_set if g in s)
        return round(min(hits / 10.0, 1.0), 4)

    df["KPI_PadraoChar"] = blob.apply(_score).values
    return df, top_padroes


# ==================================================================
# 6. KPIs DE NEGOCIO + RISCO
# ==================================================================

def calcular_kpis_negocio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ValorImportado" in df.columns and "PesoKg" in df.columns:
        val = pd.to_numeric(df["ValorImportado"], errors="coerce")
        peso = pd.to_numeric(df["PesoKg"], errors="coerce").replace(0, np.nan)
        vk = val / peso
        mediana = vk.median()
        df["KPI_Subvalorizacao"] = np.clip(
            (mediana - vk) / (mediana if mediana else 1), 0, 1
        ).fillna(0).values
        df["ValorKg_Calculado"] = vk.values
    else:
        df["KPI_Subvalorizacao"] = 0.0

    if COL_HS_CODE in df.columns and "DescricaoNCM" in df.columns:
        hs = df[COL_HS_CODE].astype(str).str.replace(r"\D", "", regex=True)
        zeros_finais = hs.str.count(r"0+$") / hs.str.len().replace(0, 1)
        desc_len = df["DescricaoNCM"].astype(str).str.len()
        df["KPI_DisparidadeHS"] = np.clip(
            zeros_finais * (desc_len / (desc_len.max() or 1)), 0, 1
        ).fillna(0).values
    else:
        df["KPI_DisparidadeHS"] = 0.0

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

    if COL_ETA in df.columns and COL_ETA_REAL in df.columns:
        eta = pd.to_datetime(df[COL_ETA], errors="coerce")
        real = pd.to_datetime(df[COL_ETA_REAL], errors="coerce")
        desvio = (real - eta).dt.days.abs()
        df["KPI_DesvioETA"] = np.clip(desvio / LIMIAR_ETA_DIAS, 0, 1).fillna(0).values
    else:
        df["KPI_DesvioETA"] = 0.0

    df["KPI_Completude"] = 1 - df.notna().mean(axis=1)

    if "EXT_Situacao" in df.columns:
        ativa = df["EXT_Situacao"].astype(str).str.lower().str.contains("ativa")
        df["KPI_Ext_ConsistenciaCadastral"] = (~ativa).astype(float)
    else:
        df["KPI_Ext_ConsistenciaCadastral"] = 0.0

    if "EXT_IdadeEmpresa" in df.columns:
        idade = pd.to_numeric(df["EXT_IdadeEmpresa"], errors="coerce")
        df["KPI_Ext_IdadeEmpresa"] = np.clip(1 - idade / 10, 0, 1).fillna(0.5).values
    else:
        df["KPI_Ext_IdadeEmpresa"] = 0.0

    if "EXT_ScoreCredito" in df.columns:
        sc = pd.to_numeric(df["EXT_ScoreCredito"], errors="coerce")
        df["KPI_Ext_ScoreCredito"] = (1 - sc).fillna(0.5).values
    else:
        df["KPI_Ext_ScoreCredito"] = 0.0

    return df


def calcular_outliers(df: pd.DataFrame) -> np.ndarray:
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


def calcular_risco(df: pd.DataFrame) -> pd.DataFrame:
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
        "score_credito":     "KPI_Ext_ScoreCredito",
        "padrao_char":       "KPI_PadraoChar",
    }
    score = np.zeros(len(df))
    for kpi, peso in PESO_KPIS.items():
        col = mapping.get(kpi)
        if col and col in df.columns:
            score += peso * df[col].fillna(0).values

    df["RiskScore"] = np.clip(score * 100, 0, 100).round(2)
    bins = [0, 25, 50, 75, 100.01]
    labels = ["BAIXO", "MEDIO", "ALTO", "CRITICO"]
    cores = {"BAIXO": "#C6EFCE", "MEDIO": "#FFEB9C",
             "ALTO": "#FFC7CE", "CRITICO": "#C00000"}
    df["RiskLevel"] = pd.cut(df["RiskScore"], bins=bins, labels=labels, right=False)
    df["RiskColor"] = df["RiskLevel"].map(cores).fillna("#FFFFFF")
    return df


# ==================================================================
# 7. PREPROCESSADOR + ENSEMBLE ENRIQUECIDO
# ==================================================================

class TextConcat(BaseEstimator):
    def __init__(self, colunas): self.colunas = colunas
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.colunas if c in X.columns]
            if not cols:
                return np.array([""] * len(X))
            return X[cols].fillna("").astype(str).agg(" ".join, axis=1).values
        return np.array([" ".join(map(str, r)) for r in X])


def construir_preprocessador(df: pd.DataFrame) -> ColumnTransformer:
    txt = [c for c in (COLUNAS_TEXTO + COLUNAS_EXT_TEXTO + ["TextoPDF"]) if c in df.columns]
    cat = [c for c in (COLUNAS_CAT + COLUNAS_EXT_CAT) if c in df.columns]
    num = [c for c in (COLUNAS_NUM + COLUNAS_EXT_NUM + COLUNAS_KPI) if c in df.columns]

    transformers = []
    if txt:
        transformers.append((
            "tfidf",
            Pipeline([
                ("concat", TextConcat(txt)),
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2), min_df=1, max_features=8000,
                    sublinear_tf=True, strip_accents="unicode",
                    analyzer="word",
                )),
            ]), txt,
        ))
        transformers.append((
            "char",
            Pipeline([
                ("concat", TextConcat(txt)),
                ("cnt", CountVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5),
                    min_df=1, max_features=3000,
                )),
            ]), txt,
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
            ]), num,
        ))

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


class NT1Ensemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble construido em cima do roadmap:
      - SGD log_loss
      - SGD modified_huber
      - PassiveAggressive (calibrado)
      - ComplementNB
      - LinearSVC (calibrado)
    Ponderado por validacao cruzada (weight = accuracy).
    """

    def __init__(self, alpha: float = 1e-4, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state

    def _build(self) -> dict:
        base_sgd = dict(
            alpha=self.alpha, max_iter=3000, tol=1e-4,
            learning_rate="adaptive", eta0=0.01,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, class_weight="balanced",
            average=True, random_state=self.random_state,
        )
        return {
            "sgd_log":    SGDClassifier(loss="log_loss",       **base_sgd),
            "sgd_huber":  SGDClassifier(loss="modified_huber", **base_sgd),
            "pa":         PassiveAggressiveClassifier(
                C=1.0, max_iter=2000, class_weight="balanced",
                average=True, random_state=self.random_state),
            "cnb":        ComplementNB(alpha=0.3),
            "svc":        CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced",
                          max_iter=5000, random_state=self.random_state),
                cv=3, method="sigmoid"),
        }

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = self._build()
        self._trained = {}
        self._weights = {}

        n_min = np.min(np.bincount(y)) if len(self.classes_) > 1 else 0
        cv = min(3, n_min) if n_min >= 2 else None

        for nome, m in self.models_.items():
            try:
                m.fit(X, y)
                w = 1.0
                if cv and cv >= 2:
                    try:
                        sc = cross_val_score(
                            clone(m), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
                        w = float(np.clip(sc.mean(), 0.1, 1.0))
                    except Exception:
                        pass
                self._trained[nome] = m
                self._weights[nome] = w
                print(f"   [OK] {nome:<10} treinado (peso={w:.3f})")
            except Exception as e:
                print(f"   [ERRO] {nome}: {e}")

        if not self._trained:
            raise RuntimeError("Nenhum modelo conseguiu treinar.")

        tot = sum(self._weights.values()) or 1.0
        self._weights = {k: v / tot for k, v in self._weights.items()}
        return self

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for nome, m in self._trained.items():
            if not hasattr(m, "predict_proba"):
                continue
            p = m.predict_proba(X)
            if not np.array_equal(m.classes_, self.classes_):
                p_full = np.zeros((X.shape[0], len(self.classes_)))
                for i, c in enumerate(m.classes_):
                    idx = np.where(self.classes_ == c)[0]
                    if len(idx):
                        p_full[:, idx[0]] = p[:, i]
                p = p_full
            probs += p * self._weights[nome]
        probs /= probs.sum(axis=1, keepdims=True) + 1e-12
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
        if not hasattr(self, "_trained") or not self._trained:
            base_sgd = dict(average=True, class_weight="balanced",
                            random_state=self.random_state)
            self._trained = {
                "sgd_log":   SGDClassifier(loss="log_loss",       **base_sgd),
                "sgd_huber": SGDClassifier(loss="modified_huber", **base_sgd),
                "pa":        PassiveAggressiveClassifier(
                    average=True, class_weight="balanced",
                    random_state=self.random_state),
            }
            self._weights = {k: 1.0 / 3 for k in self._trained}
        for nome, m in self._trained.items():
            try:
                if not hasattr(m, "classes_"):
                    m.partial_fit(X, y, classes=self.classes_)
                else:
                    m.partial_fit(X, y)
            except Exception:
                pass
        return self


# ==================================================================
# 8. ROADMAP - TREINAMENTO EM FASES
# ==================================================================

def treinar_incremental(modelo, X, y, batch_size: int = BATCH_SIZE) -> list[dict]:
    classes = np.unique(y)
    n = X.shape[0]
    hist = []
    for i in range(0, n, batch_size):
        Xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        if len(Xb) == 0:
            continue
        try:
            modelo.partial_fit(Xb, yb, classes=classes)
            y_pred = modelo.predict(Xb)
            acc = accuracy_score(yb, y_pred)
            f1 = f1_score(yb, y_pred, average="macro", zero_division=0)
            hist.append({"batch": i // batch_size + 1, "acc": acc, "f1": f1})
            print(f"   Batch {hist[-1]['batch']:>3} | acc={acc:.4f} | f1={f1:.4f}")
        except Exception:
            continue
    return hist


def avaliar_modelo(modelo, X, y) -> dict:
    try:
        y_pred = modelo.predict(X)
    except Exception:
        return {}
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
                log_loss(y, y_proba, labels=modelo.classes_)), 4)
        except Exception:
            pass
    except Exception:
        pass
    return metricas


# ==================================================================
# 9. HISTORICO DE SESSOES + DETECCAO DE PADRAO (F7/F8)
# ==================================================================

def carregar_historico() -> list[dict]:
    os.makedirs(PASTA_HIST, exist_ok=True)
    if os.path.exists(HIST_PATH):
        try:
            with open(HIST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def salvar_sessao(sessao: dict):
    hist = carregar_historico()
    hist.append(sessao)
    with open(HIST_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False, default=str)
    return hist


def detectar_padrao_entre_sessoes(hist: list[dict]) -> dict:
    if len(hist) < 3:
        return {"aviso": f"Historico curto ({len(hist)} sessoes). Continue treinando."}

    ultimas = hist[-min(len(hist), N_SESSOES_PADRAO):]
    accs = [s.get("metricas", {}).get("accuracy", 0) for s in ultimas]
    tops = [s.get("metricas", {}).get("top_5", 0) for s in ultimas]

    kpi_counter = Counter()
    for s in ultimas:
        for k, v in (s.get("kpis_medio_top10", {}) or {}).items():
            if v > 0.5:
                kpi_counter[k] += 1

    tendencia = "estavel"
    if len(accs) >= 3:
        if accs[-1] > accs[0] + 0.02:
            tendencia = "melhora"
        elif accs[-1] < accs[0] - 0.02:
            tendencia = "degradacao"

    limiar = 0.6 * len(ultimas)
    kpis_palpite = [k for k, c in kpi_counter.items() if c >= limiar]

    return {
        "sessoes_analisadas": len(ultimas),
        "accuracy_media":     round(float(np.mean(accs)), 4),
        "accuracy_std":       round(float(np.std(accs)), 4),
        "top5_media":         round(float(np.mean(tops)), 4),
        "tendencia":          tendencia,
        "kpis_recorrentes":   kpis_palpite,
        "assinatura":         hashlib.md5(
            ",".join(sorted(kpis_palpite)).encode()).hexdigest()[:12],
        "pronto_para_palpite": len(hist) >= N_SESSOES_PADRAO,
    }


def salvar_padrao(padrao: dict):
    with open(PADRAO_PATH, "w", encoding="utf-8") as f:
        json.dump(padrao, f, indent=2, ensure_ascii=False, default=str)


# ==================================================================
# 10. EXPORTACAO XLSX COLORIDO
# ==================================================================

def exportar_xlsx_colorido(df: pd.DataFrame, caminho: str):
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows

    wb = Workbook(); ws = wb.active; ws.title = "NT1 Auditoria"
    cores = {"BAIXO": "C6EFCE", "MEDIO": "FFEB9C",
             "ALTO": "FFC7CE", "CRITICO": "C00000"}

    cols = list(df.columns)
    ws.append(cols)
    for c in range(1, len(cols) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center")

    col_risk = cols.index("RiskLevel") + 1 if "RiskLevel" in cols else None
    col_score = cols.index("RiskScore") + 1 if "RiskScore" in cols else None

    for r in dataframe_to_rows(df, index=False, header=False):
        ws.append(r)
        row_idx = ws.max_row
        if col_risk:
            val = str(ws.cell(row=row_idx, column=col_risk).value or "").upper()
            fill = PatternFill("solid", fgColor=cores.get(val, "FFFFFF"))
            fc = "FFFFFF" if val == "CRITICO" else "000000"
            for c in range(1, len(cols) + 1):
                ws.cell(row=row_idx, column=c).fill = fill
            ws.cell(row=row_idx, column=col_risk).font = Font(bold=True, color=fc)
        if col_score:
            ws.cell(row=row_idx, column=col_score).number_format = "0.00"

    for i, col in enumerate(cols, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
            max(12, min(30, len(col) + 4))

    wb.save(caminho)
    print(f"[SAIDA] XLSX colorido: {caminho}")


# ==================================================================
# 11. PIPELINE PRINCIPAL
# ==================================================================

def main():
    for p in (PASTA_SAIDA, PASTA_CACHE, PASTA_HIST):
        os.makedirs(p, exist_ok=True)

    print("=" * 78)
    print("  NT-1 EXTRACTOR - Roadmap Evolutivo Multi-API + Char-Ngram + Padrao")
    print("=" * 78)
    t0 = time.time()

    # F0: menu
    resp = menu_inicial()
    if resp == "Cancelar" or resp is None:
        print("Cancelado.")
        return
    modo_completo = (resp == "Completo")

    # F1: carga local
    print("\n[F0] Carga local (XLSX/Parquet/PDF/TXT)...")
    df, relatorios = carregar_arquivos_reais(PASTA_DADOS)
    if df is None or df.empty:
        return

    # F1: enriquecimento
    if modo_completo:
        print("\n[F1] Enriquecimento multi-API...")
        df = enriquecer_cnpjs(df)
    else:
        print("\n[F1] Enriquecimento pulado (modo rapido).")

    # F2: KPIs de qualidade
    print("\n[F2] KPIs de qualidade...")
    kpis_q = calcular_kpis_qualidade(df, relatorios)
    for k, v in kpis_q.items():
        print(f"   - {k}: {v}")

    kpi_cols_df = calcular_kpi_por_coluna(df)
    kpi_cols_df.to_csv(os.path.join(PASTA_SAIDA, "kpi_por_coluna.csv"), index=False)
    print(f"   - {len(kpi_cols_df)} colunas mapeadas -> kpi_por_coluna.csv")

    # F3: KPIs de negocio
    print("\n[F3] KPIs de negocio (locais + externos)...")
    df = calcular_kpis_negocio(df)

    # F3: char-ngrams
    print("\n[F3] Mineracao de padroes de caracteres (palpites)...")
    df, top_padroes = extrair_padroes_char(df)
    print(f"   - {len(top_padroes)} padroes recorrentes capturados")

    # Risco
    print("\n[F4] Score de risco...")
    df = calcular_risco(df)
    print(df["RiskLevel"].value_counts().to_string())

    # Features
    cols_disp = [c for c in COLUNAS_ALVO if c in df.columns]
    print(f"\n[INFO] Features ({len(cols_disp)} colunas-alvo)")

    preproc = construir_preprocessador(df)
    X = preproc.fit_transform(df)
    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    print(f"   Dimensao: {X_dense.shape}")

    # Label = CNPJ
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df[CAMPO_CNPJ].astype(str).values)
    n_classes = len(label_enc.classes_)
    print(f"   Classes (CNPJs): {n_classes}")

    if n_classes < 2:
        alertar("Precisa de pelo menos 2 CNPJs distintos.", "Aviso")
        return

    # Split
    strat = y if np.min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_dense, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat)
    print(f"\n   Treino: {len(X_tr)} | Teste: {len(X_te)}")

    # F4: warmup ensemble
    print("\n[F4] Warmup - ensemble NT1...")
    ens = NT1Ensemble(alpha=1e-4).fit(X_tr, y_tr)

    # F5: incremental
    print("\n[F5] Treinamento incremental...")
    inc = NT1Ensemble(alpha=1e-4, random_state=RANDOM_STATE)
    inc._trained = {}
    hist_batches = treinar_incremental(inc, X_tr, y_tr)

    # F6: refinamento
    print("\n[F6] Refinamento (calibracao + avaliacao)...")
    try:
        cal = CalibratedClassifierCV(
            estimator=LogisticRegression(
                max_iter=3000, class_weight="balanced",
                random_state=RANDOM_STATE, C=1.0),
            method="sigmoid", cv=3)
        cal.fit(X_tr, y_tr)
        usar_cal = True
        print("   [OK] Calibracao OK")
    except Exception as e:
        print(f"   [AVISO] Calibracao falhou: {e}")
        usar_cal = False

    # Avaliacao
    m_ens = avaliar_modelo(ens, X_te, y_te)
    m_inc = avaliar_modelo(inc, X_te, y_te) if hist_batches else {}
    m_cal = avaliar_modelo(cal, X_te, y_te) if usar_cal else {}

    def _print(m, nome):
        if not m:
            return
        print(f"\n-- {nome} --")
        for k, v in m.items():
            print(f"   - {k:>10}: {v}")

    _print(m_ens, "NT1 Ensemble")
    _print(m_cal, "LogReg Calibrado")
    _print(m_inc, "SGD Incremental")

    candidatos = {"ensemble": (ens, m_ens)}
    if m_cal: candidatos["calibrado"] = (cal, m_cal)
    if m_inc: candidatos["incremental"] = (inc, m_inc)

    melhor_nome = max(
        candidatos,
        key=lambda k: candidatos[k][1].get("top_5",
                                           candidatos[k][1].get("accuracy", 0)))
    melhor_modelo, melhor_metricas = candidatos[melhor_nome]
    print(f"\n[INFO] Melhor modelo: {melhor_nome}")

    # F7: historico
    print("\n[F7] Registrando sessao no historico...")
    top10 = df.sort_values("RiskScore", ascending=False).head(10)
    kpis_medio_top10 = {
        col: round(float(pd.to_numeric(top10[col], errors="coerce").fillna(0).mean()), 4)
        for col in COLUNAS_KPI if col in top10.columns
    }
    sessao = {
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_linhas":        len(df),
        "n_classes":       int(n_classes),
        "modo":            "completo" if modo_completo else "rapido",
        "melhor_modelo":   melhor_nome,
        "metricas":        melhor_metricas,
        "distribuicao_risco": df["RiskLevel"].value_counts().to_dict(),
        "kpis_medio_top10": kpis_medio_top10,
        "top_padroes_char": dict(list(top_padroes.items())[:20]),
    }
    hist_completo = salvar_sessao(sessao)
    print(f"   - Historico: {len(hist_completo)} sessoes")

    padrao = detectar_padrao_entre_sessoes(hist_completo)
    salvar_padrao(padrao)
    print(f"   - Padrao: {padrao.get('tendencia', '-')} | "
          f"pronto={padrao.get('pronto_para_palpite', False)}")

    # Persistencia
    joblib.dump({
        "modelo":         melhor_modelo,
        "preprocessador": preproc,
        "label_encoder":  label_enc,
        "metricas":       melhor_metricas,
        "colunas_alvo":   cols_disp,
        "tipo":           melhor_nome,
        "usa_externo":    modo_completo,
        "top_padroes":    top_padroes,
        "sessao":         len(hist_completo),
    }, MODELO_PATH, compress=3)
    print(f"\n[SAIDA] Modelo: {MODELO_PATH}")

    # Anexa previsoes
    try:
        y_proba_full = melhor_modelo.predict_proba(X_dense)
        top_idx = np.argsort(y_proba_full, axis=1)[:, ::-1][:, :5]
        df["CNPJ_Previsto"]  = label_enc.inverse_transform(y_proba_full.argmax(axis=1))
        df["CNPJ_Confianca"] = np.round(y_proba_full.max(axis=1), 4)
        for r in range(5):
            df[f"CNPJ_Top{r+1}"] = label_enc.inverse_transform(top_idx[:, r])
            df[f"CNPJ_Top{r+1}_p"] = np.round(
                np.take_along_axis(y_proba_full, top_idx[:, r:r+1], axis=1).ravel(), 4)
    except Exception as e:
        print(f"[AVISO] Previsoes: {e}")

    # Exportacao
    ts = time.strftime("%Y%m%d_%H%M%S")
    parquet_path = os.path.join(PASTA_SAIDA, f"nt1_auditoria_{ts}.parquet")
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    print(f"[SAIDA] Parquet: {parquet_path}")

    top_risco = df.sort_values("RiskScore", ascending=False).head(N_TOP_RISCO_XLSX)
    xlsx_path = os.path.join(PASTA_SAIDA, f"nt1_risco_{ts}.xlsx")
    exportar_xlsx_colorido(top_risco, xlsx_path)

    kpi_path = os.path.join(PASTA_SAIDA, f"nt1_kpis_{ts}.json")
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump({
            "qualidade":          kpis_q,
            "risco_distribuicao": df["RiskLevel"].value_counts().to_dict(),
            "metricas_ml":        melhor_metricas,
            "tipo_melhor_modelo": melhor_nome,
            "classes_cnpj":       int(n_classes),
            "usa_externo":        bool(modo_completo),
            "colunas_ext":        int(sum(1 for c in df.columns if c.startswith("EXT_"))),
            "top_padroes_char":   dict(list(top_padroes.items())[:30]),
            "padrao_entre_sessoes": padrao,
            "sessao_numero":      len(hist_completo),
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"[SAIDA] KPIs JSON: {kpi_path}")

    # Resumo
    dist = df["RiskLevel"].value_counts().to_dict()
    alertar(
        f"NT-1 EXTRACTOR - sessao #{len(hist_completo)}\n\n"
        f"Linhas: {len(df)}  |  CNPJs: {n_classes}\n"
        f"Modo: {'Completo' if modo_completo else 'Rapido'}\n"
        f"Colunas EXT_: {sum(1 for c in df.columns if c.startswith('EXT_'))}\n\n"
        f"Risco:\n" + "\n".join(f"  - {k}: {v}" for k, v in dist.items()) +
        f"\n\nMelhor: {melhor_nome}\n" +
        "\n".join(f"  - {k}: {v}" for k, v in melhor_metricas.items()) +
        f"\n\nHistorico: {len(hist_completo)} sessoes\n"
        f"Tendencia: {padrao.get('tendencia', '-')}\n"
        f"Pronto p/ palpite: {padrao.get('pronto_para_palpite', False)}",
        "NT-1 Concluido"
    )

    # F9: consulta + captura PDF
    while True:
        acao = pyautogui.confirm(
            "O que deseja agora?",
            "NT-1 Console",
            ["Consultar CNPJ", "Capturar PDF", "Ver Padrao", "Sair"]
        )
        if acao in (None, "Sair"):
            break

        if acao == "Consultar CNPJ":
            valores = coletar_valores_alvo(COLUNAS_ALVO)
            if valores is None:
                continue
            try:
                Xq = preproc.transform(pd.DataFrame([valores]))
                Xq = Xq.toarray() if hasattr(Xq, "toarray") else Xq
                probs = melhor_modelo.predict_proba(Xq)[0]
                top = np.argsort(probs)[::-1][:5]
                linhas = "\n".join(
                    f"{r+1}o  CNPJ {label_enc.inverse_transform([i])[0]}  "
                    f"(p={probs[i]:.4f})"
                    for r, i in enumerate(top))
                alertar(f"Top-5 CNPJs previstos:\n\n{linhas}", "Captura de CNPJ")
            except Exception as e:
                alertar(f"Erro: {e}", "Aviso")

        elif acao == "Capturar PDF":
            txt = captura_clipboard()
            if not txt:
                continue
            cnpjs = re.findall(r"\d{14}", re.sub(r"\D", "", txt))
            cnpjs_fmt = re.findall(
                r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}", txt)
            todos = list(dict.fromkeys(
                [c.zfill(14) for c in cnpjs] +
                [re.sub(r"\D", "", c).zfill(14) for c in cnpjs_fmt]
            ))
            _, padroes_pdf = extrair_padroes_char(
                pd.DataFrame({"TextoPDF": [txt]}))
            msg = (f"Captura de PDF\n\n"
                   f"CNPJs detectados: {len(todos)}\n"
                   + "\n".join(f"  - {c}" for c in todos[:20]) +
                   f"\n\nPadroes char (top 10):\n" +
                   "\n".join(f"  - {k} ({v})"
                             for k, v in list(padroes_pdf.items())[:10]))
            alertar(msg, "Extracao PDF")

        elif acao == "Ver Padrao":
            hist = carregar_historico()
            if not hist:
                alertar("Sem historico ainda.", "Historico")
                continue
            linhas = []
            for i, s in enumerate(hist[-10:], 1):
                m = s.get("metricas", {})
                linhas.append(
                    f"#{i:>2} {s['timestamp']} | "
                    f"acc={m.get('accuracy', '-')} "
                    f"top5={m.get('top_5', '-')} "
                    f"({s.get('melhor_modelo', '-')})")
            ultimo_padrao = detectar_padrao_entre_sessoes(hist)
            alertar(
                "Ultimas sessoes:\n\n" + "\n".join(linhas) +
                f"\n\nTendencia: {ultimo_padrao.get('tendencia', '-')}\n"
                f"KPIs recorrentes: {ultimo_padrao.get('kpis_recorrentes', [])}\n"
                f"Pronto p/ palpite: {ultimo_padrao.get('pronto_para_palpite', False)}",
                "Historico NT-1"
            )

    print(f"\nTempo total: {time.time() - t0:.2f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
