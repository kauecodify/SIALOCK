# =====================================================================
#  EXTRACT V.3.3  
#  Gate 3 Pipeline - Incremental Batch Processing with Automatic Recovery
#  =====================================================================
"""
Pipeline incremental para processamento de bases em lotes (batches).
Gera arquivos de palpite (palpite0.xlsx, palpite1.xlsx, ...) automaticamente.
Suporta retomada automática do processamento e separação total entre bases.

Estrutura:
  gate3/
    ├── base2/
    │   ├── entrada/          # Arquivos de entrada (base2.csv)
    │   ├── palpite/          # Arquivos de palpite gerados
    │   │   ├── palpite0.xlsx
    │   │   ├── palpite1.xlsx
    │   │   └── ...
    │   ├── treinamento/      # Dados para treinamento incremental
    │   ├── modelos/          # Modelos persistidos
    │   └── logs/             # Logs de processamento
    │
    └── base3/
        ├── entrada/
        ├── palpite/
        ├── treinamento/
        ├── modelos/
        └── logs/

Uso:
    python gate3/extract_v.3.3.py --base base2 --batch-size 1000
    python gate3/extract_v.3.3.py --base base3 --batch-size 1000
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import shutil
import hashlib
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    top_k_accuracy_score, log_loss
)
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")

# =====================================================================
#  CONSTANTES
# =====================================================================
APP_NOME = "EXTRACT V.3.3"
APP_SUB = "Gate 3 Pipeline - Incremental Batch Processing"
APP_VERSAO = "3.3.0"

# Caminhos base
PASTA_GATE3 = "./gate3"
PASTA_DADOS = "./bases"

# Configuracoes de processamento
BATCH_SIZE = 1000  # Tamanho padrao do lote
CHUNK_SIZE = 50000  # Para leitura streaming
MAX_CLASSES = 20000
RANDOM_STATE = 42
TOP_K = 5  # Top-N candidatos
MAX_PALPITES_FINAL = 50  # Maximo de palpites por base na entrega final

# KPIs e pesos
PESO_KPIS = {
    "subvalorizacao": 0.20, "disparidade_hs": 0.12,
    "surge_volume": 0.12, "desvio_eta": 0.08,
    "completude": 0.04, "outlier": 0.05,
    "consist_cadastral": 0.13, "idade_empresa": 0.08,
    "score_credito": 0.10, "padrao_char": 0.08,
}

COLUNAS_KPI = [
    "KPI_Subvalorizacao", "KPI_DisparidadeHS", "KPI_SurgeVolume",
    "KPI_DesvioETA", "KPI_Completude", "KPI_Outlier",
    "KPI_Ext_ConsistenciaCadastral", "KPI_Ext_IdadeEmpresa",
    "KPI_Ext_ScoreCredito", "KPI_PadraoChar", "RiskScore",
]

# Limiares
LIMIAR_SURGE = 3.0
LIMIAR_ETA_DIAS = 7
HASH_WORD_FEATURES = 2 ** 18
HASH_CHAR_FEATURES = 2 ** 17

# =====================================================================
#  CONFIG
# =====================================================================

@dataclass
class Config:
    base: str = "base2"  # base2 ou base3
    batch_size: int = BATCH_SIZE
    chunk_size: int = CHUNK_SIZE
    max_palpites: int = MAX_PALPITES_FINAL
    top_k: int = TOP_K
    random_state: int = RANDOM_STATE
    max_classes: int = MAX_CLASSES
    
    # Caminhos
    pasta_gate3: str = PASTA_GATE3
    pasta_dados: str = PASTA_DADOS
    
    # Opcoes
    dry_run: bool = False
    headless: bool = True
    retomar: bool = True  # Retomar processamento de onde parou
    validar: bool = True  # Aplicar validacao e score
    
    def aplicar(self):
        global PASTA_GATE3, PASTA_DADOS
        PASTA_GATE3 = self.pasta_gate3
        PASTA_DADOS = self.pasta_dados


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        prog="extract-v3.3",
        description=f"{APP_NOME} \u00b7 {APP_SUB} \u2014 v{APP_VERSAO}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python gate3/extract_v.3.3.py --base base2\n"
            "  python gate3/extract_v.3.3.py --base base3 --batch-size 500\n"
            "  python gate3/extract_v.3.3.py --base base2 --retomar false\n"
        ),
    )
    p.add_argument("--base", choices=["base2", "base3"], default="base2")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--max-palpites", type=int, default=MAX_PALPITES_FINAL)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--retomar", action="store_true", default=True)
    p.add_argument("--no-validar", action="store_true")
    
    ns = p.parse_args(argv)
    cfg = Config()
    cfg.base = ns.base
    cfg.batch_size = ns.batch_size
    cfg.chunk_size = ns.chunk_size
    cfg.max_palpites = ns.max_palpites
    cfg.top_k = ns.top_k
    cfg.dry_run = ns.dry_run
    cfg.headless = ns.headless
    cfg.retomar = not ns.no_retomar if hasattr(ns, 'no_retomar') else ns.retomar
    cfg.validar = not ns.no_validar
    cfg.aplicar()
    return cfg


# =====================================================================
#  UI SIMPLES (sem cores para compatibilidade)
# =====================================================================

class UI:
    @classmethod
    def info(cls, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [INFO] {msg}")

    @classmethod
    def ok(cls, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [OK] {msg}")

    @classmethod
    def warn(cls, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [WARN] {msg}")

    @classmethod
    def erro(cls, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [ERRO] {msg}")

    @classmethod
    def progresso(cls, atual: int, total: int, prefixo: str = "", extra: str = ""):
        if total <= 0:
            return
        pct = min(1.0, atual / total)
        cheios = int(pct * 30)
        barra = "#" * cheios + "-" * (30 - cheios)
        print(f"\r  [{prefixo}] [{barra}] {pct*100:5.1f}% {extra}", end="", flush=True)

    @classmethod
    def fim_progresso(cls, msg: str = ""):
        print()
        if msg:
            cls.ok(msg)

    @classmethod
    def banner(cls, titulo: str):
        print("=" * 78)
        print(f"  {titulo}")
        print("=" * 78)

    @classmethod
    def secao(cls, num: str, titulo: str):
        print()
        print(f"[{num}] {titulo}")
        print("-" * 78)


# =====================================================================
#  ESTRUTURA DE DIRETORIOS
# =====================================================================

class Gate3Paths:
    """Gerencia caminhos para uma base especifica (base2 ou base3)."""
    
    def __init__(self, base: str, pasta_gate3: str = PASTA_GATE3):
        self.base = base
        self.pasta_gate3 = pasta_gate3
        self.pasta_base = os.path.join(pasta_gate3, base)
        
        # Criar estrutura de diretorios
        self.entrada = os.path.join(self.pasta_base, "entrada")
        self.palpite = os.path.join(self.pasta_base, "palpite")
        self.treinamento = os.path.join(self.pasta_base, "treinamento")
        self.modelos = os.path.join(self.pasta_base, "modelos")
        self.logs = os.path.join(self.pasta_base, "logs")
        
        # Arquivos importantes
        self.progresso_file = os.path.join(self.pasta_base, "progresso.json")
        self.entrega_file = os.path.join(self.pasta_base, f"ENTREGA_{base}.xlsx")
        self.modelo_file = os.path.join(self.modelos, f"modelo_gate3_{base}.joblib")
        self.log_file = os.path.join(self.logs, f"gate3_{base}_{datetime.now():%Y%m%d}.log")
        
        # Criar diretorios
        for pasta in [self.pasta_base, self.entrada, self.palpite, 
                       self.treinamento, self.modelos, self.logs]:
            os.makedirs(pasta, exist_ok=True)
    
    def get_palpite_path(self, lote: int) -> str:
        """Retorna o caminho para o arquivo de palpite do lote."""
        return os.path.join(self.palpite, f"palpite{lote}.xlsx")
    
    def get_treinamento_path(self, lote: int) -> str:
        """Retorna o caminho para o arquivo de treinamento do lote."""
        return os.path.join(self.treinamento, f"treinamento_{lote}.parquet")


# =====================================================================
#  PROGRESSO
# =====================================================================

@dataclass
class Progresso:
    """Armazena o estado de progresso do processamento."""
    base: str
    ultimo_lote: int = 0
    ultimo_registro: int = 0
    total_registros: int = 0
    status: str = "pendente"  # pendente, em_execucao, concluido, erro
    inicio: str = ""
    fim: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, dados: dict) -> "Progresso":
        return cls(**{k: v for k, v in dados.items() if k in cls.__dataclass_fields__})
    
    def salvar(self, caminho: str):
        """Salva o progresso em arquivo JSON."""
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def carregar(cls, caminho: str) -> Optional["Progresso"]:
        """Carrega o progresso de arquivo JSON."""
        if not os.path.exists(caminho):
            return None
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)
            return cls.from_dict(dados)
        except Exception as e:
            UI.warn(f"Erro ao carregar progresso: {e}")
            return None


# =====================================================================
#  CARREGAMENTO STREAMING
# =====================================================================

def iterar_base_streaming(caminho_base: str, chunk_size: int = CHUNK_SIZE) -> Tuple[str, pd.DataFrame]:
    """
    Itera sobre os arquivos da base em chunks.
    Suporta CSV, XLSX e Parquet.
    """
    if not os.path.isdir(caminho_base):
        UI.erro(f"Pasta de entrada nao encontrada: {caminho_base}")
        return
    
    # Encontrar todos os arquivos de entrada
    arquivos = []
    for ext in ["*.csv", "*.xlsx", "*.parquet"]:
        arquivos.extend(Path(caminho_base).glob(ext))
    
    if not arquivos:
        UI.erro(f"Nenhum arquivo de entrada encontrado em {caminho_base}")
        return
    
    UI.info(f"Encontrados {len(arquivos)} arquivos de entrada")
    
    for arq in sorted(arquivos):
        suf = arq.suffix.lower()
        try:
            if suf == ".csv":
                for chunk in pd.read_csv(arq, chunksize=chunk_size, dtype=str,
                                         sep=None, engine="python"):
                    chunk.columns = [str(c).strip() for c in chunk.columns]
                    yield arq.name, chunk
            elif suf == ".xlsx":
                df = pd.read_excel(arq, dtype=str)
                df.columns = [str(c).strip() for c in df.columns]
                for i in range(0, len(df), chunk_size):
                    yield arq.name, df.iloc[i:i + chunk_size].copy()
            elif suf == ".parquet":
                try:
                    import pyarrow.parquet as pq
                    pf = pq.ParquetFile(arq)
                    for batch in pf.iter_batches(batch_size=chunk_size):
                        df = batch.to_pandas()
                        df.columns = [str(c).strip() for c in df.columns]
                        yield arq.name, df
                except Exception:
                    df = pd.read_parquet(arq, engine="pyarrow")
                    df.columns = [str(c).strip() for c in df.columns]
                    for i in range(0, len(df), chunk_size):
                        yield arq.name, df.iloc[i:i + chunk_size].copy()
        except Exception as e:
            UI.warn(f"Erro ao processar {arq.name}: {e}")


def contar_registros_total(caminho_base: str) -> int:
    """Conta o total de registros em todos os arquivos da base."""
    total = 0
    for arq in Path(caminho_base).glob("*.csv"):
        try:
            # Contar linhas sem carregar tudo
            with open(arq, "r", encoding="utf-8", errors="ignore") as f:
                total += sum(1 for _ in f) - 1  # -1 para o cabecalho
        except Exception:
            pass
    
    for arq in Path(caminho_base).glob("*.xlsx"):
        try:
            df = pd.read_excel(arq, dtype=str)
            total += len(df)
        except Exception:
            pass
    
    for arq in Path(caminho_base).glob("*.parquet"):
        try:
            df = pd.read_parquet(arq, engine="pyarrow")
            total += len(df)
        except Exception:
            pass
    
    return total


# =====================================================================
#  PREPROCESSAMENTO E FEATURES
# =====================================================================

class TextConcat(BaseEstimator):
    def __init__(self, colunas):
        self.colunas = colunas
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.colunas if c in X.columns]
            if not cols:
                return np.array([""] * len(X))
            return X[cols].fillna("").astype(str).agg(" ".join, axis=1).values
        return np.array([" ".join(map(str, r)) for r in X])


class ColunasNumericas(BaseEstimator):
    def __init__(self, colunas):
        self.colunas = colunas
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.colunas if c in X.columns]
            if not cols:
                return np.zeros((len(X), 0))
            return X[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        return np.asarray(X)


def detectar_coluna_id(df: pd.DataFrame) -> str:
    """Detecta automaticamente a coluna de identificador (chave_item ou CNPJ)."""
    candidatos = ["chave_item", "CNPJ", "cnpj", "id", "codigo"]
    for c in df.columns:
        if str(c).lower().strip() in candidatos:
            return c
    # Se nao encontrar, usar a primeira coluna
    return df.columns[0]


def normalizar_id_cnpj(serie: pd.Series) -> pd.Series:
    """Normaliza CNPJ: remove nao-numericos e completa com zeros."""
    return serie.astype(str).str.replace(r"\D", "", regex=True).str.zfill(14)


def calcular_kpis_negocio(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula KPIs de negocio para o DataFrame."""
    df = df.copy()
    
    # Subvalorizacao
    if "ValorImportado" in df.columns and "PesoKg" in df.columns:
        val = pd.to_numeric(df["ValorImportado"], errors="coerce")
        peso = pd.to_numeric(df["PesoKg"], errors="coerce").replace(0, np.nan)
        vk = val / peso
        mediana = vk.median()
        df["KPI_Subvalorizacao"] = np.clip(
            (mediana - vk) / (mediana if mediana else 1), 0, 1
        ).fillna(0).values
    else:
        df["KPI_Subvalorizacao"] = 0.0
    
    # Disparidade HS
    if "HS_Code" in df.columns:
        hs = df["HS_Code"].astype(str).str.replace(r"\D", "", regex=True)
        zeros_finais = hs.str.count(r"0+$") / hs.str.len().replace(0, 1)
        df["KPI_DisparidadeHS"] = np.clip(zeros_finais, 0, 1).fillna(0).values
    else:
        df["KPI_DisparidadeHS"] = 0.0
    
    # Surge Volume
    if "Importador" in df.columns and "ValorImportado" in df.columns:
        val = pd.to_numeric(df["ValorImportado"], errors="coerce")
        df["_val_tmp"] = val
        stats = df.groupby("Importador")["_val_tmp"].agg(["mean", "std"]).reset_index()
        stats.columns = ["Importador", "media", "std"]
        df = df.merge(stats, on="Importador", how="left")
        df["KPI_SurgeVolume"] = np.clip(
            (val - df["media"]) / (df["std"].replace(0, np.nan) * LIMIAR_SURGE),
            0, 1
        ).fillna(0).values
        df = df.drop(columns=["_val_tmp", "media", "std"])
    else:
        df["KPI_SurgeVolume"] = 0.0
    
    # Desvio ETA
    if "ETA" in df.columns and "ETAReal" in df.columns:
        eta = pd.to_datetime(df["ETA"], errors="coerce")
        real = pd.to_datetime(df["ETAReal"], errors="coerce")
        desvio = (real - eta).dt.days.abs()
        df["KPI_DesvioETA"] = np.clip(desvio / LIMIAR_ETA_DIAS, 0, 1).fillna(0).values
    else:
        df["KPI_DesvioETA"] = 0.0
    
    # Completude
    df["KPI_Completude"] = 1 - df.notna().mean(axis=1)
    
    return df


def calcular_outliers(df: pd.DataFrame) -> np.ndarray:
    """Calcula outlier scores usando LocalOutlierFactor."""
    num_cols = [c for c in df.columns 
                if c.startswith("KPI_") or c.startswith("FE_")
                or c in ("ValorImportado", "PesoKg", "CapitalSocial")]
    num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    if not num_cols or len(df) < 10:
        return np.zeros(len(df))
    
    X = df[num_cols].fillna(0).values
    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, len(df) - 1), contamination=0.1)
        pred = lof.fit_predict(X)
        scores = -lof.negative_outlier_factor_
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return np.where(pred == -1, scores, scores * 0.3)
    except Exception:
        return np.zeros(len(df))


def calcular_risco(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula RiskScore e RiskLevel."""
    df = df.copy()
    df["KPI_Outlier"] = calcular_outliers(df)
    
    mapping = {
        "subvalorizacao": "KPI_Subvalorizacao",
        "disparidade_hs": "KPI_DisparidadeHS",
        "surge_volume": "KPI_SurgeVolume",
        "desvio_eta": "KPI_DesvioETA",
        "completude": "KPI_Completude",
        "outlier": "KPI_Outlier",
        "consist_cadastral": "KPI_Ext_ConsistenciaCadastral",
        "idade_empresa": "KPI_Ext_IdadeEmpresa",
        "score_credito": "KPI_Ext_ScoreCredito",
        "padrao_char": "KPI_PadraoChar",
    }
    
    score = np.zeros(len(df))
    for kpi, peso in PESO_KPIS.items():
        col = mapping.get(kpi)
        if col and col in df.columns:
            score += peso * df[col].fillna(0).values
    
    df["RiskScore"] = np.clip(score * 100, 0, 100).round(2)
    
    bins = [0, 25, 50, 75, 100.01]
    labels = ["BAIXO", "MEDIO", "ALTO", "CRITICO"]
    df["RiskLevel"] = pd.cut(df["RiskScore"], bins=bins, labels=labels, right=False)
    
    return df


def construir_preprocessador(df: pd.DataFrame, col_texto: list, col_cat: list, col_num: list) -> ColumnTransformer:
    """Constrói o preprocessador para features de texto, categoricas e numericas."""
    transformers = []
    
    # Features de texto
    if col_texto:
        transformers.append((
            "hash_word", Pipeline([
                ("concat", TextConcat(col_texto)),
                ("hash", HashingVectorizer(n_features=HASH_WORD_FEATURES,
                                           ngram_range=(1, 2), alternate_sign=False,
                                           norm=None, lowercase=True)),
                ("norm", Normalizer(copy=False)),
            ]), col_texto
        ))
        transformers.append((
            "hash_char", Pipeline([
                ("concat", TextConcat(col_texto)),
                ("hash", HashingVectorizer(n_features=HASH_CHAR_FEATURES,
                                           analyzer="char_wb", ngram_range=(3, 5),
                                           alternate_sign=False, norm=None,
                                           lowercase=True)),
                ("norm", Normalizer(copy=False)),
            ]), col_texto
        ))
    
    # Features categoricas
    if col_cat:
        transformers.append((
            "onehot", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="NA")),
                ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
            ]), col_cat
        ))
    
    # Features numericas
    if col_num:
        transformers.append((
            "num", Pipeline([
                ("sel", ColunasNumericas(col_num)),
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), col_num
        ))
    
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


# =====================================================================
#  ENSEMBLE CLASSIFIER
# =====================================================================

class Gate3Ensemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble de classificadores para predicao de CNPJ.
    Suporta partial_fit para treinamento incremental.
    """
    
    def __init__(self, alpha: float = 1e-4, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self._trained = {}
        self._weights = {}
        self.classes_ = None
    
    def _get_models(self, partial: bool = False) -> dict:
        """Retorna os modelos do ensemble."""
        base = dict(
            alpha=self.alpha, max_iter=2000 if not partial else 1,
            tol=1e-4 if not partial else None,
            learning_rate="adaptive", eta0=0.01,
            class_weight="balanced", average=True,
            random_state=self.random_state,
            early_stopping=True if not partial else False,
            validation_fraction=0.1 if not partial else None,
            n_iter_no_change=10 if not partial else None,
        )
        
        if partial:
            return {
                "sgd_log": SGDClassifier(loss="log_loss", **base),
                "sgd_huber": SGDClassifier(loss="modified_huber", **base),
                "pa": PassiveAggressiveClassifier(C=0.5, max_iter=1, tol=None,
                                                  class_weight="balanced", average=True,
                                                  random_state=self.random_state),
                "cnb": ComplementNB(alpha=0.3),
            }
        else:
            return {
                "sgd_log": SGDClassifier(loss="log_loss", **base),
                "sgd_huber": SGDClassifier(loss="modified_huber", **base),
                "pa": PassiveAggressiveClassifier(C=1.0, max_iter=2000,
                                                  class_weight="balanced", average=True,
                                                  random_state=self.random_state),
                "cnb": ComplementNB(alpha=0.3),
                "svc": CalibratedClassifierCV(
                    LinearSVC(C=1.0, class_weight="balanced", max_iter=5000,
                              random_state=self.random_state),
                    cv=3, method="sigmoid"),
            }
    
    def partial_fit(self, X, y, classes=None):
        """Treinamento incremental."""
        if classes is not None:
            self.classes_ = classes
        
        if not self._trained:
            self._trained = self._get_models(partial=True)
            self._weights = {k: 1.0 for k in self._trained}
        
        for nome, m in self._trained.items():
            try:
                if not hasattr(m, "classes_"):
                    m.partial_fit(X, y, classes=self.classes_)
                else:
                    m.partial_fit(X, y)
            except Exception:
                pass
        
        # Atualizar pesos com base na acuracia
        try:
            for nome, m in self._trained.items():
                try:
                    p = m.predict(X)
                    acc = accuracy_score(y, p)
                    self._weights[nome] = 0.7 * self._weights.get(nome, 1.0) + 0.3 * acc
                except Exception:
                    pass
            
            tot = sum(self._weights.values()) or 1.0
            self._weights = {k: v / tot for k, v in self._weights.items()}
        except Exception:
            pass
        
        return self
    
    def fit(self, X, y):
        """Treinamento completo."""
        self.classes_ = np.unique(y)
        self._trained = self._get_models(partial=False)
        self._weights = {}
        
        for nome, m in self._trained.items():
            try:
                m.fit(X, y)
                w = 1.0
                # Validacao cruzada para calcular peso
                try:
                    from sklearn.model_selection import cross_val_score
                    sc = cross_val_score(m, X, y, cv=min(3, len(np.unique(y))),
                                         scoring="accuracy", n_jobs=-1)
                    w = float(np.clip(sc.mean(), 0.1, 1.0))
                except Exception:
                    pass
                self._weights[nome] = w
            except Exception as e:
                UI.warn(f"{nome}: {e}")
        
        # Remover modelos que nao treinaram
        self._trained = {k: v for k, v in self._trained.items() if k in self._weights}
        
        if not self._trained:
            raise RuntimeError("Nenhum modelo treinou.")
        
        tot = sum(self._weights.values()) or 1.0
        self._weights = {k: v / tot for k, v in self._weights.items()}
        
        return self
    
    def predict_proba(self, X, batch_size: int = 2000):
        """Predicao de probabilidades."""
        n = X.shape[0]
        n_c = len(self.classes_)
        probs = np.zeros((n, n_c))
        
        for i in range(0, n, batch_size):
            Xb = X[i:i + batch_size]
            pb = np.zeros((Xb.shape[0], n_c))
            total_w = 0.0
            
            for nome, m in self._trained.items():
                if not hasattr(m, "predict_proba"):
                    continue
                try:
                    p = m.predict_proba(Xb)
                except Exception:
                    continue
                
                # Ajustar classes se necessario
                if not np.array_equal(m.classes_, self.classes_):
                    full = np.zeros((Xb.shape[0], n_c))
                    for j, c in enumerate(m.classes_):
                        idx = np.where(self.classes_ == c)[0]
                        if len(idx):
                            full[:, idx[0]] = p[:, j]
                    p = full
                
                w = self._weights.get(nome, 1.0)
                pb += p * w
                total_w += w
            
            if total_w == 0:
                pb[:, 0] = 1.0
            else:
                pb /= pb.sum(axis=1, keepdims=True) + 1e-12
            
            probs[i:i + batch_size] = pb
        
        return probs
    
    def predict(self, X, batch_size: int = 2000):
        """Predicao de classes."""
        return self.classes_[np.argmax(self.predict_proba(X, batch_size), axis=1)]


# =====================================================================
#  VALIDACAO E SELECAO DE CANDIDATOS
# =====================================================================

def validar_cnpj(cnpj: str) -> bool:
    """
    Valida se um CNPJ e valido (algoritmo de validacao oficial).
    """
    # Remover nao-numericos
    cnpj = re.sub(r"\D", "", str(cnpj)).zfill(14)
    
    if len(cnpj) != 14:
        return False
    
    # Verificar se todos os digitos sao iguais
    if len(set(cnpj)) == 1:
        return False
    
    # Calcular digitos verificadores
    def calcular_dv(cnpj_parcial: str) -> int:
        peso = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        soma = sum(int(cnpj_parcial[i]) * peso[i] for i in range(12))
        dv = 11 - (soma % 11)
        return 0 if dv >= 10 else dv
    
    # Validar primeiro digito verificador
    dv1 = calcular_dv(cnpj[:12])
    if dv1 != int(cnpj[12]):
        return False
    
    # Validar segundo digito verificador
    dv2 = calcular_dv(cnpj[:13])
    if dv2 != int(cnpj[13]):
        return False
    
    return True


def validar_candidatos(df: pd.DataFrame, col_cnpj: str = "CNPJ_Previsto") -> pd.DataFrame:
    """
    Valida os candidatos e aplica regras de negocio.
    """
    df = df.copy()
    
    # Validar CNPJ
    df["CNPJ_Valido"] = df[col_cnpj].apply(validar_cnpj)
    
    # Regras de validacao adicionais
    # 1. RiskScore deve ser >= 50 (confianca minima)
    if "RiskScore" in df.columns:
        df["Score_Valido"] = df["RiskScore"] >= 50
    else:
        df["Score_Valido"] = True
    
    # 2. RiskLevel deve ser BAIXO ou MEDIO
    if "RiskLevel" in df.columns:
        df["Risk_Valido"] = df["RiskLevel"].isin(["BAIXO", "MEDIO"])
    else:
        df["Risk_Valido"] = True
    
    # 3. Confianca minima de 0.6
    if "CNPJ_Confianca" in df.columns:
        df["Confianca_Valida"] = df["CNPJ_Confianca"] >= 0.6
    else:
        df["Confianca_Valida"] = True
    
    # Candidato valido se passar em todas as validacoes
    df["Candidato_Valido"] = (
        df["CNPJ_Valido"] & 
        df["Score_Valido"] & 
        df["Risk_Valido"] & 
        df["Confianca_Valida"]
    )
    
    return df


def seleccionar_melhores_candidatos(df: pd.DataFrame, max_palpites: int = MAX_PALPITES_FINAL) -> pd.DataFrame:
    """
    Seleciona os melhores candidatos com base em:
    1. Candidato valido
    2. Maior confianca
    3. Maior RiskScore
    4. Menor risco
    """
    if df.empty:
        return df
    
    # Filtrar candidatos validados
    validados = df[df["Candidato_Valido"]].copy()
    
    if len(validados) == 0:
        # Se nao ha candidatos validados, pegar os com maior confianca
        validados = df.copy()
    
    # Ordenar por confianca (descendente), RiskScore (descendente)
    validados = validados.sort_values(
        by=["CNPJ_Confianca", "RiskScore"],
        ascending=[False, False]
    )
    
    # Remover duplicados por chave_item (manter o de maior confianca)
    if "chave_item" in validados.columns:
        validados = validados.sort_values("CNPJ_Confianca", ascending=False)
        validados = validados.drop_duplicates(subset=["chave_item"], keep="first")
    
    # Limitar ao max_palpites
    return validados.head(max_palpites)


# =====================================================================
#  EXPORTAÇÃO
# =====================================================================

def exportar_palpite_xlsx(df: pd.DataFrame, caminho: str, lote: int):
    """
    Exporta os resultados do lote para um arquivo XLSX.
    O arquivo contem informacoes detalhadas para auditoria.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils.dataframe import dataframe_to_rows
        
        wb = Workbook()
        ws = wb.active
        ws.title = f"Lote_{lote}"
        
        # Cabecalhos
        cols = list(df.columns)
        ws.append(cols)
        
        # Estilos do cabecalho
        for c in range(1, len(cols) + 1):
            cell = ws.cell(row=1, column=c)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.alignment = Alignment(horizontal="center")
        
        # Dados
        for r in dataframe_to_rows(df, index=False, header=False):
            ws.append(r)
        
        # Ajustar largura das colunas
        for i, col in enumerate(cols, 1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = \
                max(12, min(30, len(col) + 4))
        
        wb.save(caminho)
        UI.ok(f"Exportado: {caminho}")
    except ImportError:
        # Fallback para pandas to_excel
        df.to_excel(caminho, index=False)
        UI.ok(f"Exportado (fallback): {caminho}")
    except Exception as e:
        UI.erro(f"Erro ao exportar {caminho}: {e}")


def exportar_entrega_final(df: pd.DataFrame, caminho: str):
    """
    Exporta a entrega final no formato exigido: chave_item | CNPJ
    """
    try:
        # Selecionar apenas as colunas necessarias
        if "chave_item" in df.columns and "CNPJ_Previsto" in df.columns:
            entrega = df[["chave_item", "CNPJ_Previsto"]].copy()
            entrega.columns = ["chave_item", "CNPJ"]
        elif "chave_item" in df.columns and "CNPJ" in df.columns:
            entrega = df[["chave_item", "CNPJ"]].copy()
        else:
            # Tentar encontrar colunas alternativas
            cols = [c for c in df.columns if "chave" in c.lower() or "item" in c.lower()]
            cnpj_cols = [c for c in df.columns if "cnpj" in c.lower()]
            if cols and cnpj_cols:
                entrega = df[[cols[0], cnpj_cols[0]]].copy()
                entrega.columns = ["chave_item", "CNPJ"]
            else:
                UI.erro("Nao foi possivel identificar colunas chave_item e CNPJ")
                return
        
        entrega.to_excel(caminho, index=False)
        UI.ok(f"Entrega final exportada: {caminho}")
    except Exception as e:
        UI.erro(f"Erro ao exportar entrega final: {e}")


# =====================================================================
#  PIPELINE PRINCIPAL
# =====================================================================

def processar_lote(
    df_lote: pd.DataFrame,
    lote: int,
    paths: Gate3Paths,
    preproc: ColumnTransformer,
    label_enc,
    modelo: Gate3Ensemble,
    col_id: str,
    cfg: Config
) -> pd.DataFrame:
    """
    Processa um lote de registros:
    1. Preprocessamento
    2. Predicao
    3. Validacao
    4. Exportacao
    """
    UI.info(f"Processando lote {lote} ({len(df_lote)} registros)...")
    
    # Preprocessamento
    try:
        X = preproc.transform(df_lote)
        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    except Exception as e:
        UI.erro(f"Erro no preprocessamento do lote {lote}: {e}")
        return pd.DataFrame()
    
    # Predicao
    try:
        y_proba = modelo.predict_proba(X_dense)
        top_idx = np.argsort(y_proba, axis=1)[:, ::-1][:, :cfg.top_k]
        
        # Criar DataFrame de resultados
        resultados = df_lote.copy()
        
        # Adicionar predicoes
        resultados["CNPJ_Previsto"] = label_enc.inverse_transform(y_proba.argmax(axis=1))
        resultados["CNPJ_Confianca"] = np.round(y_proba.max(axis=1), 4)
        
        # Adicionar Top-K
        for r in range(cfg.top_k):
            resultados[f"CNPJ_Top{r+1}"] = label_enc.inverse_transform(top_idx[:, r])
            resultados[f"CNPJ_Top{r+1}_Prob"] = np.round(
                np.take_along_axis(y_proba, top_idx[:, r:r+1], axis=1).ravel(), 4
            )
        
        # Adicionar metadados
        resultados["Lote"] = lote
        resultados["Timestamp"] = datetime.now().isoformat()
        resultados["Modelo"] = "Gate3Ensemble"
        
    except Exception as e:
        UI.erro(f"Erro na predicao do lote {lote}: {e}")
        return pd.DataFrame()
    
    # Validacao
    if cfg.validar:
        resultados = validar_candidatos(resultados)
    
    # Exportar palpite do lote
    if not cfg.dry_run:
        palpite_path = paths.get_palpite_path(lote)
        exportar_palpite_xlsx(resultados, palpite_path, lote)
    
    # Treinamento incremental (opcional)
    if not cfg.dry_run and hasattr(modelo, "partial_fit"):
        try:
            # Preparar dados para treinamento
            if col_id in df_lote.columns:
                y = label_enc.transform(df_lote[col_id].astype(str).values)
                modelo.partial_fit(X_dense, y, classes=modelo.classes_)
                UI.info(f"Modelo atualizado com lote {lote}")
                
                # Salvar modelo periodicamente
                if lote % 10 == 0:
                    joblib.dump({
                        "modelo": modelo,
                        "preprocessador": preproc,
                        "label_encoder": label_enc,
                    }, paths.modelo_file, compress=3)
                    UI.info(f"Modelo salvo: {paths.modelo_file}")
        except Exception as e:
            UI.warn(f"Erro no treinamento incremental do lote {lote}: {e}")
    
    return resultados


def pipeline_gate3(cfg: Config):
    """
    Pipeline principal para processamento Gate 3.
    """
    UI.banner(f"{APP_NOME} - {APP_SUB} - Base: {cfg.base}")
    
    # Inicializar caminhos
    paths = Gate3Paths(cfg.base, cfg.pasta_gate3)
    
    # Carregar progresso existente
    progresso = Progresso(base=cfg.base)
    if cfg.retomar and os.path.exists(paths.progresso_file):
        progresso = Progresso.carregar(paths.progresso_file)
        if progresso:
            UI.info(f"Retomando do lote {progresso.ultimo_lote} (registro {progresso.ultimo_registro})")
        else:
            progresso = Progresso(base=cfg.base)
    
    # Contar registros totais
    if progresso.total_registros == 0:
        progresso.total_registros = contar_registros_total(paths.entrada)
        UI.info(f"Total de registros: {progresso.total_registros:,}")
    
    # Iniciar processamento
    progresso.status = "em_execucao"
    progresso.inicio = datetime.now().isoformat()
    progresso.salvar(paths.progresso_file)
    
    # Variaveis para acumulacao de resultados
    todos_resultados = []
    lote_atual = 0
    registros_processados = 0
    
    # Carregar ou criar modelo
    if os.path.exists(paths.modelo_file) and not cfg.dry_run:
        try:
            dados_modelo = joblib.load(paths.modelo_file)
            modelo = dados_modelo["modelo"]
            preproc = dados_modelo["preprocessador"]
            label_enc = dados_modelo["label_encoder"]
            UI.info(f"Modelo carregado: {paths.modelo_file}")
        except Exception as e:
            UI.warn(f"Erro ao carregar modelo: {e}. Criando novo modelo.")
            modelo = None
            preproc = None
            label_enc = None
    else:
        modelo = None
        preproc = None
        label_enc = None
    
    # Iterar sobre os arquivos de entrada
    arquivo_count = 0
    for arq_nome, df_chunk in iterar_base_streaming(paths.entrada, cfg.chunk_size):
        arquivo_count += 1
        UI.info(f"Processando arquivo {arquivo_count}: {arq_nome}")
        
        # Processar em lotes menores
        for i in range(0, len(df_chunk), cfg.batch_size):
            df_lote = df_chunk.iloc[i:i + cfg.batch_size].copy()
            
            # Verificar se ja processamos este lote
            if registros_processados + len(df_lote) <= progresso.ultimo_registro:
                registros_processados += len(df_lote)
                continue
            
            # Detectar coluna ID
            col_id = detectar_coluna_id(df_lote)
            
            # Normalizar CNPJ se for o caso
            if "CNPJ" in col_id:
                df_lote[col_id] = normalizar_id_cnpj(df_lote[col_id])
            
            # Calcular KPIs
            df_lote = calcular_kpis_negocio(df_lote)
            df_lote = calcular_risco(df_lote)
            
            # Identificar colunas para preprocessamento
            col_texto = [c for c in df_lote.columns 
                        if df_lote[c].dtype == object and c != col_id]
            col_cat = [c for c in df_lote.columns 
                      if df_lote[c].dtype == object and c != col_id 
                      and df_lote[c].nunique() < 50]
            col_num = [c for c in df_lote.columns 
                      if pd.api.types.is_numeric_dtype(df_lote[c])]
            
            # Inicializar preprocessador se nao existir
            if preproc is None:
                preproc = construir_preprocessador(df_lote, col_texto, col_cat, col_num)
                UI.info("Preprocessador criado")
            
            # Inicializar label encoder se nao existir
            if label_enc is None:
                label_enc = None  # Sera criado no primeiro lote
            
            # Inicializar modelo se nao existir
            if modelo is None:
                modelo = Gate3Ensemble(alpha=1e-4, random_state=cfg.random_state)
                UI.info("Modelo criado")
            
            # Criar label encoder no primeiro lote
            if label_enc is None and col_id in df_lote.columns:
                from sklearn.preprocessing import LabelEncoder
                label_enc = LabelEncoder()
                label_enc.fit(df_lote[col_id].astype(str).unique())
                UI.info(f"LabelEncoder criado com {len(label_enc.classes_)} classes")
            
            # Processar lote
            resultados = processar_lote(
                df_lote, lote_atual, paths, preproc, label_enc, modelo, col_id, cfg
            )
            
            if not resultados.empty:
                todos_resultados.append(resultados)
            
            # Atualizar progresso
            registros_processados += len(df_lote)
            progresso.ultimo_lote = lote_atual
            progresso.ultimo_registro = registros_processados
            progresso.salvar(paths.progresso_file)
            
            lote_atual += 1
            
            # Mensagem de progresso
            pct = (registros_processados / progresso.total_registros * 100) if progresso.total_registros > 0 else 0
            UI.info(f"Lote {lote_atual - 1} | Processados: {registros_processados:,} | {pct:.1f}%")
    
    # Finalizar processamento
    progresso.status = "concluido"
    progresso.fim = datetime.now().isoformat()
    progresso.salvar(paths.progresso_file)
    
    UI.ok(f"Processamento concluido! Lotes processados: {lote_atual}")
    
    # Consolidar resultados
    if todos_resultados:
        df_final = pd.concat(todos_resultados, ignore_index=True)
        UI.info(f"Total de candidatos gerados: {len(df_final):,}")
        
        # Selecionar melhores candidatos
        df_entrega = seleccionar_melhores_candidatos(df_final, cfg.max_palpites)
        UI.info(f"Candidatos validados para entrega: {len(df_entrega)}")
        
        # Exportar entrega final
        if not cfg.dry_run:
            exportar_entrega_final(df_entrega, paths.entrega_file)
        
        # Salvar modelo final
        if not cfg.dry_run and modelo is not None:
            joblib.dump({
                "modelo": modelo,
                "preprocessador": preproc,
                "label_encoder": label_enc,
                "config": cfg.to_dict() if hasattr(cfg, "to_dict") else asdict(cfg),
            }, paths.modelo_file, compress=3)
            UI.ok(f"Modelo final salvo: {paths.modelo_file}")
    
    return df_final if todos_resultados else pd.DataFrame()


def main():
    """Função principal."""
    cfg = parse_args()
    
    if cfg.dry_run:
        UI.warn("Modo dry-run ativo - nao serao salvos arquivos")
    
    try:
        pipeline_gate3(cfg)
    except Exception as e:
        UI.erro(f"Erro no pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
