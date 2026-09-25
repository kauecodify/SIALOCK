# =====================================================================
#  RAPIDEZ.PY - Otimizador de Processamento para Extract V.3.3
#  =====================================================================
"""
Modulo de otimizacao para processamento rapido em qualquer dispositivo.
Inclui:
- Processamento streaming/chunked
- Parallel processing (quando possivel)
- Otimizacao de memoria
- Cache inteligente
- Ajuste automatico de batch size

Uso:
    from rapidez import (
        processar_em_lotes_rapido,
        ajustar_batch_size,
        otimizar_processamento,
        limpar_memoria
    )
"""

from __future__ import annotations

import os
import sys
import time
import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Optional, Callable, Any, List, Dict, Tuple
from functools import partial

import numpy as np
import pandas as pd

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import joblib
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# =====================================================================
#  CONFIGURACOES DE DESEMPENHO
# =====================================================================

class ConfigRapidez:
    """Configuracoes de otimizacao de desempenho."""
    
    # Processamento paralelo
    MAX_WORKERS = mp.cpu_count() if mp.cpu_count() > 1 else 1
    USE_MULTIPROCESSING = True
    USE_THREADING = True
    
    # Tamanhos de lote
    MIN_BATCH_SIZE = 100
    MAX_BATCH_SIZE = 10000
    DEFAULT_BATCH_SIZE = 1000
    
    # Limites de memoria
    MEMORY_LIMIT_GB = 4.0  # Limite de memoria em GB
    MEMORY_WARNING_GB = 3.0  # Aviso de memoria em GB
    
    # Cache
    CACHE_ENABLED = True
    CACHE_MAX_SIZE = 1000  # Maximo de itens no cache
    
    @classmethod
    def ajustar_para_memoria(cls, memoria_disponivel_gb: Optional[float] = None):
        """Ajusta configuracoes com base na memoria disponivel."""
        if not HAS_PSUTIL and memoria_disponivel_gb is None:
            return
        
        if memoria_disponivel_gb is None:
            memoria_disponivel_gb = psutil.virtual_memory().available / (1024 ** 3)
        
        # Ajustar batch size com base na memoria
        if memoria_disponivel_gb < 2:
            cls.MAX_BATCH_SIZE = 500
            cls.DEFAULT_BATCH_SIZE = 250
            cls.USE_MULTIPROCESSING = False
        elif memoria_disponivel_gb < 4:
            cls.MAX_BATCH_SIZE = 2000
            cls.DEFAULT_BATCH_SIZE = 500
        elif memoria_disponivel_gb < 8:
            cls.MAX_BATCH_SIZE = 5000
            cls.DEFAULT_BATCH_SIZE = 1000
        else:
            cls.MAX_BATCH_SIZE = 10000
            cls.DEFAULT_BATCH_SIZE = 2000
        
        # Ajustar workers
        if memoria_disponivel_gb < 2:
            cls.MAX_WORKERS = 1
            cls.USE_MULTIPROCESSING = False
        elif memoria_disponivel_gb < 4:
            cls.MAX_WORKERS = min(2, mp.cpu_count())
        else:
            cls.MAX_WORKERS = min(mp.cpu_count(), 4)


# Ajustar configuracoes no inicio
ConfigRapidez.ajustar_para_memoria()


# =====================================================================
#  FUNCOES DE MEMORIA
# =====================================================================

def get_memoria_uso() -> Dict[str, float]:
    """Retorna informacoes de uso de memoria em GB."""
    if not HAS_PSUTIL:
        return {"total": 0, "used": 0, "available": 0, "percent": 0}
    
    mem = psutil.virtual_memory()
    return {
        "total": mem.total / (1024 ** 3),
        "used": mem.used / (1024 ** 3),
        "available": mem.available / (1024 ** 3),
        "percent": mem.percent,
    }


def limpar_memoria():
    """Libera memoria nao utilizada."""
    gc.collect()
    
    # Limpar cache do pandas
    try:
        pd.DataFrame._clear_item_cache()
    except Exception:
        pass
    
    # Limpar cache do numpy
    try:
        np._clear_cache()
    except Exception:
        pass


def verificar_memoria(limite_gb: float = ConfigRapidez.MEMORY_LIMIT_GB) -> bool:
    """Verifica se ha memoria suficiente disponivel."""
    if not HAS_PSUTIL:
        return True
    
    mem = get_memoria_uso()
    if mem["available"] < limite_gb:
        return False
    return True


# =====================================================================
#  CACHE INTELIGENTE
# =====================================================================

class CacheProcessamento:
    """Cache para resultados de processamento."""
    
    def __init__(self, max_size: int = ConfigRapidez.CACHE_MAX_SIZE):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Busca item no cache."""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Adiciona item ao cache."""
        if len(self.cache) >= self.max_size:
            # Remover item mais antigo (FIFO)
            if self.cache:
                self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    
    def clear(self):
        """Limpa o cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> Dict[str, int]:
        """Retorna estatisticas do cache."""
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses + 1),
        }


# Cache global
_cache_global = CacheProcessamento()


def get_cache() -> CacheProcessamento:
    """Retorna o cache global."""
    return _cache_global


# =====================================================================
#  AJUSTE DE BATCH SIZE
# =====================================================================

def ajustar_batch_size(
    batch_size_atual: int,
    tempo_medio_por_lote: float,
    memoria_por_lote_gb: float,
    alvo_tempo_segundos: float = 1.0
) -> int:
    """
    Ajusta dinamicamente o batch size com base no tempo e memoria.
    
    Args:
        batch_size_atual: Batch size atual
        tempo_medio_por_lote: Tempo medio de processamento por lote (segundos)
        memoria_por_lote_gb: Memoria consumida por lote (GB)
        alvo_tempo_segundos: Tempo alvo por lote (segundos)
    
    Returns:
        Novo batch size otimizado
    """
    # Limites
    min_bs = ConfigRapidez.MIN_BATCH_SIZE
    max_bs = ConfigRapidez.MAX_BATCH_SIZE
    
    # Ajustar com base no tempo
    if tempo_medio_por_lote > 0:
        # Se esta muito devagar, diminuir batch size
        if tempo_medio_por_lote > alvo_tempo_segundos * 2:
            novo_bs = int(batch_size_atual * 0.7)
        # Se esta rapido, aumentar batch size
        elif tempo_medio_por_lote < alvo_tempo_segundos * 0.5:
            novo_bs = int(batch_size_atual * 1.3)
        else:
            novo_bs = batch_size_atual
    else:
        novo_bs = batch_size_atual
    
    # Ajustar com base na memoria
    if HAS_PSUTIL:
        mem = get_memoria_uso()
        if memoria_por_lote_gb > 0 and mem["available"] < ConfigRapidez.MEMORY_LIMIT_GB:
            # Reduzir batch size se memoria esta baixa
            novo_bs = int(novo_bs * 0.5)
    
    # Aplicar limites
    novo_bs = max(min_bs, min(max_bs, novo_bs))
    
    return novo_bs


# =====================================================================
#  PROCESSAMENTO PARALELO
# =====================================================================

def processar_paralelo(
    func: Callable,
    dados: List[Any],
    n_workers: Optional[int] = None,
    use_multiprocessing: bool = True,
    use_threading: bool = True,
    batch_size: Optional[int] = None,
    **kwargs
) -> List[Any]:
    """
    Processa dados em paralelo usando multiprocessing ou threading.
    
    Args:
        func: Funcao a ser aplicada a cada item
        dados: Lista de dados a processar
        n_workers: Numero de workers (default: ConfigRapidez.MAX_WORKERS)
        use_multiprocessing: Usar multiprocessing
        use_threading: Usar threading
        batch_size: Tamanho do lote para joblib
        **kwargs: Argumentos adicionais para a funcao
    
    Returns:
        Lista de resultados
    """
    if not dados:
        return []
    
    n_workers = n_workers or ConfigRapidez.MAX_WORKERS
    
    # Verificar memoria
    if not verificar_memoria():
        n_workers = 1
        use_multiprocessing = False
        use_threading = False
    
    # Usar joblib se disponivel
    if HAS_JOBLIB and use_multiprocessing and n_workers > 1:
        try:
            if batch_size is None:
                batch_size = max(1, len(dados) // (n_workers * 4))
            
            resultados = Parallel(n_jobs=n_workers, batch_size=batch_size)(
                delayed(func)(d, **kwargs) for d in dados
            )
            return resultados
        except Exception:
            pass
    
    # Usar ProcessPoolExecutor
    if use_multiprocessing and n_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                resultados = list(executor.map(partial(func, **kwargs), dados))
            return resultados
        except Exception:
            pass
    
    # Usar ThreadPoolExecutor
    if use_threading and n_workers > 1:
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                resultados = list(executor.map(partial(func, **kwargs), dados))
            return resultados
        except Exception:
            pass
    
    # Processamento sequencial
    return [func(d, **kwargs) for d in dados]


# =====================================================================
#  PROCESSAMENTO EM LOTES RAPIDO
# =====================================================================

def processar_em_lotes_rapido(
    dados: pd.DataFrame,
    func: Callable,
    batch_size: int = ConfigRapidez.DEFAULT_BATCH_SIZE,
    n_workers: Optional[int] = None,
    use_multiprocessing: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs
) -> List[Any]:
    """
    Processa um DataFrame em lotes de forma otimizada.
    
    Args:
        dados: DataFrame a processar
        func: Funcao a ser aplicada a cada lote (recebe DataFrame)
        batch_size: Tamanho do lote
        n_workers: Numero de workers para processamento paralelo
        use_multiprocessing: Usar multiprocessing
        progress_callback: Funcao de callback para progresso (atual, total)
        **kwargs: Argumentos adicionais para a funcao
    
    Returns:
        Lista de resultados
    """
    if dados.empty:
        return []
    
    # Ajustar batch size
    batch_size = ajustar_batch_size(
        batch_size,
        tempo_medio_por_lote=0,  # Seria calculado dinamicamente
        memoria_por_lote_gb=0
    )
    
    # Dividir em lotes
    lotes = [dados.iloc[i:i + batch_size] for i in range(0, len(dados), batch_size)]
    total_lotes = len(lotes)
    
    # Processar lotes
    resultados = []
    for i, lote in enumerate(lotes):
        resultado = func(lote, **kwargs)
        resultados.append(resultado)
        
        # Callback de progresso
        if progress_callback:
            progress_callback(i + 1, total_lotes)
        
        # Limpar memoria periodicamente
        if (i + 1) % 10 == 0:
            limpar_memoria()
    
    return resultados


def processar_em_lotes_paralelo(
    dados: pd.DataFrame,
    func: Callable,
    batch_size: int = ConfigRapidez.DEFAULT_BATCH_SIZE,
    n_workers: Optional[int] = None,
    **kwargs
) -> List[Any]:
    """
    Processa um DataFrame em lotes usando processamento paralelo.
    
    Args:
        dados: DataFrame a processar
        func: Funcao a ser aplicada a cada lote
        batch_size: Tamanho do lote
        n_workers: Numero de workers
        **kwargs: Argumentos adicionais
    
    Returns:
        Lista de resultados
    """
    if dados.empty:
        return []
    
    # Dividir em lotes
    lotes = [dados.iloc[i:i + batch_size] for i in range(0, len(dados), batch_size)]
    
    # Processar em paralelo
    return processar_paralelo(
        func,
        lotes,
        n_workers=n_workers,
        use_multiprocessing=True,
        **kwargs
    )


# =====================================================================
#  OTIMIZACAO DE DATAFRAME
# =====================================================================

def otimizar_dataframe(df: pd.DataFrame, downcast: bool = True) -> pd.DataFrame:
    """
    Otimiza o uso de memoria de um DataFrame.
    
    Args:
        df: DataFrame a otimizar
        downcast: Downcast tipos numericos
    
    Returns:
        DataFrame otimizado
    """
    df = df.copy()
    
    # Converter object para category quando possivel
    for col in df.select_dtypes(include=['object']):
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    
    # Downcast tipos numericos
    if downcast:
        for col in df.select_dtypes(include=['int64']):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']):
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


def reduzir_memoria_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduz o uso de memoria de um DataFrame de forma agressiva.
    """
    df = df.copy()
    
    # Converter todas as colunas object para category
    for col in df.select_dtypes(include=['object']):
        try:
            df[col] = df[col].astype('category')
        except Exception:
            pass
    
    # Downcast todos os numericos
    for col in df.select_dtypes(include=['int']):
        try:
            df[col] = pd.to_numeric(df[col], downcast='unsigned')
        except Exception:
            try:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            except Exception:
                pass
    
    for col in df.select_dtypes(include=['float']):
        try:
            df[col] = pd.to_numeric(df[col], downcast='float')
        except Exception:
            pass
    
    return df


# =====================================================================
#  FUNCOES DE TEMPO
# =====================================================================

class Timer:
    """Timer para medir tempo de execucao."""
    
    def __init__(self, nome: str = ""):
        self.nome = nome
        self.inicio = None
        self.fim = None
        self.tempos = []
    
    def start(self):
        """Inicia o timer."""
        self.inicio = time.time()
        return self
    
    def stop(self):
        """Para o timer e retorna o tempo decorrido."""
        self.fim = time.time()
        elapsed = self.fim - self.inicio if self.inicio else 0
        self.tempos.append(elapsed)
        return elapsed
    
    def lap(self) -> float:
        """Retorna o tempo desde o inicio (sem parar)."""
        return time.time() - self.inicio if self.inicio else 0
    
    def media(self) -> float:
        """Retorna a media dos tempos registrados."""
        return sum(self.tempos) / len(self.tempos) if self.tempos else 0
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()


def medir_tempo(func: Callable) -> Callable:
    """Decorator para medir tempo de execucao de uma funcao."""
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fim = time.time()
        print(f"[TIMER] {func.__name__}: {fim - inicio:.4f}s")
        return resultado
    return wrapper


# =====================================================================
#  OTIMIZACAO DE PROCESSAMENTO
# =====================================================================

def otimizar_processamento(
    func: Callable,
    dados: Any,
    n_iteracoes: int = 1,
    warmup: int = 1,
    **kwargs
) -> Tuple[Any, Dict[str, float]]:
    """
    Otimiza e mede o desempenho de uma funcao.
    
    Args:
        func: Funcao a otimizar
        dados: Dados de entrada
        n_iteracoes: Numero de iteracoes para medicao
        warmup: Numero de iteracoes de warmup
        **kwargs: Argumentos para a funcao
    
    Returns:
        Tupla: (resultado, metricas)
    """
    # Warmup
    for _ in range(warmup):
        func(dados, **kwargs)
    
    # Medicoes
    tempos = []
    resultados = []
    
    for _ in range(n_iteracoes):
        inicio = time.time()
        resultado = func(dados, **kwargs)
        fim = time.time()
        tempos.append(fim - inicio)
        resultados.append(resultado)
    
    # Retornar o ultimo resultado
    metricas = {
        "tempo_medio": sum(tempos) / len(tempos),
        "tempo_min": min(tempos),
        "tempo_max": max(tempos),
        "desvio_padrao": np.std(tempos),
    }
    
    return resultado, metricas


# =====================================================================
#  FUNCOES DE SISTEMA
# =====================================================================

def get_info_sistema() -> Dict[str, Any]:
    """Retorna informacoes do sistema."""
    info = {
        "cpu_count": mp.cpu_count(),
        "python_version": sys.version,
        "platform": sys.platform,
    }
    
    if HAS_PSUTIL:
        mem = get_memoria_uso()
        info.update({
            "memoria_total_gb": mem["total"],
            "memoria_usada_gb": mem["used"],
            "memoria_disponivel_gb": mem["available"],
            "memoria_percent": mem["percent"],
        })
    
    try:
        import pandas as pd
        info["pandas_version"] = pd.__version__
    except Exception:
        pass
    
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except Exception:
        pass
    
    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except Exception:
        pass
    
    return info


def imprimir_info_sistema():
    """Imprime informacoes do sistema."""
    info = get_info_sistema()
    
    print("=" * 60)
    print("INFORMACOES DO SISTEMA")
    print("=" * 60)
    
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60)


# =====================================================================
#  FUNCOES DE COMPATIBILIDADE
# =====================================================================

def is_notebook() -> bool:
    """Verifica se esta sendo executado em um notebook."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def is_low_memory() -> bool:
    """Verifica se o sistema tem pouca memoria."""
    if not HAS_PSUTIL:
        return False
    
    mem = get_memoria_uso()
    return mem["available"] < ConfigRapidez.MEMORY_LIMIT_GB


def is_low_cpu() -> bool:
    """Verifica se o sistema tem poucos núcleos de CPU."""
    return mp.cpu_count() <= 2


# =====================================================================
#  EXEMPLO DE USO
# =====================================================================

def exemplo_uso():
    """Exemplo de como usar as funcoes de rapidez."""
    
    # Imprimir informacoes do sistema
    imprimir_info_sistema()
    
    # Criar um DataFrame de exemplo
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    df = pd.DataFrame({
        "id": range(10000),
        "valor": np.random.randn(10000),
        "categoria": np.random.choice(["A", "B", "C", "D"], 10000),
        "texto": [f"texto_{i}" for i in range(10000)],
    })
    
    # Otimizar DataFrame
    df_otimizado = otimizar_dataframe(df)
    print(f"Memoria original: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print(f"Memoria otimizada: {df_otimizado.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    
    # Processar em lotes
    def processar_lote(lote):
        # Simular processamento
        time.sleep(0.01)
        return lote.sum()
    
    # Processamento sequencial
    with Timer("Processamento sequencial"):
        resultados_seq = processar_em_lotes_rapido(df, processar_lote, batch_size=500)
    
    # Processamento paralelo
    with Timer("Processamento paralelo"):
        resultados_par = processar_em_lotes_paralelo(df, processar_lote, batch_size=500)
    
    print(f"Resultados sequencial: {len(resultados_seq)}")
    print(f"Resultados paralelo: {len(resultados_par)}")


if __name__ == "__main__":
    exemplo_uso()
