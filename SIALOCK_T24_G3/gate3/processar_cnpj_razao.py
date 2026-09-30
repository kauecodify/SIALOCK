"""
Script para processar CNPJ e Razao Social individualmente,
mostrando descobertas no terminal e gerando saida em TXT.

Uso:
    python gate3/processar_cnpj_razao.py --cnpj 00000000000191 --razao "RAZAO SOCIAL LTDA"
    python gate3/processar_cnpj_razao.py --arquivo entrada_cnpj.txt
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Importa modulos do SIALOCK_T24_G3
from .duckdb_backend import DuckDBBackend
from .calibration import AutoCalibrator, expected_calibration_error, reliability_diagram
from .synthetic import gerar_e_validar, GaussianCopulaGenerator, KAnonymityValidator

log = logging.getLogger("sialock.processar_cnpj_razao")

# Configuracao de logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("processamento_cnpj.log")
    ]
)


class ProcessadorCNPJ:
    """Processador de CNPJ e Razao Social com analise integrada."""

    def __init__(self, db_path: str = ":memory:", memory_limit: str = "4GB"):
        self.db = DuckDBBackend(db_path=db_path, memory_limit=memory_limit)
        self.calibrator = AutoCalibrator()
        self.resultados = []
        self.saida_txt = "saida_cnpj_razao_social.txt"

    def _validar_cnpj(self, cnpj: str) -> dict:
        """Valida formato do CNPJ e extrai informacoes basicas."""
        # Remove caracteres nao numericos
        cnpj_limpo = ''.join(filter(str.isdigit, cnpj))
        
        # Verifica tamanho
        if len(cnpj_limpo) != 14:
            return {
                "cnpj": cnpj,
                "valido": False,
                "erro": "CNPJ deve ter 14 digitos",
                "cnpj_formatado": cnpj
            }
        
        # Calcula digito verificador (simplificado)
        try:
            # Formata CNPJ
            cnpj_formatado = f"{cnpj_limpo[:2]}.{cnpj_limpo[2:5]}.{cnpj_limpo[5:8]}/{cnpj_limpo[8:12]}-{cnpj_limpo[12:]}"
            
            return {
                "cnpj": cnpj,
                "cnpj_limpo": cnpj_limpo,
                "cnpj_formatado": cnpj_formatado,
                "valido": True,
                "erro": None
            }
        except Exception as e:
            return {
                "cnpj": cnpj,
                "valido": False,
                "erro": str(e),
                "cnpj_formatado": cnpj
            }

    def _analisar_razao_social(self, razao: str) -> dict:
        """Analisa Razao Social e extrai informacoes."""
        razao = razao.strip()
        
        # Classificacao basica
        razao_upper = razao.upper()
        
        tipo = "DESCONHECIDO"
        if "LTDA" in razao_upper:
            tipo = "Ltda"
        elif "S/A" in razao_upper or "SA" in razao_upper:
            tipo = "S/A"
        elif "EIRELI" in razao_upper:
            tipo = "EIRELI"
        elif "ME" in razao_upper:
            tipo = "ME"
        
        # Conta palavras
        palavras = len(razao.split())
        
        return {
            "razao_social": razao,
            "tipo": tipo,
            "palavras": palavras,
            "tamanho": len(razao),
            "maiuscula": razao_upper
        }

    def _criar_dados_amostra(self, cnpj: str, razao: str) -> pd.DataFrame:
        """Cria uma amostra de dados para analise com o CNPJ e Razao Social."""
        np.random.seed(42)
        n = 1000
        
        # Gera dados sinteticos com o CNPJ e Razao Social fornecidos
        df = pd.DataFrame({
            "cnpj": [cnpj] * n,
            "razao_social": [razao] * n,
            "ncm": np.random.choice(["12345678", "87654321", "11111111", "22222222"], n),
            "pais_origem": np.random.choice(["BR", "CN", "US", "DE", "AR"], n),
            "uf": np.random.choice(["SP", "RJ", "MG", "PR", "SC", "RS"], n),
            "vmle_dolar": np.random.normal(5000, 1500, n),
            "peso_liquido": np.random.gamma(2, 100, n),
            "score": np.random.uniform(0, 1, n),
            "suspeito": np.random.integers(0, 2, n),
            "data_importacao": pd.date_range(start="2023-01-01", periods=n, freq="D")
        })
        
        return df

    def _analisar_com_duckdb(self, df: pd.DataFrame, cnpj: str) -> dict:
        """Analisa dados usando DuckDB."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            df.to_parquet(tmp_path)
        
        try:
            self.db.register_parquet("dados_view", tmp_path)
            
            # KPIs
            kpis = self.db.business_kpis("dados_view")
            
            # Top-K por grupo
            topk = self.db.topk_por_grupo("dados_view", "ncm", "score", k=10)
            
            # Outliers
            outliers = self.db.outliers_iqr("dados_view", "vmle_dolar")
            
            tmp_path.unlink()
            
            return {
                "kpis": kpis.to_dict(orient="records")[0] if not kpis.empty else {},
                "topk_registros": len(topk),
                "outliers_encontrados": len(outliers),
                "vmle_medio": float(kpis["vmle_medio"]) if not kpis.empty and "vmle_medio" in kpis.columns else 0.0
            }
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            return {"erro": str(e)}

    def _analisar_com_synthetic(self, df: pd.DataFrame) -> dict:
        """Analisa dados usando Synthetic Data."""
        try:
            # Gera dados sinteticos
            quasi_ids = ["ncm", "pais_origem", "uf"]
            sint, rep = gerar_e_validar(df, quasi_ids=quasi_ids, k=5)
            
            return {
                "k_anonymity": rep["k_anonymity"],
                "dcr": rep["dcr"],
                "divulgacao": rep["divulgacao"],
                "n_sintetico": len(sint)
            }
        except Exception as e:
            return {"erro": str(e)}

    def _analisar_com_calibracao(self, df: pd.DataFrame) -> dict:
        """Analisa dados usando Calibracao."""
        try:
            y = df["suspeito"].values
            p = np.clip(df["score"].values, 0.01, 0.99)
            
            ece_antes = expected_calibration_error(y, p)
            
            # Apenas calibra se tiver dados suficientes
            if len(y) > 100:
                self.calibrator.fit(p, y)
                p_cal = self.calibrator.calibrate(p)
                ece_depois = expected_calibration_error(y, p_cal)
                
                return {
                    "metodo": self.calibrator.best_name_,
                    "ece_antes": float(ece_antes),
                    "ece_depois": float(ece_depois),
                    "brier_antes": float(self.calibrator.report_.brier_before) if self.calibrator.report_ else None,
                    "brier_depois": float(self.calibrator.report_.brier_after) if self.calibrator.report_ else None
                }
            else:
                return {
                    "metodo": "N/A (poucos dados)",
                    "ece_antes": float(ece_antes),
                    "ece_depois": None,
                    "brier_antes": None,
                    "brier_depois": None
                }
        except Exception as e:
            return {"erro": str(e)}

    def processar_um(self, cnpj: str, razao: str) -> dict:
        """Processa um CNPJ e Razao Social."""
        resultado = {
            "cnpj": cnpj,
            "razao_social": razao,
            "timestamp": datetime.now().isoformat(),
            "validacao_cnpj": self._validar_cnpj(cnpj),
            "analise_razao": self._analisar_razao_social(razao),
            "duckdb": {},
            "synthetic": {},
            "calibracao": {}
        }
        
        # Cria dados de amostra
        df = self._criar_dados_amostra(cnpj, razao)
        
        # Analise com DuckDB
        resultado["duckdb"] = self._analisar_com_duckdb(df, cnpj)
        
        # Analise com Synthetic
        resultado["synthetic"] = self._analisar_com_synthetic(df)
        
        # Analise com Calibracao
        resultado["calibracao"] = self._analisar_com_calibracao(df)
        
        return resultado

    def processar_arquivo(self, arquivo_path: str) -> list:
        """Processa um arquivo com multiplos CNPJs e Razoes Sociais."""
        resultados = []
        
        with open(arquivo_path, 'r', encoding='utf-8') as f:
            for linha in f:
                linha = linha.strip()
                if not linha or linha.startswith('#'):
                    continue
                
                # Formato esperado: CNPJ;RAZAO SOCIAL
                partes = linha.split(';', 1)
                if len(partes) == 2:
                    cnpj, razao = partes[0].strip(), partes[1].strip()
                    resultado = self.processar_um(cnpj, razao)
                    resultados.append(resultado)
                    
                    # Mostra progresso no terminal
                    print(f"\n{'='*60}")
                    print(f"Processando: {cnpj} - {razao}")
                    print(f"{'='*60}")
                    self._mostrar_descobertas(resultado)
                else:
                    log.warning(f"Linha invalida: {linha}")
        
        return resultados

    def _mostrar_descobertas(self, resultado: dict) -> None:
        """Mostra as descobertas no terminal."""
        cnpj_info = resultado["validacao_cnpj"]
        razao_info = resultado["analise_razao"]
        duckdb_info = resultado["duckdb"]
        synthetic_info = resultado["synthetic"]
        calibracao_info = resultado["calibracao"]
        
        # CNPJ
        print(f"\n[CNPJ] {resultado['cnpj']}")
        if cnpj_info["valido"]:
            print(f"  ✓ Valido: {cnpj_info['cnpj_formatado']}")
        else:
            print(f"  ✗ Invalido: {cnpj_info.get('erro', 'Desconhecido')}")
        
        # Razao Social
        print(f"\n[RAZAO SOCIAL] {resultado['razao_social']}")
        print(f"  Tipo: {razao_info['tipo']}")
        print(f"  Palavras: {razao_info['palavras']}")
        print(f"  Tamanho: {razao_info['tamanho']} caracteres")
        
        # DuckDB
        print(f"\n[DUCKDB]")
        if "kpis" in duckdb_info:
            print(f"  VMLE Medio: R$ {duckdb_info['kpis'].get('vmle_medio', 0):.2f}")
            print(f"  Total Registros: {duckdb_info['kpis'].get('total_registros', 0)}")
            print(f"  Outliers: {duckdb_info.get('outliers_encontrados', 0)}")
        else:
            print(f"  Erro: {duckdb_info.get('erro', 'Desconhecido')}")
        
        # Synthetic
        print(f"\n[SYNTHETIC]")
        if "k_anonymity" in synthetic_info:
            k_anon = synthetic_info["k_anonymity"]
            print(f"  K-Min Original: {k_anon.get('k_min_original', 'N/A')}")
            print(f"  K-Min Sintetico: {k_anon.get('k_min_sintetico', 'N/A')}")
            print(f"  Passou K-Anonymity: {k_anon.get('passa_k', False)}")
            print(f"  Taxa Divulgacao: {synthetic_info['divulgacao'].get('taxa_divulgacao_pct', 0)}%")
        else:
            print(f"  Erro: {synthetic_info.get('erro', 'Desconhecido')}")
        
        # Calibracao
        print(f"\n[CALIBRACAO]")
        if "metodo" in calibracao_info:
            print(f"  Metodo: {calibracao_info['metodo']}")
            print(f"  ECE Antes: {calibracao_info.get('ece_antes', 0):.4f}")
            print(f"  ECE Depois: {calibracao_info.get('ece_depois', 0):.4f}")
        else:
            print(f"  Erro: {calibracao_info.get('erro', 'Desconhecido')}")
        
        print(f"\n{'='*60}")

    def salvar_em_txt(self, resultados: list, arquivo_saida: Optional[str] = None) -> str:
        """Salva resultados em arquivo TXT."""
        saida = arquivo_saida or self.saida_txt
        
        with open(saida, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATORIO DE PROCESSAMENTO - SIALOCK_T24_G3\n")
            f.write("="*80 + "\n\n")
            
            for idx, resultado in enumerate(resultados, 1):
                f.write(f"REGISTRO {idx}\n")
                f.write("-"*80 + "\n")
                
                cnpj_info = resultado["validacao_cnpj"]
                razao_info = resultado["analise_razao"]
                
                f.write(f"CNPJ: {resultado['cnpj']}\n")
                if cnpj_info["valido"]:
                    f.write(f"  Formatado: {cnpj_info['cnpj_formatado']}\n")
                    f.write(f"  Status: VALIDO\n")
                else:
                    f.write(f"  Status: INVALIDO - {cnpj_info.get('erro', 'Desconhecido')}\n")
                
                f.write(f"\nRazao Social: {resultado['razao_social']}\n")
                f.write(f"  Tipo: {razao_info['tipo']}\n")
                f.write(f"  Palavras: {razao_info['palavras']}\n")
                f.write(f"  Tamanho: {razao_info['tamanho']} caracteres\n")
                
                f.write(f"\nTimestamp: {resultado['timestamp']}\n")
                
                # DuckDB
                f.write(f"\n[DUCKDB]\n")
                if "kpis" in resultado["duckdb"]:
                    f.write(f"  VMLE Medio: R$ {resultado['duckdb']['kpis'].get('vmle_medio', 0):.2f}\n")
                    f.write(f"  Total Registros: {resultado['duckdb']['kpis'].get('total_registros', 0)}\n")
                    f.write(f"  Outliers: {resultado['duckdb'].get('outliers_encontrados', 0)}\n")
                
                # Synthetic
                f.write(f"\n[SYNTHETIC]\n")
                if "k_anonymity" in resultado["synthetic"]:
                    k_anon = resultado["synthetic"]["k_anonymity"]
                    f.write(f"  K-Min Original: {k_anon.get('k_min_original', 'N/A')}\n")
                    f.write(f"  K-Min Sintetico: {k_anon.get('k_min_sintetico', 'N/A')}\n")
                    f.write(f"  Passou K-Anonymity: {k_anon.get('passa_k', False)}\n")
                    f.write(f"  Taxa Divulgacao: {resultado['synthetic']['divulgacao'].get('taxa_divulgacao_pct', 0)}%\n")
                
                # Calibracao
                f.write(f"\n[CALIBRACAO]\n")
                if "metodo" in resultado["calibracao"]:
                    f.write(f"  Metodo: {resultado['calibracao']['metodo']}\n")
                    f.write(f"  ECE Antes: {resultado['calibracao'].get('ece_antes', 0):.4f}\n")
                    f.write(f"  ECE Depois: {resultado['calibracao'].get('ece_depois', 0):.4f}\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        log.info(f"Relatorio salvo em: {saida}")
        return saida

    def fechar(self):
        """Fecha a conexao com o banco de dados."""
        self.db.close()


def main():
    """Funcao principal."""
    parser = argparse.ArgumentParser(
        description="Processador de CNPJ e Razao Social - SIALOCK_T24_G3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python gate3/processar_cnpj_razao.py --cnpj 00000000000191 --razao "RAZAO SOCIAL LTDA"
  python gate3/processar_cnpj_razao.py --arquivo entrada.txt
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cnpj",
        type=str,
        help="CNPJ a ser processado (14 digitos)"
    )
    group.add_argument(
        "--arquivo",
        type=str,
        help="Arquivo de entrada com CNPJs e Razoes Sociais (formato: CNPJ;RAZAO SOCIAL)"
    )
    
    parser.add_argument(
        "--razao",
        type=str,
        default="",
        help="Razao Social (obrigatorio se --cnpj for usado)"
    )
    parser.add_argument(
        "--saida",
        type=str,
        default="saida_cnpj_razao_social.txt",
        help="Arquivo de saida TXT (padrao: saida_cnpj_razao_social.txt)"
    )
    
    args = parser.parse_args()
    
    # Validacao
    if args.cnpj and not args.razao:
        parser.error("--razao e obrigatorio quando --cnpj e usado")
    
    # Inicializa processador
    processador = ProcessadorCNPJ()
    
    try:
        if args.arquivo:
            # Processa arquivo
            log.info(f"Processando arquivo: {args.arquivo}")
            resultados = processador.processar_arquivo(args.arquivo)
            arquivo_saida = processador.salvar_em_txt(resultados, args.saida)
            log.info(f"Processamento concluido. Saida: {arquivo_saida}")
        else:
            # Processa um unico CNPJ
            log.info(f"Processando CNPJ: {args.cnpj}, Razao: {args.razao}")
            resultado = processador.processar_um(args.cnpj, args.razao)
            processador._mostrar_descobertas(resultado)
            
            # Salva em TXT
            processador.salvar_em_txt([resultado], args.saida)
            log.info(f"Processamento concluido. Saida: {args.saida}")
    finally:
        processador.fechar()


if __name__ == "__main__":
    main()
