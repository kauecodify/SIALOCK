import os
from pathlib import Path
import pandas as pd

# Define dinamicamente o caminho até a pasta Downloads/dados
PASTA_DADOS = Path.home() / "Downloads" / "dados"
NOME_DO_ARQUIVO = "Base 1_Gate 1v02.parquet"  

ARQUIVO_BASE = PASTA_DADOS / NOME_DO_ARQUIVO

def carregar_base():
    """Localiza e carrega o arquivo parquet/excel da pasta de dados."""
    if not PASTA_DADOS.exists():
        print(f"❌ Pasta {PASTA_DADOS} não encontrada.")
        return None

    if not ARQUIVO_BASE.exists():
        print(f"⚠️ Arquivo '{NOME_DO_ARQUIVO}' não encontrado em: {PASTA_DADOS}")
        return None

    print(f"📂 Carregando base: {ARQUIVO_BASE.name}...")

    if ARQUIVO_BASE.suffix == '.parquet':
        return pd.read_parquet(ARQUIVO_BASE, engine='pyarrow')
    else:
        return pd.read_excel(ARQUIVO_BASE)

def analisar_potencial_reidentificacao(df):
    """Realiza o profiling de vulnerabilidades para reidentificação (Red Team)."""
    print(f"\n📊 Total de registros na base: {len(df):,}")
    print("=" * 60)

    # 1. Maiores operações por Valor em Dólar (Outliers Financeiros)
    print("\n💰 TOP 5 MAIORES OPERAÇÕES (Maior rastreabilidade por valor):")
    
    # Garantir ordenação numérica do valor
    df['vmle_dolar_num'] = pd.to_numeric(df['vmle_dolar'], errors='coerce')
    top_valores = df.nlargest(5, 'vmle_dolar_num')
    
    for idx, row in top_valores.iterrows():
        print(f"- NCM: {row['cod_ncm']} | Valor: US$ {row['vmle_dolar_num']:,.2f} | Origem: {row['pais_de_origem']}")
        print(f"  Descrição: {str(row['descricao_do_produto'])[:100]}...\n")

    # 2. NCMs mais Raros/Específicos (Menor concorrência de importadores)
    print("=" * 60)
    print("\n🎯 NCMS RAROS (Poucas ocorrências na base = universo menor de CNPJs):")
    contagem_ncm = df['cod_ncm'].value_counts()
    ncms_unicos = contagem_ncm[contagem_ncm == 1].head(5)
    
    for ncm in ncms_unicos.index:
        reg = df[df['cod_ncm'] == ncm].iloc[0]
        print(f"- NCM Único: {ncm} | Valor: US$ {pd.to_numeric(reg['vmle_dolar'], errors='coerce'):,.2f}")
        print(f"  Descrição: {str(reg['descricao_do_produto'])[:100]}...\n")

    # 3. Descrições mais longas/detalhadas (Possíveis falhas da IA de anonimização)
    print("=" * 60)
    print("\n🔍 DESCRIÇÕES COM MAIOR DETALHAMENTO TEXTUAL:")
    df['tamanho_desc'] = df['descricao_do_produto'].astype(str).str.len()
    top_descricoes = df.nlargest(3, 'tamanho_desc')

    for idx, row in top_descricoes.iterrows():
        print(f"- NCM: {row['cod_ncm']} | Tamanho do texto: {row['tamanho_desc']} caracteres")
        print(f"  Texto: {row['descricao_do_produto']}\n")

if __name__ == "__main__":
    df = carregar_base()
    if df is not None:
        analisar_potencial_reidentificacao(df)