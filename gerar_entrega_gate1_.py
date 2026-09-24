import pandas as pd

# 1. INSIRA AQUI OS PALPITES ENCONTRADOS PELO SEU GRUPO (Até 50 registros)
# Subtitua os exemplos abaixo pelos dados reais trazidos pelos integrantes do grupo
palpites_grupo = [
    {"numero_de_ordem": 102, "CNPJ": "12345678000195"},
    {"numero_de_ordem": 450, "CNPJ": "98765432000101"},
    # ... adicione os demais até o limite de 50
]

# 2. Criar o DataFrame
df_entrega = pd.DataFrame(palpites_grupo)

if not df_entrega.empty:
    # Garantir que a coluna CNPJ seja string e possua exatamente 14 dígitos (com zeros à esquerda se necessário)
    df_entrega['CNPJ'] = df_entrega['CNPJ'].astype(str).str.replace(r'\D', '', regex=True).str.zfill(14)

    # Regras de validação do Regulamento
    # - Limite de 50 palpites
    df_entrega = df_entrega.head(50)
    
    # - Garantir colunas exatas exigidas no Regulamento (numero_de_ordem e CNPJ)
    df_entrega = df_entrega[['numero_de_ordem', 'CNPJ']]

    # 3. Exportar no formato Parquet exigido
    NOME_ARQUIVO_SAIDA = "entrega_gate1_palpites.parquet"
    df_entrega.to_parquet(NOME_ARQUIVO_SAIDA, engine='pyarrow', index=False)
    
    print(f"✅ Arquivo Parquet gerado com sucesso: {NOME_ARQUIVO_SAIDA}")
    print(f"📊 Total de palpites incluídos: {len(df_entrega)}")
    print("\nVisualização das primeiras linhas:")
    print(df_entrega.head())
else:
    print("⚠️ Nenhum palpite inserido na lista.")