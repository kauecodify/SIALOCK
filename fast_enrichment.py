import os
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurações alinhadas com o script do seu colega
PASTA_CACHE = "./cache_externo"
CACHE_FILE = os.path.join(PASTA_CACHE, "cnpj_cache.parquet")
API_BRASILAPI = "https://brasilapi.com.br/api/cnpj/v1/{cnpj}"
MAX_WORKERS = 8  # Número de requisições em paralelo


def consultar_cnpj_single(cnpj: str) -> dict:
    """Consulta um único CNPJ na BrasilAPI."""
    cnpj_limpo = "".join(filter(str.isdigit, str(cnpj))).zfill(14)
    url = API_BRASILAPI.format(cnpj=cnpj_limpo)

    try:
        response = requests.get(
            url, timeout=10, headers={"User-Agent": "FastEnrichment/1.0"}
        )
        if response.status_code == 200:
            data = response.json()
            return {
                "CNPJ": cnpj_limpo,
                "EXT_RazaoSocial": data.get("razao_social"),
                "EXT_NomeFantasia": data.get("nome_fantasia"),
                "EXT_CNAE_Codigo": str(data.get("cnae_fiscal") or ""),
                "EXT_CNAE_Descricao": data.get("cnae_fiscal_descricao"),
                "EXT_NaturezaJuridica": data.get("natureza_juridica"),
                "EXT_UF": data.get("uf"),
                "EXT_Municipio": data.get("municipio"),
                "EXT_Situacao": data.get("descricao_situacao_cadastral"),
                "EXT_Porte": data.get("porte"),
                "EXT_CapitalSocial": data.get("capital_social"),
                "EXT_DataAbertura": data.get("data_inicio_atividade"),
                "EXT_QtdSocios": len(data.get("qsa") or []),
                "EXT_QtdCNAESecundarios": len(
                    data.get("cnaes_secundarios") or []
                ),
            }
    except Exception:
        pass
    return None


def enriquecer_base_paralelo(cnpjs_list: list):
    """Processa a lista de CNPJs em paralelo e salva diretamente no cache."""
    os.makedirs(PASTA_CACHE, exist_ok=True)

    # Carrega cache existente
    cache_df = pd.DataFrame()
    if os.path.exists(CACHE_FILE):
        try:
            cache_df = pd.read_parquet(CACHE_FILE, engine="pyarrow")
            print(f"💾 Cache atual carregado: {len(cache_df)} CNPJs")
        except Exception:
            cache_df = pd.DataFrame()

    cnpjs_no_cache = (
        set(cache_df["CNPJ"].tolist()) if not cache_df.empty else set()
    )
    cnpjs_para_buscar = [c for c in cnpjs_list if c not in cnpjs_no_cache]

    if not cnpjs_para_buscar:
        print("✅ Todos os CNPJs já estão no cache!")
        return

    print(
        f"🚀 Iniciando consulta paralela para {len(cnpjs_para_buscar)} CNPJs ({MAX_WORKERS} threads)..."
    )

    novos_registros = []
    t0 = time.time()

    # Execução concorrente usando ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(consultar_cnpj_single, cnpj): cnpj
            for cnpj in cnpjs_para_buscar
        }

        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res:
                novos_registros.append(res)

            if i % 20 == 0 or i == len(cnpjs_para_buscar):
                print(f"   ... Processados {i}/{len(cnpjs_para_buscar)}")

    # Atualiza e salva o cache em Parquet
    if novos_registros:
        novos_df = pd.DataFrame(novos_registros)
        cache_final = (
            pd.concat([cache_df, novos_df], ignore_index=True)
            if not cache_df.empty
            else novos_df
        )
        cache_final = cache_final.drop_duplicates(subset=["CNPJ"], keep="last")
        cache_final.to_parquet(CACHE_FILE, engine="pyarrow", index=False)
        print(
            f"💾 Cache atualizado com sucesso! Total: {len(cache_final)} CNPJs em {time.time() - t0:.2f}s"
        )


if __name__ == "__main__":
    # Exemplo de teste simples isolado
    cnpjs_teste = ["00000000000191", "33000167000101"]
    enriquecer_base_paralelo(cnpjs_teste)