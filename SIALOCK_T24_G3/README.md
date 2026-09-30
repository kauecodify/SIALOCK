# SIALOCK_T24_G3 - Documentacao Tecnica

## Visao Geral

SIALOCK_T24_G3 e um projeto de processamento de dados em larga escala para analise de importacoes, com foco em deteccao de anomalias e geracao de insights para compliance e tomadas de decisao. O sistema e projetado para processar bases de dados com mais de 23 milhoes de registros de forma eficiente, utilizando tecnologias modernas de processamento out-of-core e tecnicas de machine learning.

## Estrutura do Projeto

```
SIALOCK_T24_G3/
├── gate3/
│   ├── __init__.py
│   ├── duckdb_backend.py       - Backend DuckDB para operacoes out-of-core
│   ├── calibration.py          - Modulo de calibracao de probabilidades
│   ├── synthetic.py            - Geracao de dados sinteticos e validacao K-Anonymity
│   └── extract_v3.4.py         - Pipeline principal de processamento
├── tests/
│   ├── __init__.py
│   ├── test_duckdb_backend.py  - Testes para o backend DuckDB
│   ├── test_calibration.py     - Testes para o modulo de calibracao
│   └── test_synthetic.py       - Testes para o modulo de dados sinteticos
├── scripts/
│   ├── __init__.py
│   └── validar_modulos.py      - Script de validacao ponta a ponta
└── requirements.txt            - Dependencias do projeto
```

## Modulos Principais

### 1. DuckDB Backend (gate3/duckdb_backend.py)

O modulo DuckDB Backend fornece uma interface para operacoes de processamento de dados em larga escala sem a necessidade de carregar todos os dados na memoria RAM. Este modulo e especialmente util para processar bases de dados com dezenas de milhoes de registros.

#### Funcionalidades:

- **Registro de fontes de dados**: Suporta registro de arquivos Parquet e CSV como views no DuckDB
- **KPIs de negocio**: Calcula metricas agregadas diretamente no banco de dados sem carregar dados em memoria
- **Operacoes de window functions**: Implementa operacoes como Top-K por grupo usando QUALIFY
- **Detecao de outliers**: Identifica outliers usando o metodo IQR (Interquartile Range)
- **Streaming Arrow**: Permite iterar sobre resultados de consultas em lotes, ideal para processamento em memoria limitada
- **Exportacao de dados**: Exporta resultados de consultas para arquivos Parquet

#### Classes e Metodos:

**Classe: DuckDBBackend**

- `__init__(db_path, memory_limit, threads, temp_dir)`: Inicializa a conexao com o DuckDB
  - `db_path`: Caminho para o banco de dados (padrao: ":memory:")
  - `memory_limit`: Limite de memoria (padrao: "4GB")
  - `threads`: Numero de threads (padrao: None - auto)
  - `temp_dir`: Diretorio temporario para spill-to-disk

- `register_parquet(view, path)`: Registra uma view sobre arquivos Parquet
- `register_csv(view, path, sep, header)`: Registra uma view sobre arquivos CSV
- `register_auto(view, path)`: Detecta automaticamente o tipo de arquivo (Parquet ou CSV)

- `business_kpis(view)`: Retorna um DataFrame com KPIs agregados
  - total_registros: Numero total de registros
  - cnpjs_unicos: Numero de CNPJs unicos
  - ncms_distintos: Numero de NCMs distintos
  - paises_distintos: Numero de paises distintos
  - vmle_medio: Valor medio de vmle_dolar
  - vmle_total: Valor total de vmle_dolar
  - peso_medio: Peso medio liquido
  - vmle_p95: Percentil 95 de vmle_dolar
  - vmle_p99: Percentil 99 de vmle_dolar

- `topk_por_grupo(view, group_col, score_col, k)`: Retorna os top-K registros por grupo
- `outliers_iqr(view, col)`: Identifica outliers baseados no metodo IQR
- `iter_arrow_batches(sql, batch_size)`: Itera sobre resultados em lotes Arrow
- `iter_pandas_batches(sql, batch_size)`: Itera sobre resultados em lotes Pandas
- `export_parquet(sql, out)`: Exporta resultados para arquivo Parquet
- `close()`: Fecha a conexao com o banco de dados

#### Vantagens em relacao ao Pandas:

- Le Parquet/CSV diretamente do disco (zero RAM para scan completo)
- Executa operacoes SQL com window functions, GROUP BY, JOIN em disco
- Spill-to-disk automatico (memory_limit configuravel)
- Utiliza threads nativas (usa todos os cores disponiveis)
- Streaming Arrow zero-copy para integracao com modelos de machine learning

### 2. Calibracao de Probabilidades (gate3/calibration.py)

O modulo de calibracao de probabilidades tem como objetivo ajustar as probabilidades geradas por modelos de machine learning para que sejam estatisticamente confiaveis. Um modelo bem calibrado e aquele em que a probabilidade prevista corresponde a frequencia real de acertos.

#### Metodos de Calibracao Implementados:

1. **Platt Scaling**: Ajusta uma funcao sigmoide logistica aos dados. E um metodo parametrico simples e eficiente.
2. **Isotonic Regression**: Metodo nao-parametrico que produz uma funcao monotona. E mais flexivel que Platt Scaling mas pode overfitar.
3. **Temperature Scaling**: Ajusta um unico parametro (temperatura) para escalar os logits. Ideal para redes neurais e ensembles.
4. **Beta Calibration**: Mais flexivel que Platt Scaling, usa uma transformacao baseada na distribuicao Beta.

#### Metricas de Avaliacao:

- **Brier Score**: Medida de acuracia das probabilidades. Quanto menor, melhor. Range: [0, 1]
- **ECE (Expected Calibration Error)**: Erro medio de calibracao. Quanto menor, melhor.
- **MCE (Maximum Calibration Error)**: Maior erro de calibracao em qualquer bin.
- **Reliability Diagram**: Diagrama que compara a confianca media com a acuracia media em cada bin de probabilidade.

#### Classes e Metodos:

**Funcoes de Metricas:**

- `brier_score(y_true, p)`: Calcula o Brier Score
- `expected_calibration_error(y_true, p, n_bins)`: Calcula o ECE
- `maximum_calibration_error(y_true, p, n_bins)`: Calcula o MCE
- `reliability_diagram(y_true, p, n_bins)`: Gera dados para o diagrama de confiabilidade

**Classes de Calibradores:**

- `PlattCalibrator`: Implementa Platt Scaling
- `IsotonicCalibrator`: Implementa Isotonic Regression
- `BetaCalibrator`: Implementa Beta Calibration
- `TemperatureScaling`: Implementa Temperature Scaling

**Classe: AutoCalibrator**

- `__init__(candidates, n_bins)`: Inicializa com lista de metodos candidatos e numero de bins
- `fit(p, y)`: Treina todos os calibradores e seleciona o melhor baseado no ECE
- `calibrate(p)`: Aplica a calibracao usando o melhor metodo
- `reliability(p, y)`: Gera o diagrama de confiabilidade

**Classe: CalibrationReport**

Contem o relatorio de calibracao com:
- method: Metodo selecionado
- brier_before/after: Brier Score antes e depois
- ece_before/after: ECE antes e depois
- mce_before/after: MCE antes e depois
- temperature: Temperatura (para Temperature Scaling)
- params: Parametros adicionais

### 3. Synthetic Data Generator (gate3/synthetic.py)

O modulo de geracao de dados sinteticos tem como objetivo criar dados artificiais que preservam as caracteristicas estatisticas dos dados originais, ao mesmo tempo em que garantem a privacidade dos dados sensiveis. Este modulo e especialmente importante para compliance com regulamentacoes como LGPD e Portaria RFB 100/2021.

#### Estrategia de Geracao:

1. **Gaussian Copula**: Metodo rapido e deterministico que preserva:
   - Distribuicoes marginais de cada variavel
   - Correlacoes entre variaveis numericas
   - Distribuicoes categoricas
   - Ranges de datas

2. **Detecao Automatica de Tipos**: Classifica colunas em:
   - Numericas
   - Categoricas
   - Datetime

3. **Validacao K-Anonymity**: Verifica se os dados sinteticos atendem ao criterio de K-Anonymity

4. **Metricas de Risco de Reidentificacao**:
   - DCR (Distance to Closest Record)
   - Taxa de Divulgacao

#### Classes e Metodos:

**Funcoes de Utilidade:**

- `_classify_columns(df)`: Classifica colunas por tipo

**Classe: GaussianCopulaGenerator**

- `__init__(seed, max_categories)`: Inicializa o gerador
- `fit(df)`: Ajusta o modelo aos dados originais
- `sample(n)`: Gera uma amostra de n registros sinteticos

**Classe: KAnonymityValidator**

- `__init__(quasi_ids, k)`: Inicializa com quasi-identificadores e valor de k
- `evaluate(original, sintetico)`: Avalia K-Anonymity em ambos os datasets

**Classe: KAnonReport**

Contem o relatorio de K-Anonymity com:
- k_min_original/sintetico: Menor grupo em cada dataset
- k_medio_original/sintetico: Tamanho medio dos grupos
- grupos_violados_original/sintetico: Numero de grupos com tamanho < k
- registros_descartados_pct: Percentual de registros em grupos violados
- passa_k: Booleano indicando se passa no teste K-Anonymity

**Funcoes de Metricas de Risco:**

- `distancia_mais_proximo_vizinho(original, sintetico, num_cols, sample)`: Calcula DCR
- `taxa_divulgacao(original, sintetico, quasi_ids)`: Calcula taxa de divulgacao

**Funcao Principal:**

- `gerar_e_validar(original, quasi_ids, k, seed)`: Gera dados sinteticos e retorna (df_sintetico, relatorio)

### 4. Pipeline Principal (gate3/extract_v3.4.py)

O pipeline principal integra todos os modulos para processar as bases de dados de forma eficiente e gerar os artefatos necessarios para analise e compliance.

#### Fluxo de Processamento:

1. **KPIs Rapidos via DuckDB**: Calcula metricas agregadas sem carregar todos os dados em memoria
2. **Treinamento do Ensemble**: Processa dados em lotes para treinamento do modelo
3. **Calibracao de Probabilidades**: Aplica calibracao nas probabilidades geradas pelo modelo
4. **Teste de K-Anonymity**: Gera dados sinteticos e valida compliance com K-Anonymity
5. **Top-K via DuckDB**: Identifica os top-K registros por grupo usando window functions

#### Funcoes Principais:

- `processar_base(base, backend, resume)`: Processa uma base de dados completa
- `_feature_engineering(df)`: Aplica transformacoes de features
- `_to_matrix(df)`: Converte DataFrame para matriz de features
- `iter_auto(path, batch_size)`: Itera sobre arquivos Parquet/CSV em lotes

**Classe: Gate3Ensemble**

- `partial_fit(X, y, classes)`: Treina o modelo de forma incremental
- `score(X)`: Retorna probabilidades para os dados de entrada

## Requisitos do Sistema

### Dependencias (requirements.txt):

- pandas>=2.2: Manipulacao de dados tabulares
- numpy>=1.26: Operacoes numericas
- scikit-learn>=1.4: Algoritmos de machine learning
- scipy>=1.12: Funcoes cientificas avancadas
- pyarrow>=15: Suporte a formato Arrow e Parquet
- polars>=0.20: Processamento de dados alternativo
- duckdb>=1.0: Banco de dados analitico embutido
- openpyxl>=3.1: Suporte a arquivos Excel
- joblib>=1.4: Serializacao de objetos Python
- psutil>=5.9: Monitoramento de recursos
- requests>=2.31: Requisicoes HTTP
- httpx>=0.27: Requisicoes HTTP assincronas
- tenacity>=8.3: Retry com backoff
- pandera>=0.19: Validacao de schemas de dados
- pydantic>=2.7: Validacao de dados com type hints
- pyyaml>=6.0: Suporte a YAML
- tqdm>=4.66: Barras de progresso
- orjson>=3.10: Serializacao JSON rapida
- pytest>=8.0: Framework de testes

## Instalacao e Configuracao

### Passo 1: Clonar o Repositorio

```bash
cd /workspace
```

O projeto ja esta clonado em `/workspace/SIALOCK_T24_G3`

### Passo 2: Instalar Dependencias

```bash
cd /workspace/SIALOCK_T24_G3
pip install -r requirements.txt
```

### Passo 3: Configurar Variaveis de Ambiente

Defina a variavel PYTHONPATH para que os modulos sejam encontrados:

```bash
export PYTHONPATH=/workspace/SIALOCK_T24_G3:$PYTHONPATH
```

Ou execute os comandos com PYTHONPATH explicitamente:

```bash
PYTHONPATH=/workspace/SIALOCK_T24_G3 python scripts/validar_modulos.py
```

## Execucao do Projeto

### Validacao dos Modulos

Para validar que todos os modulos estao funcionando corretamente:

```bash
PYTHONPATH=/workspace/SIALOCK_T24_G3 python scripts/validar_modulos.py
```

Este script:
1. Cria dados de teste sinteticos (5000 registros)
2. Testa o modulo DuckDB (KPIs, Top-K, Streaming)
3. Testa o modulo de Calibracao (ECE, Brier Score)
4. Testa o modulo Synthetic (K-Anonymity, DCR, Taxa de Divulgacao)

### Execucao dos Testes

Para executar todos os testes unitarios:

```bash
PYTHONPATH=/workspace/SIALOCK_T24_G3 python -m pytest tests/ -v
```

Este comando executa 12 testes distribuidos em 3 arquivos:
- test_duckdb_backend.py: 3 testes
- test_calibration.py: 3 testes
- test_synthetic.py: 6 testes

### Processamento das Bases de Dados

Para processar uma base de dados especifica:

```bash
PYTHONPATH=/workspace/SIALOCK_T24_G3 python gate3/extract_v3.4.py --base base2 --backend duckdb --resume
```

Ou como modulo:

```bash
PYTHONPATH=/workspace/SIALOCK_T24_G3 python -m gate3.extract_v3.4 --base base2 --backend duckdb --resume
```

#### Argumentos:

- `--base`: Base a processar (base2 ou base3)
- `--backend`: Backend a usar (auto, duckdb, pandas)
- `--resume`: Retomar processamento a partir do checkpoint

## Diretorio de Dados

O sistema espera que os dados estejam organizados na seguinte estrutura:

```
SIALOCK_T24_G3/
├── data/
│   ├── base2/
│   │   └── entrada/          - Arquivos Parquet/CSV da base 2
│   └── base3/
│       └── entrada/          - Arquivos Parquet/CSV da base 3
```

## Artefatos Gerados

O pipeline gera os seguintes artefatos para cada base processada:

### DuckDB Backend

- `kpis_duckdb.json`: Arquivo JSON com KPIs agregados da base de dados
  - total_registros
  - cnpjs_unicos
  - ncms_distintos
  - paises_distintos
  - vmle_medio
  - vmle_total
  - peso_medio
  - vmle_p95
  - vmle_p99

### Calibracao

- `calibracao.json`: Arquivo JSON com relatorio de calibracao
  - metodo: Metodo de calibracao selecionado
  - brier_antes/depois: Brier Score antes e depois da calibracao
  - ece_antes/depois: ECE antes e depois da calibracao
  - diagrama: Dados para o diagrama de confiabilidade

### Synthetic Data

- `k_anonymity_report.json`: Arquivo JSON com relatorio de K-Anonymity
  - k_anonymity: Relatorio completo do K-Anonymity
  - dcr: Metricas de Distance to Closest Record
  - divulgacao: Taxa de divulgacao
  - n_original: Numero de registros originais
  - n_sintetico: Numero de registros sinteticos

## Aplicacoes no Contexto do Projeto

### Para o Pitch Pre-Seed (R$ 200M)

Os modulos entregam os seguintes beneficios para o pitch:

1. **DuckDB**: Demonstra escalabilidade real
   - Capacidade de processar 23 milhoes de registros em minutos
   - Uso de apenas 4GB de RAM
   - Operacoes complexas (JOIN, GROUP BY, Window Functions) em disco
   - Argumento: "Processamos 23M registros em X min com 4GB RAM"

2. **Calibracao**: Garante confiança estatistica
   - Probabilidades calibradas com ECE < 5%
   - Relatorio de confiabilidade auditavel
   - Argumento: "Nossas probabilidades sao estatisticamente confiaveis (ECE < 5%)"

3. **Synthetic Data**: Compliance com regulamentacoes
   - K-Anonymity >= 5 comprovado
   - Dados sinteticos auditaveis
   - Argumento: "Provamos K-Anonymity >= 5 com dados sinteticos auditaveis - LGPD-safe"

### Ganchos de Venda Tecnicos

- **Escalabilidade**: DuckDB elimina o SPOF de memoria, permitindo processar grandes volumes sem infraestrutura cara
- **Confianca**: Calibracao garante que as probabilidades sao confiaveis, essencial para compliance bancario
- **Privacidade**: Synthetic Data resolve questoes de LGPD e Portaria RFB 100/2021 com base matematica

## Boas Praticas

### Gerenciamento de Memoria

- Use o backend DuckDB para operacoes em grandes bases de dados
- Evite carregar DataFrames completos em memoria
- Use os metodos de streaming (`iter_arrow_batches`, `iter_pandas_batches`) para processar dados em lotes

### Reprodutibilidade

- Defina sempre o parametro `seed` para funcoes que usam aleatoriedade
- O `GaussianCopulaGenerator` aceita seed para reprodutibilidade
- Os calibradores usam solvers deterministicos quando possivel

### Performance

- Ajuste o `memory_limit` do DuckDB de acordo com a memoria disponivel
- Use `threads=None` para usar todos os cores disponiveis
- Para operacoes de window functions, prefira DuckDB a Pandas

### Qualidade de Dados

- Valide sempre os dados de entrada
- Verifique os relatorios de KPIs e K-Anonymity
- Monitore as metricas de calibracao (ECE, Brier Score)

## Solucao de Problemas

### Problemas de Importacao

Se encontrar erros de `ModuleNotFoundError`, verifique:

1. Que a variavel PYTHONPATH esta configurada corretamente
2. Que voce esta executando os comandos a partir do diretorio do projeto
3. Que todos os pacotes do requirements.txt estao instalados

### Problemas de Memoria

Se o DuckDB estourar o limite de memoria:

1. Aumente o `memory_limit` ao inicializar o DuckDBBackend
2. Verifique se o `temp_dir` tem espaco suficiente para spill-to-disk
3. Reduza o tamanho dos lotes (`batch_size`) em operacoes de streaming

### Problemas de Calibracao

Se os calibradores nao convergirem:

1. Aumente o numero de iteracoes (`max_iter`) nos calibradores
2. Verifique se os dados de entrada sao validos (probabilidades entre 0 e 1)
3. Tente usar um subconjunto menor dos candidatos no AutoCalibrator

## Contribuicoes

Para contribuir com o projeto:

1. Crie uma branch a partir da branch principal
2. Facam suas alteracoes
3. Adicione testes para nova funcionalidade
4. Execute todos os testes para garantir que nada quebrou
5. Abra um Pull Request com uma descricao detalhada das mudancas

## Licenca

Este projeto e propriedade intelectual da equipe SIALOCK_T24_G3 e destinada ao uso exclusivo no contexto do projeto.

## Contato

Para duvidas ou suporte, entre em contato com a equipe do projeto SIALOCK_T24_G3.
