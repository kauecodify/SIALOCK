# SIALOCK - Sistema Integrado de Analise de Locks

## Visao Geral

SIALOCK e um sistema integrado para analise de dados de importacoes, deteccao de anomalias e geracao de insights para compliance e tomadas de decisao. Este repositorio contem a implementacao completa do Gate 3 (G3) do projeto, com modulos especializados para processamento em larga escala, calibracao de probabilidades e geracao de dados sinteticos.

## Estrutura do Repositorio

```
SIALOCK/
├── SIALOCK_T24_G3/                 # Implementacao do Gate 3
│   ├── gate3/
│   │   ├── __init__.py
│   │   ├── duckdb_backend.py       # Backend DuckDB para operacoes out-of-core
│   │   ├── calibration.py          # Modulo de calibracao de probabilidades
│   │   ├── synthetic.py            # Geracao de dados sinteticos e validacao K-Anonymity
│   │   └── extract_v3.4.py         # Pipeline principal de processamento
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_duckdb_backend.py  # Testes para o backend DuckDB
│   │   ├── test_calibration.py     # Testes para o modulo de calibracao
│   │   └── test_synthetic.py       # Testes para o modulo de dados sinteticos
│   ├── scripts/
│   │   ├── __init__.py
│   │   └── validar_modulos.py      # Script de validacao ponta a ponta
│   ├── README.md                   # Documentacao detalhada do Gate 3
│   └── requirements.txt            # Dependencias do projeto
├── cache_externo/
├── Documentation/
├── extracts/
├── gate3/
├── LICENSE
└── SIALOCK.docx
```

## SIALOCK_T24_G3 - Gate 3

O Gate 3 e a versao mais avancada do sistema, com os seguintes modulos:

### 1. DuckDB Backend

Backend de processamento de dados em larga escala usando DuckDB. Permite:
- Leitura direta de arquivos Parquet/CSV sem carregar em memoria
- Execucao de operacoes SQL complexas (JOIN, GROUP BY, Window Functions) em disco
- Streaming de dados em lotes para integracao com modelos de machine learning
- Spill-to-disk automatico com limite de memoria configuravel

### 2. Calibracao de Probabilidades

Modulo para ajustar probabilidades geradas por modelos de machine learning, garantindo que sejam estatisticamente confiaveis. Inclui:
- Platt Scaling (sigmoide logistico)
- Isotonic Regression (nao-parametrico, monotono)
- Temperature Scaling (ideal para ensembles)
- Beta Calibration (mais flexivel que Platt)

### 3. Synthetic Data Generator

Modulo para geracao de dados sinteticos que preservam as caracteristicas dos dados originais, com validacao de privacidade:
- Gaussian Copula para preservar marginais e correlacoes
- Validador de K-Anonymity
- Metricas de risco de reidentificacao (DCR, Taxa de Divulgacao)

## Como Usar o Gate 3

### Instalacao

```bash
cd SIALOCK_T24_G3
pip install -r requirements.txt
```

### Validacao dos Modulos

```bash
PYTHONPATH=/workspace/github__kauecodify__SIALOCK/SIALOCK_T24_G3 python SIALOCK_T24_G3/scripts/validar_modulos.py
```

### Execucao dos Testes

```bash
PYTHONPATH=/workspace/github__kauecodify__SIALOCK/SIALOCK_T24_G3 python -m pytest SIALOCK_T24_G3/tests/ -v
```

### Processamento de Dados

```bash
PYTHONPATH=/workspace/github__kauecodify__SIALOCK/SIALOCK_T24_G3 python SIALOCK_T24_G3/gate3/extract_v3.4.py --base base2 --backend duckdb
```

## Documentacao Detalhada

Para a documentacao completa do Gate 3, consulte:
- [SIALOCK_T24_G3/README.md](SIALOCK_T24_G3/README.md)

Este arquivo contem:
- Descricao detalhada de todos os modulos
- API completa de cada classe e metodo
- Exemplos de uso
- Boas praticas
- Solucao de problemas

## Contribuindo

1. Faca um fork do repositorio
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudancas (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licenca

Este projeto e licenciado sob a licenca MIT. Veja [LICENSE](LICENSE) para mais detalhes.

## Contato

Para duvidas ou suporte, entre em contato com a equipe do projeto SIALOCK.
