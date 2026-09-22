# SIALOCK

## External Enrichment, Risk Intelligence & Supervised CNPJ Prediction Pipeline

SIALOCK is a data intelligence and machine learning pipeline designed to process corporate datasets, enrich CNPJ records with external information, calculate data-quality and business-risk indicators, and predict the most likely CNPJ associated with a given operation or business record.

The system combines **local XLSX and Parquet datasets**, external corporate data sources, feature engineering, risk scoring, anomaly detection, supervised machine learning, incremental training, probability calibration, and interactive CNPJ querying.

Its primary purpose is to support **corporate data reconciliation, import/export auditing, entity identification, risk analysis, and automated CNPJ matching** in environments where information is distributed across multiple datasets.

---

## Overview

The SIALOCK pipeline follows this process:

```text
Local XLSX / Parquet Data
          │
          ▼
Data Loading & Normalization
          │
          ▼
CNPJ Consolidation
          │
          ▼
External Data Enrichment
          │
          ├── BrasilAPI
          ├── Minha Receita
          └── IBGE / Regional Mapping
          │
          ▼
Local Parquet Cache
          │
          ▼
Data Quality KPIs
          │
          ▼
Business KPIs
          │
          ▼
Risk Score & Outlier Detection
          │
          ▼
Feature Engineering
          │
          ▼
Supervised Machine Learning
          │
          ├── SGD Logistic Model
          ├── SGD Modified Huber
          ├── Calibrated Classifier
          └── Incremental Training
          │
          ▼
Model Evaluation
          │
          ▼
CNPJ Prediction
          │
          ├── Predicted CNPJ
          ├── Confidence
          └── Top-5 Predictions
          │
          ▼
XLSX / Parquet / JSON Outputs
          │
          ▼
Interactive CNPJ Query
```

---

## Main Purpose

SIALOCK is designed to solve a practical entity-resolution problem:

> Given a corporate operation containing multiple attributes, identify the CNPJ that is most likely associated with that record.

Instead of relying exclusively on an exact CNPJ match, the system combines several characteristics of the operation and company, including:

* Corporate names
* Products
* NCM descriptions
* Suppliers
* CNAE
* State and country
* Importer
* HS Code
* Capital social
* Import values
* Weight
* Freight
* Corporate registration status
* Company size
* Legal nature
* Company age
* Number of partners
* Number of secondary CNAEs
* Data completeness
* Importation behavior
* Statistical outliers
* External registration consistency

These variables are transformed into machine-learning features and used to classify the most probable CNPJ.

---

# Features

## 1. Local XLSX and Parquet Processing

The pipeline automatically scans the configured data directory and loads:

* `.xlsx`
* `.parquet`

Files without a `CNPJ` column are ignored.

CNPJs are normalized by removing non-numeric characters and padding the value to 14 digits.

Example:

```text
12.345.678/0001-90
```

becomes:

```text
12345678000190
```

Multiple datasets can be progressively merged using the normalized CNPJ as the key.

The system also generates metadata for each input file, including:

* File name
* File type
* Number of rows
* Number of columns
* Header hash
* Data completeness

---

# 2. External CNPJ Enrichment

SIALOCK can optionally enrich the local datasets using external data sources.

### BrasilAPI

The system retrieves corporate registration information such as:

* Corporate legal name
* Trade name
* CNAE code
* CNAE description
* Legal nature
* State
* Municipality
* Registration status
* Company size
* Share capital
* Company opening date
* Number of partners
* Number of secondary CNAEs

### Minha Receita

Additional information can be obtained, including:

* Simples Nacional status
* MEI status
* Email
* Telephone

### IBGE / Regional Classification

The pipeline derives the Brazilian geographic region from the company's state:

```text
North
Northeast
Central-West
Southeast
South
```

The regional information becomes an additional machine-learning feature.

### Comex Stat

The architecture contains fields intended for foreign-trade enrichment, including:

```text
EXT_TotalImportacoes
EXT_ValorTotalImportado
```

These fields are currently represented as placeholders in the enrichment stage when no external Comex data is populated.

---

# 3. Local API Cache

External queries are cached locally in Parquet format.

Default location:

```text
./cache_externo/cnpj_cache.parquet
```

The cache prevents unnecessary repeated API requests for CNPJs that have already been processed.

The system also limits the number of external CNPJs processed in a single execution:

```python
MAX_CNPJ_EXTERNO = 500
```

This helps control request volume and external API usage.

---

# 4. Data Quality KPIs

SIALOCK calculates structural and quality indicators before the machine-learning stage.

Examples include:

```text
Total rows
Total columns
Average completeness
Duplicated rows
Unique CNPJs
CNPJ duplication rate
Number of XLSX files
Number of Parquet files
Unique header hashes
```

These indicators provide a high-level view of the quality and structure of the input data.

---

# 5. Business KPIs

The system derives business-oriented indicators from the available operational data.

### Underpricing Indicator

When import value and weight are available, the pipeline calculates:

```text
Import Value / Weight
```

and compares each record against the dataset median.

The resulting indicator is represented by:

```text
KPI_Subvalorizacao
```

---

### HS-Code Disparity

The pipeline analyzes the HS Code and product description to generate:

```text
KPI_DisparidadeHS
```

This provides a numerical indicator based on the relationship between the HS Code structure and description length.

---

### Import Volume Surge

For each importer, the system calculates historical mean and standard deviation of import values.

The resulting indicator is:

```text
KPI_SurgeVolume
```

It identifies records whose import value deviates substantially from the importer's observed distribution.

---

### ETA Deviation

When both expected and actual arrival dates exist, SIALOCK calculates:

```text
KPI_DesvioETA
```

based on the absolute difference between the expected and actual dates.

---

### Data Completeness

Each record receives:

```text
KPI_Completude
```

which measures the proportion of missing information in the row.

---

# 6. External Corporate Risk Indicators

External company information is also converted into machine-learning and risk features.

### Corporate Registration Consistency

The system checks the external registration status and generates:

```text
KPI_Ext_ConsistenciaCadastral
```

Records associated with a non-active registration status receive a higher risk contribution.

### Company Age

Company age is transformed into:

```text
KPI_Ext_IdadeEmpresa
```

The implementation assigns a higher risk contribution to younger companies according to the configured scoring formula.

---

# 7. Outlier Detection

SIALOCK uses:

```python
LocalOutlierFactor
```

to detect unusual numerical patterns.

The analysis considers available operational numerical variables such as:

* Capital social
* Import value
* Weight
* Freight

The resulting anomaly indicator is:

```text
KPI_Outlier
```

---

# 8. Risk Score

The pipeline combines multiple indicators into a weighted risk score between:

```text
0 - 100
```

The current weighting configuration is:

| Indicator             | Weight |
| --------------------- | -----: |
| Underpricing          |    25% |
| HS disparity          |    15% |
| Import volume surge   |    15% |
| ETA deviation         |    10% |
| Data completeness     |     5% |
| Outlier detection     |     5% |
| Corporate consistency |    15% |
| Company age           |    10% |

The resulting score is stored as:

```text
RiskScore
```

Risk levels are classified as:

```text
0–24.99    → LOW
25–49.99   → MEDIUM
50–74.99   → HIGH
75–100     → CRITICAL
```

The generated XLSX report uses color coding to make these classifications easier to inspect.

---

# 9. Feature Engineering

SIALOCK automatically constructs a machine-learning preprocessing pipeline.

### Text Features

Text fields are combined and transformed using:

```text
TF-IDF
```

with:

```text
ngram_range = (1, 2)
max_features = 5000
sublinear_tf = True
strip_accents = "unicode"
```

Examples:

```text
RazaoSocial
Produto
DescricaoNCM
Fornecedor
EXT_RazaoSocial
EXT_NomeFantasia
EXT_CNAE_Descricao
```

---

### Categorical Features

Categorical variables are transformed using:

```text
OneHotEncoder
```

with unknown categories handled automatically.

Examples include:

```text
CNAE
UF
PaisOrigem
SituacaoCadastral
HS_Code
Importador
EXT_UF
EXT_Municipio
EXT_Situacao
EXT_Porte
EXT_NaturezaJuridica
EXT_Regiao
```

---

### Numerical Features

Numerical variables are processed through:

```text
Median Imputation
+
Standard Scaling
```

Examples include:

```text
CapitalSocial
ValorImportado
PesoKg
ValorFrete
EXT_CapitalSocial
EXT_IdadeEmpresa
EXT_QtdSocios
EXT_QtdCNAESecundarios
RiskScore
```

---

# 10. Supervised Machine Learning

The target variable is:

```text
CNPJ
```

Each distinct CNPJ is represented as a classification class using:

```python
LabelEncoder
```

The model therefore learns relationships between company/operation characteristics and the CNPJ associated with those records.

---

## SGD Ensemble

SIALOCK implements an ensemble containing two SGD-based classifiers:

```text
SGDClassifier(loss="log_loss")
SGDClassifier(loss="modified_huber")
```

Their probability outputs are combined using weighted probabilities.

The ensemble provides the primary classification model.

---

# 11. Probability Calibration

The pipeline also attempts to train a calibrated classifier using:

```python
CalibratedClassifierCV
```

with:

```text
Sigmoid calibration
```

This provides an additional model whose probability estimates can be compared with the ensemble and incremental approaches.

---

# 12. Incremental Training

SIALOCK supports incremental learning using:

```python
partial_fit()
```

The training data is processed in batches:

```python
BATCH_SIZE = 64
```

Each batch is evaluated using:

* Accuracy
* F1-score

This provides a training history and allows the system to process data incrementally.

---

# 13. Model Evaluation

The available models are evaluated using:

```text
Accuracy
F1 Macro
Precision Macro
Recall Macro
Top-1 Accuracy
Top-3 Accuracy
Top-5 Accuracy
Top-10 Accuracy
Log Loss
```

The pipeline compares the available trained models and selects the model with the highest configured Top-5 performance when available, otherwise falling back to accuracy.

---

# 14. CNPJ Prediction

After model selection, the system generates predictions for the complete dataset.

The output includes:

```text
CNPJ_Previsto
CNPJ_Confianca
```

and the five highest-probability candidates:

```text
CNPJ_Top1
CNPJ_Top1_p

CNPJ_Top2
CNPJ_Top2_p

CNPJ_Top3
CNPJ_Top3_p

CNPJ_Top4
CNPJ_Top4_p

CNPJ_Top5
CNPJ_Top5_p
```

This makes the prediction more useful than a simple single-class output because alternative candidates and their associated probabilities are preserved.

---

# 15. Interactive CNPJ Query

After the main pipeline finishes, SIALOCK provides an interactive PyAutoGUI interface.

The user can enter the values of the configured feature fields and request a prediction.

The system processes the input through the same preprocessing pipeline and returns the five most probable CNPJs.

Example output:

```text
1º CNPJ 12345678000190 (p=0.8421)
2º CNPJ 98765432000110 (p=0.0914)
3º CNPJ 11222333000144 (p=0.0372)
4º CNPJ 55666777000188 (p=0.0183)
5º CNPJ 99888777000155 (p=0.0107)
```

---

# 16. Model Persistence

The selected model and its preprocessing components are persisted using `joblib`.

Default path:

```text
./modelo_importacao_externo.joblib
```

The persisted object contains:

```text
Model
Preprocessor
Label Encoder
Model Metrics
Target Columns
Model Type
External Enrichment Flag
```

This allows the trained pipeline to be reused without reconstructing the entire training process.

---

# 17. Output Files

The system generates three primary output formats.

### Parquet

Contains the complete enriched and analyzed dataset.

Example:

```text
saida/auditoria_externo_YYYYMMDD_HHMMSS.parquet
```

---

### XLSX

Contains the highest-risk records, sorted by `RiskScore`.

Example:

```text
saida/risco_colorido_externo_YYYYMMDD_HHMMSS.xlsx
```

The spreadsheet contains visual risk classification:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

---

### JSON

Contains consolidated pipeline metrics.

Example:

```text
saida/kpis_externo_YYYYMMDD_HHMMSS.json
```

The JSON includes:

* Data quality metrics
* Risk distribution
* Machine-learning metrics
* Selected model
* Number of CNPJ classes
* External enrichment status
* Number of external columns

---

# Project Structure

A typical project structure is:

```text
SIALOCK/
│
├── dados/
│   ├── input.xlsx
│   └── input.parquet
│
├── cache_externo/
│   └── cnpj_cache.parquet
│
├── saida/
│   ├── auditoria_externo_*.parquet
│   ├── risco_colorido_externo_*.xlsx
│   └── kpis_externo_*.json
│
├── modelo_importacao_externo.joblib
│
├── main.py
└── README.md
```

---

# Installation

Install the required Python dependencies:

```bash
pip install numpy pandas pyarrow openpyxl joblib requests pyautogui tenacity scikit-learn
```

Or install them from a requirements file:

```bash
pip install -r requirements.txt
```

---

# Requirements

Recommended environment:

```text
Python 3.10+
```

Main libraries:

```text
pandas
numpy
pyarrow
openpyxl
scikit-learn
requests
joblib
pyautogui
tenacity
```

---

# Running the Application

Place the source datasets inside:

```text
./dados
```

Then run:

```bash
python main.py
```

The application will display a PyAutoGUI dialog asking whether external enrichment should be enabled.

### With external enrichment

The system queries the configured external services and stores the results in the local cache.

### Without external enrichment

The pipeline uses the information available in the local datasets.

---

# Data Flow

The complete operational flow is:

```text
1. Load XLSX / Parquet
        ↓
2. Normalize CNPJs
        ↓
3. Merge datasets
        ↓
4. Query external sources
        ↓
5. Update local cache
        ↓
6. Calculate data-quality KPIs
        ↓
7. Calculate business KPIs
        ↓
8. Detect numerical outliers
        ↓
9. Calculate RiskScore
        ↓
10. Build ML features
        ↓
11. Encode CNPJ classes
        ↓
12. Split training/test datasets
        ↓
13. Train SGD ensemble
        ↓
14. Train calibrated classifier
        ↓
15. Train incremental model
        ↓
16. Evaluate models
        ↓
17. Select model
        ↓
18. Generate CNPJ predictions
        ↓
19. Export Parquet / XLSX / JSON
        ↓
20. Enable interactive CNPJ queries
```

---

# Use Cases

SIALOCK can be applied to scenarios involving:

* Corporate entity resolution
* CNPJ matching
* Import/export data analysis
* Corporate data reconciliation
* Supplier identification
* Importer identification
* Corporate registration analysis
* Data quality auditing
* Operational anomaly detection
* Risk-based prioritization
* Business intelligence
* Corporate data enrichment
* Automated classification of business records

---

# Architecture

The project combines several data and machine-learning layers:

```text
Data Sources
     │
     ├── Internal datasets
     ├── BrasilAPI
     ├── Minha Receita
     └── IBGE / regional information
     │
     ▼
Data Integration
     │
     ▼
Feature Engineering
     │
     ├── Text
     ├── Categorical
     ├── Numerical
     └── Risk / KPI features
     │
     ▼
Machine Learning
     │
     ├── SGD Log Loss
     ├── SGD Modified Huber
     ├── Calibration
     └── Incremental Learning
     │
     ▼
Prediction & Risk Intelligence
     │
     ├── CNPJ Prediction
     ├── Probability
     ├── Top-5 Candidates
     └── Risk Score
     │
     ▼
Business Outputs
     │
     ├── Parquet
     ├── XLSX
     └── JSON
```

---

# Important Implementation Notes

The external enrichment layer depends on the availability and behavior of the configured external APIs. API responses may be unavailable, incomplete, rate-limited, or changed by the external provider.

The local cache is therefore an important part of the pipeline and allows previously retrieved CNPJ information to be reused.

The Comex Stat integration fields are present in the data architecture, while the current implementation initializes the corresponding import indicators as placeholders when they are not populated by an external Comex query.

The machine-learning output should be interpreted as a **probabilistic classification result**, not as an authoritative determination of corporate identity. The `CNPJ_Confianca` and Top-5 probabilities represent the model's estimated probabilities within the trained class space.

---

# Technology Stack

| Layer                | Technology                       |
| -------------------- | -------------------------------- |
| Language             | Python                           |
| Data Processing      | Pandas / NumPy                   |
| Storage              | Parquet                          |
| Spreadsheet          | XLSX / OpenPyXL                  |
| HTTP                 | Requests                         |
| Retry Strategy       | Tenacity                         |
| Machine Learning     | Scikit-learn                     |
| Text Processing      | TF-IDF                           |
| Categorical Encoding | One-Hot Encoding                 |
| Classification       | SGDClassifier                    |
| Calibration          | CalibratedClassifierCV           |
| Anomaly Detection    | LocalOutlierFactor               |
| Model Persistence    | Joblib                           |
| Interface            | PyAutoGUI                        |
| External Data        | BrasilAPI / Minha Receita / IBGE |

---

# Summary

SIALOCK is a **corporate data intelligence and supervised machine-learning pipeline for CNPJ identification and operational risk analysis**.

It combines local business data with external corporate information, transforms the resulting dataset into structured analytical features, calculates quality and risk indicators, trains multiple classification strategies, evaluates their performance, and generates probabilistic CNPJ predictions.

The application is particularly focused on environments where corporate records must be **enriched, reconciled, classified, audited, and prioritized automatically**.

At its core, SIALOCK connects:

```text
Corporate Data
+
External Enrichment
+
Business KPIs
+
Risk Intelligence
+
Machine Learning
=
Automated CNPJ Intelligence
```
