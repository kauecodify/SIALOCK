```markdown
# Extract V.2

Extract V.2 is an evolutionary extraction pipeline that turns raw files into a scored, enriched, and ML-ready dataset, and can predict the target CNPJ from textual evidence.

It reads local files (XLSX, Parquet, CSV, PDF, TXT), enriches each CNPJ with Brazilian public APIs, computes quality and business KPIs, applies a risk score, trains an ensemble of supervised models across a 10-phase roadmap (F0 to F9), and keeps a persistent history of sessions that detects recurring patterns over time.

---

## Table of Contents

- [How Extract V.2 Works](#how-extract-v2-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Execution Modes](#execution-modes)
- [The 10-Phase Roadmap F0 to F9](#the-10-phase-roadmap-f0-to-f9)
- [External Enrichment](#external-enrichment)
- [Calculated KPIs](#calculated-kpis)
- [Risk Score](#risk-score)
- [Output Data](#output-data)
- [Interactive Console](#interactive-console)
- [Pattern Detection Between Sessions](#pattern-detection-between-sessions)
- [Project Structure](#project-structure)
- [How to Feed the Pipeline](#how-to-feed-the-pipeline)
- [Troubleshooting](#troubleshooting)

---

## How Extract V.2 Works

Every run follows the same cycle. The pipeline is designed so each execution is a new "session", and sessions accumulate over time.

```
              +----------------------+
              |   ./data  (input)    |
              |  XLSX PARQUET CSV    |
              |  PDF TXT             |
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  F0  Bootstrap       |  load + unify by CNPJ
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  F1  Enrichment      |  BrasilAPI / Minha Receita
              |                      |  ReceitaWS / CNPJ.ws / IBGE
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  F2  KPI per column  |  completeness / cardinality
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  F3  Char-Ngram      |  textual fingerprint
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  F4  Warmup          |  ensemble of 5 models
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  F5  Incremental     |  partial_fit in batches
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  F6  Refinement      |  calibration + top-k
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  F7  History         |  append session JSON
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  F8  Pattern         |  trend + recurring KPIs
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  F9  Console         |  query / capture / inspect
              +----------+-----------+
                         |
                         v
              +----------------------+
              |  output/  + model    |
              +----------------------+
```

Each session is time-stamped and appended to the history. After 10 sessions, the pattern detection module flips `ready_for_prediction` to `true` and returns the KPIs that keep appearing at the top of the risk ranking. These KPIs are the "signature" of the target.

---

## Requirements

- Python 3.10 or higher (3.11 recommended)
- Linux, macOS, or Windows
- Internet access for enrichment (optional — the Fast mode works offline)

---

## Installation

### Conda (recommended)

```bash
conda create -n extract2 python=3.11 -y
conda activate extract2
conda install -c conda-forge pandas numpy scikit-learn joblib \
    pyarrow openpyxl requests tenacity pyautogui pyperclip \
    pdfplumber pypdf pillow pytesseract -y
pip install python-Levenshtein
```

### Pip with venv

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install --upgrade pip
pip install pandas numpy scikit-learn joblib pyarrow openpyxl \
    requests tenacity pyautogui pyperclip pdfplumber pypdf \
    pillow pytesseract
```

### Optional — OCR and headless display

```bash
# OCR for scanned PDFs
sudo apt-get install tesseract-ocr tesseract-ocr-por    # Linux
brew install tesseract tesseract-lang                    # macOS
# Windows: https://github.com/UB-Mannheim/tesseract/wiki

# PyAutoGUI on headless Linux
sudo apt-get install python3-tk python3-dev scrot
```

---

## Quick Start

```bash
mkdir -p data
# copy your XLSX / Parquet / CSV / PDF / TXT files into ./data
python extract_v2.py
```

Then pick a mode in the initial dialog: `Full`, `Fast`, or `Cancel`.

---

## Execution Modes

| Mode | Runs | When to use |
|------|------|-------------|
| `Full` | F0 to F9 with external API enrichment | When you want the full signal, online |
| `Fast` | F0 to F9 without enrichment | Offline, quick iteration, cached data |
| `Cancel` | Nothing | Exit immediately |

---

## The 10-Phase Roadmap F0 to F9

| Phase | Name | What it does |
|-------|------|--------------|
| F0 | Bootstrap | Loads XLSX, Parquet, CSV, PDF, TXT. Normalizes CNPJ to 14 digits. Merges files progressively on CNPJ. |
| F1 | Enrichment | Queries BrasilAPI, Minha Receita, ReceitaWS, CNPJ.ws. Caches to Parquet. Derives region, company age, and heuristic credit score. |
| F2 | KPI-All | Computes per-column KPIs (completeness, cardinality, null rate) and exports `kpi_by_column.csv`. |
| F3 | Char-Ngram | Mines 2 to 5-gram character sequences across text columns. Produces `KPI_CharPattern` per row and a global frequency map. |
| F4 | Warmup | Trains an ensemble of 5 models (SGD log, SGD huber, PassiveAggressive, ComplementNB, calibrated LinearSVC). Weights are proportional to cross-validation accuracy. |
| F5 | Incremental | Runs `partial_fit` on batches of 64 samples, recording accuracy and F1 per batch. |
| F6 | Refinement | Calibrates probabilities with sigmoid `CalibratedClassifierCV`. Compares ensemble, calibrated, and incremental candidates; picks the best by top-5 accuracy. |
| F7 | History | Appends the session (timestamp, rows, classes, mode, best model, metrics, risk distribution, top-10 KPIs, top char patterns) to `history/extract_v2_training_history.json`. |
| F8 | Pattern | Compares the last 10 sessions. Reports accuracy mean/std, top-5 mean, trend, recurring KPIs, and a stable MD5 signature. Marks `ready_for_prediction`. |
| F9 | Console | Opens the PyAutoGUI console: query CNPJ, capture PDF, view pattern, exit. |

---

## External Enrichment

Four APIs are queried per CNPJ, and the results are merged into one row.

| API | Contributes |
|-----|-------------|
| BrasilAPI | Legal name, trade name, CNAE, legal nature, state, city, status, size, share capital, partners, secondary CNAEs, email, phone |
| Minha Receita | Simples Nacional flag, MEI flag, email, phone |
| ReceitaWS | Registration status, primary activity |
| CNPJ.ws | Neighborhood, postal code, city |

Derived fields:

- `EXT_Region` — Brazilian region derived from the state
- `EXT_CompanyAge` — years since the opening date
- `EXT_CreditScore` — heuristic score 0 to 1 (status, age, share capital, partners)

Cache: `./cache_external/cnpj_cache.parquet`. Cap per run: `MAX_EXTERNAL_CNPJ` (default 500).

---

## Calculated KPIs

### Quality KPIs

Total rows, total columns, average completeness, duplicate rows, duplication rate, unique CNPJs, file count by type, unique header hashes.

### Business KPIs (row by row)

| KPI | Meaning |
|-----|---------|
| `KPI_Undervaluation` | Deviation of value per kg from the median |
| `KPI_HSDisparity` | Inconsistency between HS code and NCM description |
| `KPI_VolumeSurge` | Deviation of import value from the importer s average |
| `KPI_ETADeviation` | Absolute difference between planned and actual ETA |
| `KPI_Completeness` | Fraction of null fields in the row |
| `KPI_Outlier` | Outlier score from LocalOutlierFactor |
| `KPI_Ext_RegistrationConsistency` | Flags status other than "ACTIVE" |
| `KPI_Ext_CompanyAge` | Newer companies score higher on risk |
| `KPI_Ext_CreditScore` | Inverse of the heuristic credit score |
| `KPI_CharPattern` | Adherence to recurring char-ngrams |

---

## Risk Score

`RiskScore` is the weighted sum of the KPIs above, multiplied by 100 and clipped to [0, 100].

| Range | Level |
|-------|-------|
| 0 – 25 | `LOW` |
| 25 – 50 | `MEDIUM` |
| 50 – 75 | `HIGH` |
| 75 – 100 | `CRITICAL` |

Each row is tagged with `RiskLevel` and a `RiskColor` used in the XLSX export.

---

## Output Data

| File | Contents |
|------|----------|
| `output/kpi_by_column.csv` | One row per column: completeness, unique, null rate, cardinality |
| `output/extract_v2_audit_<ts>.parquet` | Full dataset with original + external + KPI + risk + prediction columns |
| `output/extract_v2_risk_<ts>.xlsx` | Top `TOP_RISK_XLSX` rows sorted by risk, each row colored by level |
| `output/extract_v2_kpis_<ts>.json` | Session summary: quality, risk distribution, ML metrics, top char patterns, pattern result |
| `history/extract_v2_training_history.json` | Accumulating array, one entry per run |
| `history/extract_v2_patterns.json` | Latest pattern detection result |
| `extract_v2_model.joblib` | Serialized winning model + preprocessor + encoder + metadata |

Load the model:

```python
import joblib

pkg = joblib.load("extract_v2_model.joblib")
model        = pkg["model"]
preprocessor = pkg["preprocessor"]
encoder      = pkg["label_encoder"]
metrics      = pkg["metrics"]
top_patterns = pkg["top_patterns"]
```

---

## Interactive Console

After the batch phase, the PyAutoGUI console opens with four actions.

| Action | Behavior |
|--------|----------|
| `Query CNPJ` | Prompts for each target column, returns the top-5 most likely CNPJs with probabilities |
| `Capture PDF` | Reads clipboard (Ctrl+C first), lists detected CNPJs and the 10 most frequent char-patterns |
| `View Pattern` | Shows the last 10 sessions with accuracy, top-5, best model, trend, recurring KPIs, and `ready_for_prediction` |
| `Exit` | Closes the program |

---

## Pattern Detection Between Sessions

After every run, the pattern module compares the last N sessions (default 10) and returns:

```json
{
  "sessions_analyzed": 10,
  "accuracy_mean": 0.8391,
  "accuracy_std": 0.0187,
  "top5_mean": 0.9703,
  "trend": "improving",
  "recurring_kpis": [
    "KPI_Undervaluation",
    "KPI_Ext_RegistrationConsistency",
    "KPI_CharPattern"
  ],
  "signature": "a3f9b12c7e04",
  "ready_for_prediction": true
}
```

- `recurring_kpis` — KPIs that fired in at least 60% of the sessions
- `signature` — MD5 hash of the recurring KPIs; if it stays the same across runs, the pattern is stable
- `trend` — `improving`, `stable`, or `degrading`
- `ready_for_prediction` — flips to `true` when the history has at least 10 sessions

---

## Project Structure

```
project/
├── data/                                  # Input (created by the user)
├── cache_external/
│   └── cnpj_cache.parquet                 # Enrichment cache
├── history/
│   ├── extract_v2_training_history.json   # Session history
│   └── extract_v2_patterns.json           # Most recent pattern
├── output/
│   ├── kpi_by_column.csv
│   ├── extract_v2_audit_<ts>.parquet
│   ├── extract_v2_risk_<ts>.xlsx
│   └── extract_v2_kpis_<ts>.json
├── extract_v2_model.joblib
└── extract_v2.py
```

---

## How to Feed the Pipeline

### Tabular files

Any XLSX, Parquet, or CSV must contain a column named `CNPJ`. It will be normalized to 14 digits (punctuation stripped, left-padded with zeros).

Natively recognized columns:

- Text: `LegalName`, `Product`, `NCMDescription`, `Supplier`, `Notes`
- Categorical: `CNAE`, `State`, `CountryOfOrigin`, `RegistrationStatus`, `HS_Code`, `Importer`
- Numeric: `ShareCapital`, `ImportValue`, `WeightKg`, `FreightValue`, `Quantity`
- Temporal: `ETA`, `ActualETA`

Extra columns still enter the feature vector if listed in the `COLUMNS_*` config.

### PDFs and TXT

PDFs and TXT files in `./data` are scanned for CNPJ patterns (raw 14 digits or formatted `00.000.000/0000-00`). Text is extracted with `pdfplumber`, then `pypdf`, then OCR with `pytesseract` as a fallback. Extracted rows enter the same pipeline as tabular data.

### Real-time capture

Use the `Capture PDF` action in the console to copy (Ctrl+C) content from an open PDF and paste it into the clipboard. Extract V.2 reads the clipboard and extracts CNPJs and char-patterns on the fly.

---

## Troubleshooting

### PyAutoGUI fails on a headless server

Run under a virtual display:

```bash
xvfb-run -a python extract_v2.py
```

### Tesseract not found

Install the Tesseract binary (see Installation). Without it, scanned PDFs are skipped; PDFs with a text layer still work.

### Rate limiting on public APIs

Extract V.2 sleeps 0.15 s between queries, but ReceitaWS has daily limits. Reduce `MAX_EXTERNAL_CNPJ` or use `Fast` mode.

### "Insufficient classes" warning

If fewer than 2 distinct CNPJs exist per class, stratified splitting fails. Extract V.2 automatically falls back to non-stratified splitting when `min(bincount(y)) < 2`.

### Accents on Windows

Internal strings are accent-free by design. If exported files show garbled characters, open them with `encoding="utf-8"`.

---

## Author

Extract V.2 — Technical Documentation and System Manual.

```
