# EXTRACT V.3

**Overview:** EXTRACT V.3 is a comprehensive machine learning pipeline designed for data extraction, enrichment, risk scoring, and continuous training with neural network support. The application processes large datasets, enriches CNPJ records via external APIs, calculates business KPIs, trains ensemble and neural network models, and outputs risk assessments in structured formats.

---

## Table of Contents

1. Features
2. Architecture
3. System Requirements
4. Installation
5. Quick Start
6. Configuration
7. Usage Modes
8. Pipeline Stages
9. Output Files
10. API Integration
11. Model Architecture
12. Examples
13. Troubleshooting
14. Performance Considerations

---

## Features

### Core Capabilities

- **Multi-format Data Ingestion:** Supports CSV, Excel, Parquet, PDF, and TXT files
- **CNPJ Enrichment:** Integrates with 4 external CNPJ APIs for business data retrieval
- **Feature Engineering:** Automatic column classification and advanced feature extraction
- **Business KPIs:** Calculates 10+ risk indicators including undervaluation, volume surge, ETA deviation
- **Ensemble Classification:** 5-model ensemble combining SGD, Passive-Aggressive, Naive Bayes, and SVM classifiers
- **Neural Network Training:** MLPClassifier with SVD dimensionality reduction and warm-start learning
- **Continuous Learning:** Partial-fit support for streaming data and incremental model updates
- **Stream Enrichment:** Budget-controlled API calls during training with fallback to cache
- **Risk Scoring:** Weighted KPI aggregation producing risk levels (LOW, MEDIUM, HIGH, CRITICAL)
- **Structured Logging:** JSONL output for all pipeline events and metrics
- **Interactive Console:** Query interface for model inference and PDF capture
- **Model Persistence:** Joblib serialization of trained models and preprocessors

### UI Features

- Rich terminal output with colored sections and progress bars
- Live table rendering for epoch metrics
- Code viewer showing neural network training flow
- Interactive menu-driven mode or headless CLI mode
- Optional GUI alerts via PyAutoGUI

---

## Architecture

### High-Level Pipeline

```
[Input Data] --> [F0: Sampling & Detection] --> [F1: Enrichment & Features]
                                                    |
                                                    v
[Interactive Console] <-- [F2: Continuous Training] <-- [Ensemble Init]
                                |
                                v
                          [F2.5: Reservoir Enrichment]
                                |
                                v
                          [F3: Model Refinement]
                                |
                                v
                          [F3.5: Neural Training]
                                |
                                v
                          [F4: Result Generation]
                                |
                                v
                          [F5: Persistence & History]
```

### Component Hierarchy

**Data Layer:**
- Chunked iterators for large file processing
- Lazy data sampling with configurable batch sizes
- Automatic ID column detection and normalization

**Enrichment Layer:**
- Multi-source CNPJ API integration with retry logic
- Persistent parquet cache for API responses
- Heuristic credit score calculation
- Region mapping from state codes

**Feature Engineering:**
- Text: Word-level and character-level hashing vectorizers
- Categorical: OneHot encoding with frequency filtering
- Numerical: Median imputation and standard scaling
- KPI Features: Business logic calculations (undervaluation ratio, surge detection, etc.)
- Character N-grams: Pattern frequency scoring

**Model Layer:**
- NT1Ensemble: Custom ensemble with dynamic weight adjustment
- NeuralTrainer: MLP classifier with SVD reduction and live visualization
- ColumnTransformer: Multi-pipeline preprocessing with sparse output handling

**Evaluation & Storage:**
- Session history tracking with JSON serialization
- Pattern detection via KPI frequency analysis
- Risk color-coding for Excel export
- Parquet partitioning for scalable output

---

## System Requirements

### Python

- Python 3.9 or higher

### Core Dependencies

```
numpy >= 1.21
pandas >= 1.3
scikit-learn >= 1.0
joblib >= 1.1
requests >= 2.28
tenacity >= 8.0
openpyxl >= 3.6  # for XLSX export
pyarrow >= 10.0  # for Parquet I/O
```

### Optional Dependencies

```
pdfplumber >= 0.7    # PDF text extraction
pypdf >= 3.0         # PDF fallback reader
pytesseract >= 0.3   # Scanned PDF OCR
pillow >= 9.0        # Image processing
pyautogui >= 0.9.7   # GUI alerts
pyperclip >= 1.8.2   # Clipboard capture
```

---

## Installation

### 1. Clone or Download

```bash
git clone https://github.com/yourusername/extract-v3.git
cd extract-v3
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas scikit-learn joblib requests tenacity openpyxl pyarrow
```

### 4. Create Required Directories

```bash
mkdir -p dados saida cache_externo historico logs
```

---

## Quick Start

### Interactive Mode

```bash
python extract_v3.py
```

Follow the menu prompts to select:
1. Completo (full mode with API enrichment)
2. Rapido (fast mode, local data only)

### Headless Mode

```bash
python extract.py --headless --modo completo --epochs 3
```

### With Configuration File

```bash
python extract.py --headless --config config.json
```

---

## Configuration

### Command-Line Arguments

#### Core Options

```
--headless                  Run without interactive prompts
--modo {completo,rapido}    Execution mode (default: rapido)
--epochs N                  Training epochs (0 = infinite)
--dry-run                   Validate configuration without processing
```

#### Data & Paths

```
--dados PATH                Input data directory (default: ./dados)
--saida PATH                Output directory (default: ./saida)
--modelo PATH               Model save path (default: ./modelo_extract_v3.joblib)
--sample-rows N             Sampling size (default: 30000)
--max-classes N             Maximum classes to train (default: 20000)
--max-cnpj N                Max CNPJ API calls (default: 500)
--top-risco N               Top risk records in XLSX (default: 5000)
```

#### Column Selection

```
--texto COL1 COL2 ...       Text columns
--cat COL1 COL2 ...         Categorical columns
--num COL1 COL2 ...         Numerical columns
```

#### Neural Network

```
--no-neural                 Disable neural network training
--neural-epochs N           Neural training epochs (default: 15)
--neural-hidden 256,128     Hidden layer sizes
--neural-lr FLOAT           Learning rate (default: 0.001)
--neural-max-rows N         Max samples for neural (default: 8000)
--neural-svd N              SVD dimensions (default: 256)
--no-neural-viewer          Disable code visualization
--neural-delay FLOAT        Typing speed for viewer (default: 0.015)
```

#### Streaming Enrichment

```
--stream-enriquecido        Enable stream enrichment during training
--stream-budget N           API calls budget (default: 500)
--stream-so-cache           Use only cache, zero API calls
```

#### Other

```
--config FILE               Load configuration from JSON
--log-level {DEBUG,INFO,WARN,ERRO}
--no-jsonl                  Disable JSONL logging
--no-color                  Disable terminal colors
--sem-refinamento           Skip model refinement phase
--sem-xlsx                  Skip XLSX export
```

### Configuration File (config.json)

```json
{
  "headless": true,
  "modo": "completo",
  "epochs": 3,
  "sample_rows": 30000,
  "neural": true,
  "neural_epochs": 15,
  "neural_hidden": "256,128",
  "stream_enriquecido": false,
  "stream_budget": 500
}
```

---

## Usage Modes

### Mode 1: Interactive

Default mode with menu-driven interface:

```bash
python extract.py
```

Menu options:
- Completo: Full pipeline with API enrichment, continuous training, refinement, and neural network
- Rapido: Fast mode using only local data
- Config: View and modify settings

In interactive mode, after training, access the console loop:
- c: Query - infer predictions for custom records
- p: PDF capture - extract CNPJs from clipboard
- h: History - view pattern trends
- m: Metrics - inspect model weights
- s: Exit

### Mode 2: Headless CLI

Automated mode for scripts and CI/CD:

```bash
python extract.py --headless --modo completo --epochs 2 --neural-epochs 10
```

Features:
- No user prompts
- Full logging to JSONL
- Automatic folder creation
- Exit codes for error handling

### Mode 3: Dry Run

Validate configuration without processing:

```bash
python extract.py --dry-run
```

---

## Pipeline Stages

### F0: Sampling & Detection

**Purpose:** Load data and discover schema

**Operations:**
- Read files from input directory (CSV, Excel, Parquet, PDF, TXT)
- Sample up to `--sample-rows` records
- Auto-detect ID column (CNPJ, CPF, order number, etc.)
- Normalize ID format (remove non-alphanumeric, zero-pad)
- Classify columns by type (text, categorical, numerical, temporal)

**Output:**
- Sample DataFrame with normalized IDs
- Detected ID column name
- Column type classification

### F1: Enrichment & Features

**Purpose:** Add external data and engineer features

**Operations:**
1. **CNPJ Enrichment (optional):**
   - Query BrasilAPI, Minha Receita, ReceiptaWS, CNPJWS
   - Compute heuristic credit score
   - Map UF to geographic region
   - Calculate company age
   - Cache results to parquet

2. **Feature Generation:**
   - Text features: length, word count per column
   - Categorical features: frequency encoding
   - Character n-grams: bigrams to 5-grams in lowercase alphanumeric text
   - KPI features: business-logic indicators

3. **Preprocessor Setup:**
   - Pipeline 1: Word-level hashing (262144 features)
   - Pipeline 2: Character-level hashing (131072 features)
   - Pipeline 3: OneHot encoding for categoricals
   - Pipeline 4: StandardScaler for numericals
   - Output: Sparse matrix (262K columns typical)

**Output:**
- Enriched DataFrame with EXT_* columns
- Fitted ColumnTransformer
- Top 50 character n-grams pattern dictionary

### F2: Continuous Training

**Purpose:** Stream data and train ensemble model

**Operations:**
- Iterate over data chunks
- Apply preprocessing and KPI calculations
- Partial-fit ensemble with dynamic weight adjustment
- Accumulate samples for reservoir
- Optional: Stream enrichment with API budget

**Models (Partial-Fit):**
1. SGDClassifier (log loss)
2. SGDClassifier (modified Huber loss)
3. PassiveAggressiveClassifier
4. ComplementNB

**Weight Adjustment:**
- Initial: uniform (1.0)
- Per epoch: 70% momentum + 30% chunk accuracy
- Normalize to sum = 1.0

**Metrics Tracked:**
- Accuracy per chunk
- F1 score (macro average)
- Training time per epoch
- Chunk count and row count

### F2.5: Reservoir Enrichment

**Purpose:** Enrich accumulated samples with full CNPJ API queries

**Operations:**
- Take up to 30K samples from training
- Query all 4 CNPJ APIs for new CNPJs
- Add external features to reservoir
- Prepare for refinement stage

**Output:**
- Enriched reservoir with full feature set
- EXT_* columns populated

### F3: Model Refinement

**Purpose:** Compare streaming ensemble vs. refined ensemble

**Operations:**
1. Train new ensemble on enriched reservoir (80% train, 20% test)
2. Evaluate both models on test set
3. Compare via Top-5 accuracy or raw accuracy
4. Select best model

**Decision Logic:**
```
if refined_score > streaming_score:
    use refined model
else:
    keep streaming model
```

### F3.5: Neural Network Training

**Purpose:** Train MLP as alternative or supplement

**Operations:**
1. Apply SVD dimensionality reduction (256 dims typical)
2. Split into 80% train, 20% validation
3. Train MLP with warm-start for N epochs
4. Track per-epoch metrics (loss, train acc, val acc)
5. Compare best validation accuracy to ensemble

**Network Architecture:**
- Input: 256 (after SVD)
- Hidden: (256, 128) by default
- Output: num_classes
- Activation: ReLU
- Solver: Adam
- Learning rate: 0.001 (configurable)

**Metrics Tracked:**
- Loss (cross-entropy)
- Training accuracy
- Validation accuracy
- Training time

**Model Selection:**
```
if neural_best_val_acc > ensemble_metric:
    use neural model (via NeuralAdapter)
else:
    keep ensemble model
```

### F4: Result Generation

**Purpose:** Apply final model to all data and produce outputs

**Operations:**
1. Process all data chunks through final model
2. Generate predictions and confidence scores
3. Compute top-5 predicted classes and probabilities
4. Calculate prediction entropy and margin
5. Export to Parquet partitions
6. Generate XLSX with risk coloring

**Output Columns:**
- RES_Previsto: Top predicted class
- RES_Confianca: Max probability (0-1)
- RES_Top1 to RES_Top5: Ranking of predicted classes
- RES_Top1_p to RES_Top5_p: Associated probabilities
- RES_Entropia: Shannon entropy of prediction distribution
- RES_Margem: Margin between top-2 predictions

### F5: Persistence & History

**Purpose:** Save model, metrics, and track patterns

**Operations:**
1. Save trained model, preprocessor, and class maps to joblib
2. Record session metadata (timestamp, metrics, KPIs)
3. Append to session history JSON
4. Detect patterns (trend, recurring KPIs)
5. Save pattern signature to file

**Saved Files:**
- modelo_extract_v3.joblib: Complete model bundle
- extract_v3_kpis_YYYYMMDD_HHMMSS.json: Session KPIs
- extract_v3_history.json: All-time session history
- extract_v3_patterns.json: Detected patterns

**Pattern Detection:**
- Trend: improving, stable, or degrading accuracy
- Recurring KPIs: most common risk factors
- Readiness: sufficient sessions for pattern confidence

---

## Output Files

### Directory Structure

```
saida/
  extract_v3_risco_YYYYMMDD_HHMMSS.xlsx    Top 5000 risk records (color-coded)
  extract_v3_kpis_YYYYMMDD_HHMMSS.json     Session summary (metrics, KPIs)
  partes/
    part_00000.parquet                      Chunk 0 with predictions
    part_00001.parquet                      Chunk 1 with predictions
    ...

historico/
  extract_v3_history.json                   All session history
  extract_v3_patterns.json                  Detected patterns

cache_externo/
  cnpj_cache.parquet                        CNPJ enrichment cache

logs/
  extract_v3_YYYYMMDD.jsonl                 Event log (one JSON per line)

modelo_extract_v3.joblib                    Trained model bundle
```

### XLSX Output

**Columns:**
- All original data columns
- All KPI columns (KPI_Subvalorizacao, KPI_DisparidadeHS, etc.)
- Risk columns (RiskScore 0-100, RiskLevel, RiskColor)
- Prediction columns (RES_Previsto, RES_Confianca, RES_Top1-5, etc.)

**Risk Level Color Coding:**
- BAIXO (0-25): Green (#C6EFCE)
- MEDIO (25-50): Yellow (#FFEB9C)
- ALTO (50-75): Light Red (#FFC7CE)
- CRITICO (75-100): Dark Red (#C00000)

### JSONL Log Format

Each line is a JSON event:

```json
{
  "ts": "2024-01-15T14:30:45.123",
  "app": "EXTRACT V.3",
  "ver": "3.1",
  "level": "INFO",
  "event": "chunk",
  "epoch": 1,
  "n": 5000,
  "acc": 0.8234,
  "f1": 0.7912
}
```

Event types: log, chunk, table_row, code_line, neural_epoch, etc.

---

## API Integration

### Supported APIs

#### 1. BrasilAPI

```
https://brasilapi.com.br/api/cnpj/v1/{cnpj}
```

Fields extracted:
- Razao Social (company name)
- Nome Fantasia (trading name)
- CNAE description
- Juridical nature
- UF, municipality
- Cadastral status
- Company size
- Capital
- Opening date
- Number of partners
- Secondary CNAEs
- Email, phone

#### 2. Minha Receita

```
https://minhareceita.org/{cnpj}
```

Fields:
- Simples registration
- MEI enrollment
- Email, phone (alternative)

#### 3. ReceiptaWS

```
https://receitaws.com.br/v1/cnpj/{cnpj}
```

Fields:
- Cadastral status
- Main activity

#### 4. CNPJWS

```
https://publica.cnpjws.com/cnpj/{cnpj}
```

Fields:
- Neighborhood
- ZIP code
- City

### API Management

- Retry logic: 2 attempts with exponential backoff
- Timeout: 15 seconds per request
- Rate limiting: 0.1s sleep between calls
- Cache: Persistent Parquet storage
- Stream budget: Configurable max calls during training

---

## Model Architecture

### NT1Ensemble

**Training Mode (Batch):**

5 models with cross-validated weights:

1. SGDClassifier (log loss) - early stopping, validation-based
2. SGDClassifier (modified Huber) - robust to outliers
3. PassiveAggressiveClassifier - online learning friendly
4. ComplementNB - handles imbalanced data
5. CalibratedSVM - probability calibration

**Partial-Fit Mode (Streaming):**

4 models without validation (too small per-chunk):

1. SGDClassifier (log loss)
2. SGDClassifier (modified Huber)
3. PassiveAggressiveClassifier
4. ComplementNB

**Prediction:**

```python
probs = np.zeros((n_samples, n_classes))
for model, weight in weights.items():
    probs += weight * model.predict_proba(X)
probs /= probs.sum(axis=1, keepdims=True)
prediction = argmax(probs)
```

### NeuralTrainer

**Data Preprocessing:**

1. Subsample to 8000 rows max (configurable)
2. SVD reduction to 256 dimensions (configurable)
3. 80/20 train-validation split

**Architecture:**

```
Input (256)
  |
Dense (256, ReLU)
  |
Dense (128, ReLU)
  |
Output (num_classes, softmax)
```

**Training:**

- Epochs: 15 (configurable)
- Batch size: auto (up to 128)
- Optimizer: Adam
- Loss: cross-entropy
- Warm-start: True (preserves weights between epochs)

**Monitoring:**

- Per-epoch metrics: loss, train accuracy, validation accuracy
- Best validation accuracy tracked
- Early stopping available if gap between train/val grows

### Feature Dimensionality

**Typical Output:**

- Word-level hashing: 262144
- Character-level hashing: 131072
- OneHot (categories): 1000-5000
- Numerical/KPI: 50-100

**Total:** 394K-401K features
**After SVD:** 256 features

---

## Risk Scoring

### KPI Calculation

Each KPI scaled to [0, 1]:

1. **Subvalorizacao** (20% weight)
   - Ratio: (median_value - record_value) / median_value
   - Detects underpriced imports

2. **DisparidadeHS** (12%)
   - Trailing zeros in HS code
   - Indicator of generic/incomplete classification

3. **SurgeVolume** (12%)
   - Z-score of importer volume
   - Threshold: 3 standard deviations

4. **DesvioETA** (8%)
   - Absolute difference: actual vs. planned arrival
   - Threshold: 7 days

5. **Completude** (4%)
   - 1 - fraction of missing values

6. **Outlier** (5%)
   - Local Outlier Factor score
   - Neighbors: 20, contamination: 10%

7. **ConsistenciaCadastral** (13%)
   - Inverse: 1 if cadastral status not "active"

8. **IdadeEmpresa** (8%)
   - Inverse: capped at 10 years
   - Newer companies = higher risk

9. **ScoreCredito** (10%)
   - Heuristic based on status, capital, partners
   - Range: [0, 1]

10. **PadraoChar** (8%)
    - Character n-gram matching
    - Detects anomalous text patterns

### Risk Aggregation

```python
risk_score = sum(weight * kpi for weight, kpi in zip(weights, kpis))
risk_score = clip(risk_score * 100, 0, 100)
risk_level = categorize(risk_score)  # BAIXO, MEDIO, ALTO, CRITICO
```

---

## Examples

### Example 1: Quick Local Analysis

```bash
python extract.py --headless --modo rapido --epochs 1 \
  --dados ./sample_data --saida ./results
```

### Example 2: Full Pipeline with Neural

```bash
python extract.py --headless --modo completo --epochs 3 \
  --neural --neural-epochs 20 --neural-hidden 512,256,128
```

### Example 3: Stream Enrichment with Budget

```bash
python extract.py --headless --modo completo --epochs 5 \
  --stream-enriquecido --stream-budget 1000
```

### Example 4: Custom Columns

```bash
python extract.py --headless --modo completo \
  --texto "Description" "Comments" \
  --cat "Status" "Region" \
  --num "Value" "Quantity"
```

### Example 5: Dry Run Validation

```bash
python extract.py --dry-run --config config.json
```

---

## Interactive Example

```bash
$ python extract.py
```

Menu:
```
1. Completo  -- enriquece APIs + treino continuo + refinamento + rede neural
2. Rapido    -- apenas dados locais (sem APIs externas)
3. Config    -- ajustar parametros
0. Cancelar
```

Choose `1` for full mode. After training completes:

```
O que deseja agora?
c. Consultar   -- inferir para um registro digitado
p. Capturar PDF / clipboard
h. Ver padrao / historico
m. Metricas do modelo atual
s. Sair
```

Example query (option c):
- Input Description: "Importacao de componentes eletronicos"
- Input Status: "Ativo"
- Output: Top-5 predicted classes with probabilities

---

## Troubleshooting

### Issue: No input files found

**Error:** "Nenhum arquivo em ./dados"

**Solution:**
- Ensure `dados/` directory exists
- Place CSV, Excel, Parquet, PDF, or TXT files in `dados/`
- Check file extensions are lowercase (.csv, .xlsx, .parquet)

### Issue: Cannot detect ID column

**Error:** "Nenhuma coluna ID."

**Solution:**
- Ensure at least one column with mostly 6-20 digit identifiers
- Rename column to CNPJ, order_id, or similar
- Use `--texto` or `--cat` flags to guide detection

### Issue: API timeouts during enrichment

**Error:** "Consultando CNPJs em 4 APIs..." hangs

**Solution:**
- Reduce `--max-cnpj` (e.g., 100 instead of 500)
- Enable cache-only mode: `--stream-so-cache`
- Check internet connectivity
- APIs may be rate-limited; retry after delay

### Issue: Memory error with large files

**Error:** "MemoryError" during XLSX export

**Solution:**
- Reduce `--sample-rows` (e.g., 10000)
- Split input files into smaller chunks
- Use `--sem-xlsx` to skip XLSX export
- Process in multiple sessions

### Issue: Neural network training fails

**Error:** "nenhuma classe valida encontrada"

**Solution:**
- Ensure at least 2 classes in training data
- Reduce `--neural-epochs` if timeout
- Check `--neural-max-rows` is not too small
- Disable with `--no-neural` if problematic

### Issue: Headless mode exits unexpectedly

**Error:** Exit code 1 without output

**Solution:**
- Enable logging: check `logs/extract_v3_*.jsonl`
- Run with `--dry-run` first to validate config
- Check `--dados` and `--saida` paths are writable
- Use interactive mode for debugging

---

## Performance Considerations

### Typical Runtimes

- **F0 (Sampling):** 1-5 seconds (depends on file I/O)
- **F1 (Features):** 5-15 seconds (preprocessor fit)
- **F2 (Training, 1 epoch):** 30-120 seconds (depends on data size)
- **F2.5 (Enrichment):** 2-10 minutes (API calls)
- **F3 (Refinement):** 10-30 seconds
- **F3.5 (Neural, 15 epochs):** 1-5 minutes
- **F4 (Predictions):** 10-30 seconds
- **Total:** 5-30 minutes for full pipeline (30K rows)

### Optimization Tips

1. **Sampling:** Increase `--sample-rows` for better model accuracy but slower training
2. **Epochs:** Balance between model convergence and runtime
3. **Neural:** Disable (`--no-neural`) if time is critical
4. **Refinement:** Skip (`--sem-refinamento`) for faster iteration
5. **XLSX:** Disable (`--sem-xlsx`) to skip export overhead
6. **Cache:** Pre-populate CNPJ cache to avoid API calls

### Scaling

For datasets > 100K rows:

```bash
python extract.py --headless --modo rapido --epochs 1 \
  --sample-rows 50000 --no-neural --sem-refinamento
```

For distributed processing:

1. Split input into chunks
2. Process in parallel with different seeds
3. Merge predictions manually

---

## Advanced Usage

### Custom Feature Selection

Inspect auto-classification and override:

```bash
python extract.py
# Interactive mode: Select custom columns
# Record selections, then use:
python extract.py --headless \
  --texto "col1" "col2" \
  --cat "col3" "col4" \
  --num "col5" "col6"
```

### Model Reuse

Load a trained model for inference:

```python
import joblib
bundle = joblib.load("modelo_extract_v3.joblib")
modelo = bundle["modelo"]
preproc = bundle["preprocessador"]
classes_map = bundle["classes_map"]

# Preprocess new data
X_new = preproc.transform(new_dataframe)

# Predict
predictions = modelo.predict(X_new)
probabilities = modelo.predict_proba(X_new)
```

### Session Chaining

Use history patterns to inform next session:

```python
import json
with open("historico/extract_v3_history.json") as f:
    history = json.load(f)
    
last_trend = history[-1].get("metricas", {}).get("accuracy")
if last_trend > 0.8:
    # Model converging, reduce epochs
    epochs = 1
else:
    epochs = 3
```

---

## License

This software is provided as-is for research and commercial use.

---

## Support & Contributing

For issues, feature requests, or contributions:

1. Check troubleshooting section
2. Review logs: `logs/extract_v3_*.jsonl`
3. Run `--dry-run` to validate setup
4. Report with full config and error output

---

## Changelog

### v3.1 (Current)

- Headless mode with CLI arguments
- Stream enrichment with API budgeting
- Neural trainer with visual code viewer
- JSONL event logging
- Session history and pattern detection
- Multi-source CNPJ API integration
- Risk scoring with 10 KPIs
- Character n-gram feature extraction
- Continuous learning with partial-fit
- Model refinement stage

### v3.0

- Initial release
- Ensemble classification
- KPI calculation
- XLSX output
- Interactive console

---

## References

### Scientific Foundations

- Ensemble Methods: Kuncheva & Whitaker (2003)
- SVD Reduction: Halko et al. (2011)
- Risk Scoring: Internal methodology
- Outlier Detection: Breunig et al. (2000)

### Tools Used

- scikit-learn: Machine learning
- pandas: Data manipulation
- numpy: Numerical computing
- requests: HTTP API calls
- openpyxl: Excel generation
- joblib: Model serialization

---

**Last Updated:** 2026-09-24

**Maintainer:** Extract V.3 Development Team
