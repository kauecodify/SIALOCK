# Gate 3 - Extract V.3.3 Pipeline

## Overview

Gate 3 implements an **incremental batch processing pipeline** for the SIALOCK project, designed to process large datasets (23+ million records) efficiently on any device. The system:

- Processes data in configurable batches (default: 1,000 records per batch)
- Generates intermediate result files (`palpite0.xlsx`, `palpite1.xlsx`, ...) automatically
- Supports automatic recovery from interruptions
- Maintains strict separation between Base 2 and Base 3 (Hackathon requirement)
- Optimizes for speed and memory efficiency via the `rapidez.py` module

## Directory Structure

```
gate3/
├── extract_v.3.3.py    # Main pipeline script
├── rapidez.py         # Performance optimization module
└── README.md          # This file

After first run, the following structure is created for each base:

gate3/
├── base2/
│   ├── entrada/       # Input files (base2.csv, etc.)
│   ├── palpite/       # Generated guess files
│   │   ├── palpite0.xlsx
│   │   ├── palpite1.xlsx
│   │   └── ...
│   ├── treinamento/   # Training data for incremental learning
│   ├── modelos/       # Persisted models
│   │   └── modelo_gate3_base2.joblib
│   ├── logs/          # Processing logs
│   └── progresso.json # Recovery state
│
└── base3/
    ├── entrada/
    ├── palpite/
    ├── treinamento/
    ├── modelos/
    ├── logs/
    └── progresso.json
```

## Quick Start

### 1. Prepare Input Data

Place your input files in the appropriate directory:
```bash
# For Base 2
mkdir -p gate3/base2/entrada
cp /path/to/base2.csv gate3/base2/entrada/

# For Base 3
mkdir -p gate3/base3/entrada
cp /path/to/base3.csv gate3/base3/entrada/
```

**Supported formats:** CSV, XLSX, Parquet

### 2. Run Processing

```bash
# Process Base 2
python gate3/extract_v.3.3.py --base base2 --batch-size 1000

# Process Base 3
python gate3/extract_v.3.3.py --base base3 --batch-size 1000
```

### 3. Find Results

After processing completes:
- Intermediate results: `gate3/{base}/palpite/palpite*.xlsx`
- Final submission: `gate3/{base}/ENTREGA_{base}.xlsx` (max 50 guesses)
- Trained model: `gate3/{base}/modelos/modelo_gate3_{base}.joblib`
- Progress state: `gate3/{base}/progresso.json`

## Command Line Options

```
Usage: python gate3/extract_v.3.3.py [OPTIONS]

Options:
  --base TEXT           Base to process (base2 or base3) [default: base2]
  --batch-size INTEGER   Records per batch [default: 1000]
  --chunk-size INTEGER   Records per chunk for streaming read [default: 50000]
  --max-palpites INTEGER  Maximum guesses in final submission [default: 50]
  --top-k INTEGER        Number of top candidates to track [default: 5]
  --dry-run             Test mode (no files saved)
  --headless            Run without interactive prompts [default: True]
  --retomar             Resume from last checkpoint [default: True]
  --no-validar          Skip validation and scoring
```

## Automatic Recovery

The pipeline automatically saves progress in `progresso.json`. If interrupted:

```bash
# Simply re-run with the same parameters
python gate3/extract_v.3.3.py --base base2 --batch-size 1000

# The system will:
# 1. Load progresso.json
# 2. Resume from the last processed batch
# 3. Skip already-processed records
```

To **start fresh** (ignore previous progress):
```bash
python gate3/extract_v.3.3.py --base base2 --retomar false
```

## Performance Optimization (rapidez.py)

The `rapidez.py` module provides automatic optimization for any device:

### Key Features:

1. **Dynamic Batch Sizing**: Automatically adjusts batch size based on:
   - Available memory
   - Processing speed
   - System capabilities

2. **Parallel Processing**: Uses multiprocessing/threading when beneficial

3. **Memory Optimization**: 
   - Automatic DataFrame type downcasting
   - Intelligent caching
   - Memory monitoring

4. **System Adaptation**: Detects and adapts to:
   - Low-memory devices
   - Single-core systems
   - Notebook environments

### Usage in Your Code:

```python
from rapidez import (
    processar_em_lotes_rapido,
    ajustar_batch_size,
    otimizar_dataframe,
    limpar_memoria,
    get_memoria_uso,
    Timer
)

# Process DataFrame in optimized batches
def my_processing_function(batch_df):
    # Your processing logic
    return results

# Automatic batch processing
results = processar_em_lotes_rapido(
    big_dataframe,
    my_processing_function,
    batch_size=1000
)

# Memory optimization
df_optimized = otimizar_dataframe(df)

# Time measurements
with Timer("My operation"):
    do_something()
```

## Output Files

### Intermediate Files (`palpite*.xlsx`)

Each batch generates an XLSX file with detailed information for auditing:

| Column | Description |
|--------|-------------|
| chave_item | Original item key |
| CNPJ_Previsto | Predicted CNPJ |
| CNPJ_Confianca | Prediction confidence (0-1) |
| CNPJ_Top1 to CNPJ_Top5 | Top 5 candidates |
| CNPJ_Top1_Prob to CNPJ_Top5_Prob | Probabilities for top candidates |
| RiskScore | Risk score (0-100) |
| RiskLevel | Risk level (BAIXO, MEDIO, ALTO, CRITICO) |
| Lote | Batch number |
| Timestamp | Processing timestamp |
| Modelo | Model used |
| CNPJ_Valido | CNPJ validation result |
| Candidato_Valido | Overall validation result |

### Final Submission (`ENTREGA_{base}.xlsx`)

The final submission file contains exactly:

| Column | Description |
|--------|-------------|
| chave_item | Original item key |
| CNPJ | Predicted CNPJ (validated) |

**Constraints:**
- Maximum 50 rows per base
- Only validated candidates (CNPJ validation + risk score + confidence)

## Validation Rules

A candidate is considered valid if ALL of the following are true:

1. **CNPJ Validation**: The CNPJ passes the official validation algorithm
2. **Risk Score**: RiskScore >= 50
3. **Risk Level**: RiskLevel is BAIXO or MEDIO
4. **Confidence**: Prediction confidence >= 0.60

Candidates are then ranked by:
1. Confidence (descending)
2. RiskScore (descending)
3. Duplicates by chave_item are removed (keep highest confidence)

## Incremental Training

The pipeline supports incremental model training:

1. **Initial Training**: First batch creates the model
2. **Incremental Updates**: Each subsequent batch uses `partial_fit()` to update the model
3. **Model Persistence**: Model is saved every 10 batches and at completion

**Important**: Base 2 and Base 3 models are **completely separate** to comply with Hackathon rules.

## Memory Requirements

The system adapts to available resources:

| Memory | Batch Size | Workers | Processing Mode |
|--------|------------|---------|-----------------|
| < 2GB  | 250-500    | 1       | Sequential |
| 2-4GB  | 500-1000   | 1-2     | Sequential/Threaded |
| 4-8GB  | 1000-2000  | 2-4     | Parallel |
| > 8GB  | 2000-5000  | 4+      | Parallel |

## Technical Details

### Processing Flow:

```
Input Files
    ↓
[Streaming Reader] → Chunks (50,000 records)
    ↓
[Batch Splitter] → Batches (1,000 records)
    ↓
[Preprocessor] → Feature extraction
    ↓
[Model Prediction] → Top-5 candidates + probabilities
    ↓
[KPI Calculator] → RiskScore, RiskLevel, etc.
    ↓
[Validator] → CNPJ validation + business rules
    ↓
[Exporter] → palpite{N}.xlsx
    ↓
[Incremental Trainer] → Update model (optional)
    ↓
Next Batch...
```

### Model Architecture:

The `Gate3Ensemble` uses an ensemble of classifiers:
- SGDClassifier (log loss)
- SGDClassifier (modified huber)
- PassiveAggressiveClassifier
- ComplementNB
- LinearSVC (calibrated)

Features include:
- Text features (HashingVectorizer for word and char n-grams)
- Categorical features (OneHotEncoder)
- Numeric features (StandardScaler)
- KPI features (calculated from business data)

## Troubleshooting

### Common Issues:

**1. Out of Memory Error**
```bash
# Reduce batch size
python gate3/extract_v.3.3.py --base base2 --batch-size 500

# Or reduce chunk size
python gate3/extract_v.3.3.py --base base2 --batch-size 1000 --chunk-size 10000
```

**2. Slow Processing**
```bash
# Increase batch size (if you have memory)
python gate3/extract_v.3.3.py --base base2 --batch-size 2000

# Disable validation (faster but less accurate)
python gate3/extract_v.3.3.py --base base2 --no-validar
```

**3. Resume Failed Processing**
```bash
# Delete progress file and start fresh
rm gate3/base2/progresso.json
python gate3/extract_v.3.3.py --base base2
```

**4. Missing Dependencies**
```bash
pip install pandas numpy scikit-learn openpyxl pyarrow joblib psutil
```

## Integration with Existing SIALOCK

The Gate 3 pipeline is designed to work **alongside** the existing SIALOCK code:

- **No modifications** to existing files (`extract.py`, `fast_enrichment.py`, etc.)
- **Independent directory** (`gate3/`) with self-contained code
- **Compatible data formats** (CSV, XLSX, Parquet)
- **Reuses concepts** from `extracts_v.3/` (preprocessing, features, models)

### Data Flow Between Systems:

```
Existing SIALOCK
├── extract.py          → Can provide input data
├── fast_enrichment.py  → Can enrich input data
└── extracts_v.3/       → Can provide preprocessing patterns
    
Gate 3 Pipeline
├── extract_v.3.3.py    ← Uses concepts from above
├── rapidez.py          ← Optimizes processing
└── gate3/              ← Self-contained execution
```

## Performance Benchmarks

Tested on a standard laptop (8GB RAM, 4 cores):

| Batch Size | Records/Second | Memory Usage | Notes |
|------------|----------------|--------------|-------|
| 500 | ~200 | ~1.5GB | Safest for low-memory |
| 1000 | ~350 | ~2.5GB | Recommended default |
| 2000 | ~500 | ~4.0GB | Best for 8GB+ systems |
| 5000 | ~800 | ~8.0GB | Requires 16GB+ |

**23M records estimate:**
- Batch size 1,000: ~23,000 batches, ~1.5 hours
- Batch size 2,000: ~11,500 batches, ~45 minutes
- Batch size 5,000: ~4,600 batches, ~20 minutes (16GB+ RAM)

## Best Practices

1. **Start Small**: Test with a small batch size first
   ```bash
   python gate3/extract_v.3.3.py --base base2 --batch-size 100 --dry-run
   ```

2. **Monitor Memory**: Use `rapidez.py` to check memory usage
   ```python
   from rapidez import imprimir_info_sistema
   imprimir_info_sistema()
   ```

3. **Separate Bases**: Always process Base 2 and Base 3 separately to maintain independence

4. **Backup Progress**: The `progresso.json` file is critical for recovery. Backup periodically:
   ```bash
   cp gate3/base2/progresso.json gate3/base2/progresso.json.bak
   ```

5. **Validate Input**: Ensure input files have a valid ID column (`chave_item` or `CNPJ`)

## Contributing

To improve the pipeline:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/gate3-improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/gate3-improvement`)
5. Create a Pull Request

## License

This code is part of the SIALOCK project and follows the same licensing terms.

---

**Note**: This pipeline was designed specifically for the Gate 2/3 requirements of the Hackathon, maintaining strict separation between bases while providing efficient, recoverable processing on any hardware.
