# lang-translator — English → zanX

Lightweight research translator: **BERT-Mini** English encoder + **custom decoder** trained on your zanX corpus. Exports to **ONNX v2** for CPU inference on Windows (8 GB RAM).

## Development steps

Run from the project root. Use `configs/dev.yaml` for quick runs on sample data.

### Step 0 — Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 1 — Download encoder (one-time)

```powershell
python -m src.data.download_encoder
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
```

Use **Python 3.10–3.12** for best PyTorch compatibility on Windows.

### Step 2 — Prepare data

```powershell
python -m src.data.prepare_corpus --input data/raw/sample_en_zanx_pairs.tsv
python -m src.data.build_zanx_tokenizer --input data/processed/train.jsonl
```

Replace `sample_en_zanx_pairs.tsv` with your own TSV when ready.

### Step 3 — Train

```powershell
python -m src.train.train --config configs/dev.yaml
python -m src.train.evaluate --config configs/dev.yaml --split test
```

### Step 4 — Export ONNX v2

```powershell
python -m src.export.export_onnx --config configs/dev.yaml
```

### Step 5 — Inference

```powershell
python -m src.infer.onnx_runtime_infer --text "Hello world"
```

## Docs

| Document | Description |
|----------|-------------|
| [docs/documentation.md](docs/documentation.md) | **Detailed `src/` code reference** — every file explained |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design |
| [docs/PREREQUISITES.md](docs/PREREQUISITES.md) | Setup & encoder options |
| [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | Full training workflow |

## Project layout

```
src/
  data/      corpus prep, zanX tokenizer, dataset
  model/     encoder, decoder, translator
  train/     train.py, evaluate.py
  export/    export_onnx.py
  infer/     onnx_runtime_infer.py
configs/     default.yaml, dev.yaml
data/raw/    your TSV pairs + lexicon
artifacts/   models, checkpoints, ONNX releases (gitignored)
```
