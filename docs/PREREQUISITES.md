# Prerequisites & Technical Reference

Everything needed to build, train, and export the English → zanX translator on a **Windows 10/11 machine with 8 GB RAM and CPU only**.

---

## 1. Hardware & OS

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10 (19045+) | Windows 11 |
| RAM | 8 GB | 16 GB (still train with 8 GB using small batch) |
| CPU | 4 cores | 8+ cores (AVX2 helps ONNX Runtime) |
| Disk | 2 GB free | 5 GB (venv, models, checkpoints) |
| GPU | Not required | Optional — not in scope for v1 |

---

## 2. Software Prerequisites

### 2.1 Core

| Tool | Version | Purpose |
|------|---------|---------|
| [Python](https://www.python.org/downloads/) | **3.10 – 3.12** | Training & export |
| [Git](https://git-scm.com/download/win) | latest | Version control |
| [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) | 2015–2022 | Required by PyTorch & ONNX Runtime on Windows |

### 2.2 Python Environment Setup

```powershell
# From project root
cd C:\Users\karpr\LanguageTranslator\lang-translator

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 `requirements.txt` (pin for reproducibility)

```text
# Core ML
torch>=2.1.0,<2.5.0          # CPU wheel from pytorch.org
transformers>=4.38.0,<5.0.0
tokenizers>=0.15.0
datasets>=2.18.0
huggingface_hub>=0.21.0
sentencepiece>=0.1.99        # if using SentencePiece for zanX

# Training utilities
pyyaml>=6.0
tqdm>=4.66.0
numpy>=1.24.0,<2.0.0
scikit-learn>=1.3.0

# Export & inference
onnx>=1.15.0
onnxruntime>=1.17.0
onnxscript>=0.1.0            # helps torch.onnx.export in PyTorch 2.x

# Optional: metrics & quant
sacrebleu>=2.4.0
onnxruntime-tools>=1.7.0     # dynamic quantization (optional)

# Dev
pytest>=8.0.0
```

**Install PyTorch CPU on Windows:**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Base Encoder Model (English Only, Offline)

You need **only** a small English encoder — no translation model. The default is **BERT-Mini** (~45 MB on disk).

### 3.1 Recommended default

| Asset | Hugging Face ID | Params | Disk | Use |
|-------|-----------------|--------|------|-----|
| **BERT-Mini (default)** | `prajjwal1/bert-mini` | 11M | ~45 MB | English encoder |
| Tokenizer | same repo | — | ~1 MB | English WordPiece |

```python
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    "artifacts/models/bert-mini", local_files_only=True
)
encoder = BertModel.from_pretrained(
    "artifacts/models/bert-mini", local_files_only=True
)
```

### 3.2 Alternative encoders (all work offline)

| Model | ID | Params | Disk | Notes |
|-------|-----|--------|------|-------|
| BERT-Tiny | `prajjwal1/bert-tiny` | 4.4M | ~17 MB | Dev/smoke tests only |
| TinyBERT | `huawei-noah/TinyBERT_4L_312D` | 14.5M | ~55 MB | Distilled; better English |
| MiniLM-L3 | `sentence-transformers/paraphrase-MiniLM-L3-v2` | 17M | ~70 MB | Includes pre-exported ONNX |
| DistilBERT | `distilbert-base-uncased` | 66M | ~260 MB | Heavier fallback |

### 3.3 One-time offline download (Windows)

Requires internet **once**. After this, set offline flags and work air-gapped.

```powershell
pip install huggingface_hub transformers

# Default encoder (recommended)
huggingface-cli download prajjwal1/bert-mini `
  --local-dir artifacts/models/bert-mini

# Optional: stronger tiny encoder
# huggingface-cli download huawei-noah/TinyBERT_4L_312D `
#   --local-dir artifacts/models/tinybert-4l
```

**Enable fully offline mode:**

```powershell
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
```

**Verify offline load:**

```powershell
python -c "from transformers import BertModel; BertModel.from_pretrained('artifacts/models/bert-mini', local_files_only=True); print('OK')"
```

### 3.4 What you are NOT downloading

| Skip | Reason |
|------|--------|
| MarianMT / OPUS-MT | Pretrained on massive web translation data |
| T5 / mBART / NLLB | Internet-scale multilingual corpora |
| LLMs (Phi, Llama) | Too heavy for 8 GB RAM |

---

## 4. Data Prerequisites (zanX)

Before training, prepare:

1. **Writing system** — Unicode code points, normalization rules (NFC)
2. **Lexicon** — zanX word ↔ English gloss, part of speech
3. **Parallel corpus** — TSV or JSONL:
   ```tsv
   en	source
   zanx	target
   Hello world.	<zanX translation>
   ```
4. **Splits** — 80% train / 10% val / 10% test (stratified if multi-domain)
5. **zanX tokenizer training file** — one zanX sentence per line (`data/raw/zanx_monolingual.txt`)

**Quality rules:**
- Consistent orthography across all pairs
- No mixed English in zanX targets (unless intentional)
- Document neologisms and morphology in lexicon

---

## 5. Technical Concepts

### 5.1 Why Encoder–Decoder?

- **Encoder (BERT-Mini):** maps English tokens → contextual vectors (understands syntax and semantics in English).
- **Decoder (custom):** autoregressively predicts zanX token by token conditioned on encoder output.
- Your research data trains **only the mapping** English context → zanX — not general web knowledge.

### 5.2 Tokenization

| Language | Method | Library |
|----------|--------|---------|
| English | WordPiece (fixed) | `BertTokenizer` (`prajjwal1/bert-mini`) |
| zanX | BPE or Word-level | `tokenizers` (Hugging Face) or `sentencepiece` |

For a **constructed research language** with finite vocabulary, **word-level** tokenization is often best (every zanX word = one token).

### 5.3 Training Objective

Standard causal language modeling on the decoder side:

```
Loss = CrossEntropy(predicted_zanx_tokens, ground_truth_zanx_tokens)
```

Teacher forcing during training; greedy or beam search at inference.

### 5.4 Freezing the encoder

```python
for param in encoder.parameters():
    param.requires_grad = False
```

Unfreeze only the last encoder layer if val loss plateaus and you have RAM headroom.

---

## 6. ONNX Export (v2) — Technical Notes

### 6.1 Opset & Runtime

| Item | Recommendation |
|------|----------------|
| ONNX opset | **17** (widely supported by ONNX Runtime 1.17+) |
| Provider | `CPUExecutionProvider` |
| Dynamic axes | `batch_size`, `sequence_length` |

### 6.2 Export Strategies

**Option A — Full graph (harder):** export entire encode + autoregressive loop (fixed `max_decode_steps`).

**Option B — Hybrid (recommended for research):**
- Export **encoder** → `encoder-v2.onnx`
- Export **single decoder step** → `decoder-step-v2.onnx`
- Run token loop in Python (simple, debuggable, paper-friendly)

**Option C — End-to-end greedy:** unroll decoder for `N` steps in TorchScript trace (larger graph, faster inference).

### 6.3 Validation Checklist

```powershell
python -c "import onnx; onnx.checker.check_model('artifacts/releases/v2/zanx-translator-v2.onnx')"
python src/export/validate_onnx.py --compare pytorch
```

Compare PyTorch vs ONNX outputs — max abs diff should be &lt; 1e-4 (FP32).

### 6.4 Optional Quantization (fit tighter RAM)

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model.onnx", "model-int8.onnx", weight_type=QuantType.QUInt8)
```

---

## 7. Memory Budget (8 GB RAM)

| Phase | Approx. peak RAM |
|-------|------------------|
| Load BERT-Mini (FP32) | ~300 MB |
| Training (batch=4, frozen encoder) | ~2–3 GB |
| Training (unfrozen encoder) | ~4–6 GB |
| ONNX inference | ~0.2–0.35 GB |

**Tips:** `batch_size=4`, `gradient_accumulation_steps=4`, `torch.set_num_threads(4)`, close browser during training.

---

## 8. Official Documentation Links

### PyTorch & Transformers
- PyTorch: https://pytorch.org/docs/stable/index.html
- Transformers: https://huggingface.co/docs/transformers/index
- BERT-Mini (default encoder): https://huggingface.co/prajjwal1/bert-mini
- Compact BERT paper: https://arxiv.org/abs/1908.08962
- TinyBERT (alternative): https://huggingface.co/huawei-noah/TinyBERT_4L_312D
- DistilBERT (heavier fallback): https://huggingface.co/distilbert-base-uncased

### ONNX
- ONNX spec: https://github.com/onnx/onnx/blob/main/docs/docsgen/source/introduction.md
- `torch.onnx.export`: https://pytorch.org/docs/stable/onnx.html
- ONNX Runtime: https://onnxruntime.ai/docs/
- Python API: https://onnxruntime.ai/docs/api/python/api_summary.html
- Quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

### Tokenization
- Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers/index
- SentencePiece: https://github.com/google/sentencepiece

### Seq2seq / NMT background (conceptual only — you are not using full Marian)
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Helsinki OPUS-MT (reference only): https://github.com/Helsinki-NLP/Opus-MT

### Windows-specific
- PyTorch on Windows: https://pytorch.org/get-started/locally/
- Long paths on Windows (if needed): https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation

---

## 9. Environment Variables

```powershell
# Optional performance tuning
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:TOKENIZERS_PARALLELISM = "false"

# Offline mode (after one-time download)
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"

# Reproducibility
$env:PYTHONHASHSEED = "42"
```

---

## 10. Pre-Flight Checklist

- [ ] Python 3.10+ installed, `python --version` works in PowerShell
- [ ] venv created and activated
- [ ] `pip install torch` (CPU) succeeds
- [ ] `python -c "import torch; print(torch.__version__)"` runs
- [ ] `python -c "from transformers import BertModel; BertModel.from_pretrained('artifacts/models/bert-mini', local_files_only=True)"` loads offline
- [ ] Encoder downloaded to `artifacts/models/bert-mini/` (or chosen alternative)
- [ ] `python -c "import onnxruntime; print(onnxruntime.__version__)"` runs
- [ ] `$env:TRANSFORMERS_OFFLINE = "1"` works after local download
- [ ] zanX parallel corpus prepared (min. 500 pairs)
- [ ] zanX lexicon documented
- [ ] `data/` and `artifacts/` in `.gitignore`

---

## 11. Glossary

| Term | Meaning |
|------|---------|
| **zanX** | Your constructed target language |
| **v2** | Second research release — ONNX bundle ready for inference |
| **Encoder** | BERT-Mini (`prajjwal1/bert-mini`) — reads English |
| **Decoder** | Custom model — writes zanX |
| **Teacher forcing** | Training with ground-truth previous tokens as decoder input |
| **ONNX** | Open Neural Network Exchange — portable inference format |
| **WordPiece** | Subword tokenizer used by BERT family |
