# WebShell Detector

Machine-learning based webshell detection, built as a security research project.
Two approaches are implemented side by side so their trade-offs can be compared
directly:

| | v1 — `webshell_detector.py` | v2 — `webshell_detector_v2.py` |
|---|---|---|
| Method | One-class anomaly detection | Supervised binary classification |
| Pipeline | Doc2Vec → AutoEncoder (trained on benign only) | Hand-crafted security features + Doc2Vec → LightGBM |
| Score | Reconstruction MSE (higher = more suspicious) | P(webshell) (higher = more suspicious) |
| Best when | Malicious samples are scarce; exploring unknown threats | Labeled samples exist; maximizing accuracy |

Both share the same honest evaluation protocol: stratified train/validation/test
split, threshold selected from the validation ROC curve (Youden's J), and
precision / recall / F1 / ROC-AUC reported on a held-out test set.

## Features

- **Code-aware tokenization** — regex tokenizer that keeps `$variables`,
  identifiers and hex escapes instead of naive whitespace splitting
- **Hand-crafted security features (v2)** — 26 suspicious-API counters
  (`eval`, `assert`, `base64_decode`, `gzinflate`, `system`, …), tainted-sink
  detection (dangerous function consuming `$_GET`/`$_POST`/… directly),
  backtick execution, dynamic function calls, string entropy, long-string and
  base64-blob analysis
- **Variant generator** — semantics-preserving mutations (identifier
  obfuscation via word-boundary regex, comment/blank-line injection) for data
  augmentation; refuses to write back into the training set
- **Reproducible** — fixed global seeds, no hardcoded absolute paths
  (everything is CLI arguments)
- **SOC-friendly output** — scan results exported to Excel with a thresholded
  verdict column and a Top-K triage list

## Project structure

```
webshell-detector/
├── webshell_detector.py       # v1: one-class (Doc2Vec + AutoEncoder)
├── webshell_detector_v2.py    # v2: supervised (features + Doc2Vec + LightGBM)
├── requirements.txt
├── data/
│   ├── webshell/              # malicious samples (you provide; NOT committed)
│   ├── benign/                # benign samples, e.g. CMS / framework sources
│   └── variants/              # generated variants (kept out of training data)
├── check/                     # unknown files to scan
└── models/                    # trained artifacts (git-ignored)
```

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/Naomi9964/webshell-detector.git
cd webshell-detector
pip install -r requirements.txt
```

`lightgbm` is optional — v2 automatically falls back to sklearn's
`HistGradientBoostingClassifier` when it is not installed.

## Usage

### 1. Prepare data

```
data/webshell/   # webshell samples
data/benign/     # normal source code (CMS, frameworks, …)
```

> ⚠️ **Do not commit real webshell samples to a public repository.**
> `data/webshell/`, `data/benign/`, `check/` and `models/` are git-ignored by
> default. This project is for defensive security research and education.

### 2. (Optional) Generate variants for augmentation

```bash
python webshell_detector.py gen-variants \
  --source data/webshell --out data/variants --per-file 2
```

### 3. Train

```bash
# v1 — one-class
python webshell_detector.py train \
  --webshell-dir data/webshell --benign-dir data/benign --model-dir models

# v2 — supervised
python webshell_detector_v2.py train \
  --webshell-dir data/webshell --benign-dir data/benign --model-dir models_v2
```

Training prints the validation ROC-AUC, the selected threshold, test-set
metrics and a confusion matrix. v2 additionally prints the top-15 feature
importances, so you can see what the model actually learned.

### 4. Scan unknown files

```bash
python webshell_detector_v2.py scan \
  --scan-dir check --model-dir models_v2 --out scan_report.xlsx --top-k 10
```

## How it works

**v1 (one-class).** A Doc2Vec model embeds each file; an AutoEncoder is trained
*only* on benign embeddings, learning what "normal" looks like. Files with high
reconstruction error deviate from normality and are flagged. The decision
threshold comes from the validation ROC curve, not a hardcoded formula.

**v2 (supervised).** Each file is represented by hand-crafted security features
concatenated with its Doc2Vec embedding, and a LightGBM classifier predicts
P(webshell). The hybrid representation combines domain knowledge (suspicious
APIs, tainted sinks, entropy signals — robust and cheap) with learned semantics.

## Roadmap

- [ ] AST / opcode features for better obfuscation resistance
- [ ] YARA / keyword rule baseline for comparison
- [ ] Leave-one-family-out evaluation against real-world families
  (China Chopper, Behinder, Godzilla)
- [ ] Per-language models (PHP / JSP / ASPX)

## Disclaimer

Built for defensive security research and education. Do not use it to develop
or deploy malicious software, and handle real webshell samples with appropriate
care (isolated environment, no public distribution).

## License

MIT — see [LICENSE](LICENSE).
