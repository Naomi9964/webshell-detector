#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# webshell_detector.py — WebShell detection (Doc2Vec + AutoEncoder, one-class anomaly detection)
# See README.md for usage and methodology.

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Global settings
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CODE_EXTS = {".php", ".php5", ".phtml", ".jsp", ".jspx", ".asp", ".aspx",
             ".py", ".js", ".html", ".htm", ".txt", ".java", ".rb", ".pl"}

# Code-aware tokenization: keeps $variables, identifiers and hex escapes;
# remaining symbols are grouped into runs.
TOKEN_RE = re.compile(
    r"\$[A-Za-z_][A-Za-z0-9_]*"      # $variables (PHP)
    r"|[A-Za-z_][A-Za-z0-9_]*"       # identifiers / keywords
    r"|\\x[0-9A-Fa-f]{2}"            # hex escapes
    r"|[^\sA-Za-z0-9_$]+"            # operators / punctuation runs
)

# For variant generation: word-boundary substitution (\b keeps us from
# replacing substrings, e.g. "system" inside "filesystem").
VARIANT_REPLACEMENTS = [
    (re.compile(r"\bcmd\b"), ("_cmdX", "cm__d", "arg_val")),
    (re.compile(r"\bexec\b"), ("ex__ec", "runner", "___x")),
    (re.compile(r"\bsystem\b"), ("sys__", "shelling", "zzz_run")),
    (re.compile(r"\beval\b"), ("ev__al", "dyn_exec", "run_str")),
]

COMMENT_PREFIX = {".php": "//", ".php5": "//", ".phtml": "//",
                  ".jsp": "//", ".jspx": "//", ".aspx": "//",
                  ".asp": "'", ".py": "#", ".js": "//",
                  ".java": "//", ".rb": "#", ".pl": "#"}
PHP_LIKE = {".php", ".php5", ".phtml"}


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def tokenize_code(text: str) -> list:
    """Split source code into tokens. Better suited to programming languages
    than plain str.split()."""
    return TOKEN_RE.findall(text)


def read_text_safe(path: Path, max_bytes: int = 2_000_000):
    """Read a file as text. Returns None for binary files (contain null
    bytes) or on read errors."""
    try:
        data = path.read_bytes()[:max_bytes]
    except OSError:
        return None
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="ignore")


def collect_files(folder, exts=CODE_EXTS) -> list:
    """Recursively collect source files, filtered by extension. Model files
    are naturally excluded because they live in models/."""
    folder = Path(folder)
    out = []
    for root, _, files in os.walk(folder):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in exts:
                out.append(p)
    return sorted(out)


def build_tagged_documents(files, ws_dir, be_dir):
    """Build Doc2Vec training documents. Tags use the 'ws/<relpath>' /
    'be/<relpath>' scheme so files with the same basename never collide."""
    docs, skipped = [], []
    for p in files:
        text = read_text_safe(p)
        toks = tokenize_code(text) if text is not None else []
        if not toks:
            skipped.append(str(p))
            continue
        try:
            rel = p.relative_to(ws_dir)
            prefix = "ws"
        except ValueError:
            rel = p.relative_to(be_dir)
            prefix = "be"
        docs.append(TaggedDocument(words=toks, tags=[f"{prefix}/{rel.as_posix()}"]))
    return docs, skipped


def infer_matrix(d2v_model, files):
    """Run infer_vector on unseen files. Returns (vectors, kept_files)."""
    vecs, kept = [], []
    for p in files:
        text = read_text_safe(p)
        toks = tokenize_code(text) if text is not None else []
        if not toks:
            continue
        vecs.append(d2v_model.infer_vector(toks))
        kept.append(p)
    return np.array(vecs, dtype=np.float32), kept


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def train_doc2vec(tagged_docs, vector_size=100, window=5, min_count=2,
                  epochs=40, seed=SEED) -> Doc2Vec:
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count,
                    workers=os.cpu_count() or 4, epochs=epochs, seed=seed)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def doc_vectors(d2v_model: Doc2Vec, tagged_docs) -> np.ndarray:
    """Fetch training-document vectors straight from model.dv[tag].
    (infer_vector is meant for unseen documents.)"""
    return np.array([d2v_model.dv[d.tags[0]] for d in tagged_docs], dtype=np.float32)


def train_autoencoder(vectors, encoding_dim=64, epochs=100, batch_size=32, seed=SEED):
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    tf.random.set_seed(seed)
    input_dim = vectors.shape[1]
    inp = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation="relu")(inp)
    decoded = Dense(input_dim, activation="linear")(encoded)
    ae = Model(inp, decoded)
    ae.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    ae.fit(vectors, vectors, epochs=epochs, batch_size=batch_size,
           shuffle=True, verbose=0)
    return ae


def reconstruction_mse(autoencoder, vectors) -> np.ndarray:
    recon = autoencoder.predict(vectors, verbose=0)
    return np.mean(np.square(vectors - recon), axis=1)


def pick_threshold_roc(y_true, scores):
    """Pick the ROC operating point that maximizes Youden's J (tpr - fpr)."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best = int(np.argmax(tpr - fpr))
    return float(thresholds[best]), float(roc_auc_score(y_true, scores))


# ----------------------------------------------------------------------------
# Variant generator (writes to an isolated folder, never back into training data)
# ----------------------------------------------------------------------------
def mutate_lines(lines, rng: random.Random, suffix: str) -> list:
    comment = COMMENT_PREFIX.get(suffix, "//")
    out = []
    for line in lines:
        if rng.random() < 0.25:
            out.append(f"{comment} {rng.choice(['obfuscate', 'bypass', 'noop'])}\n")
        for pat, choices in VARIANT_REPLACEMENTS:
            if pat.search(line):
                line = pat.sub(rng.choice(choices), line)
        if suffix in PHP_LIKE and rng.random() < 0.10:
            out.append('eval(base64_decode("ZWNobyAnaGVsbG8nOw=="));\n')
        out.append(line)
    if rng.random() < 0.3 and out:
        out.insert(rng.randint(0, len(out)), "\n")
    return out


def cmd_gen_variants(args):
    source_dir, output_dir = Path(args.source), Path(args.out)
    src_r, out_r = source_dir.resolve(), output_dir.resolve()
    if out_r == src_r or src_r in out_r.parents:
        sys.exit("[ERROR] --out must not equal --source or sit inside it: "
                 "variants must never be written back into the training set.")
    rng = random.Random(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates = collect_files(source_dir)
    n = 0
    for path in templates:
        text = read_text_safe(path)
        if text is None:
            continue
        lines = text.splitlines(keepends=True)
        for i in range(args.per_file):
            mutated = mutate_lines(lines, rng, path.suffix.lower())
            out_name = f"{path.stem}_variant{i}{path.suffix}"  # keep original extension
            (output_dir / out_name).write_text("".join(mutated), encoding="utf-8")
            n += 1
    print(f"Generated {n} variants from {len(templates)} templates -> {output_dir}")


# ----------------------------------------------------------------------------
# train: split -> Doc2Vec -> AE (benign only) -> ROC threshold -> test evaluation
# ----------------------------------------------------------------------------
def cmd_train(args):
    ws_dir, be_dir, model_dir = Path(args.webshell_dir), Path(args.benign_dir), Path(args.model_dir)

    ws_files = collect_files(ws_dir)
    be_files = collect_files(be_dir)
    if not ws_files:
        sys.exit(f"[ERROR] No usable samples in {ws_dir} "
                 f"(extensions limited to {sorted(CODE_EXTS)}).")
    if not be_files:
        sys.exit(f"[ERROR] No usable samples in {be_dir}. One-class training "
                 "needs a benign set — add normal source code (e.g. CMS / "
                 "framework sources) and retry.")

    files = ws_files + be_files
    labels = [1] * len(ws_files) + [0] * len(be_files)  # 1 = webshell (anomaly class)
    train_f, tmp_f, _, tmp_y = train_test_split(
        files, labels, test_size=0.40, random_state=SEED, stratify=labels)
    val_f, test_f, val_y, test_y = train_test_split(
        tmp_f, tmp_y, test_size=0.50, random_state=SEED, stratify=tmp_y)
    print(f"Split: train={len(train_f)}  val={len(val_f)}  test={len(test_f)}")

    # Train Doc2Vec on the train split only — never peek at val / test.
    train_docs, skipped = build_tagged_documents(train_f, ws_dir, be_dir)
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} unreadable or empty files.")
    d2v = train_doc2vec(train_docs, vector_size=args.vector_size, window=args.window,
                        min_count=args.min_count, epochs=args.d2v_epochs)
    print(f"Doc2Vec trained ({len(train_docs)} documents, dim={args.vector_size})")

    # The AutoEncoder only learns the benign (normal) distribution.
    be_train_docs = [d for d in train_docs if d.tags[0].startswith("be/")]
    X_be = doc_vectors(d2v, be_train_docs)
    ae = train_autoencoder(X_be, encoding_dim=args.encoding_dim,
                           epochs=args.ae_epochs, batch_size=args.batch_size)
    print(f"AutoEncoder trained on {len(be_train_docs)} benign vectors only")

    ws_set = {p.resolve() for p in ws_files}

    def labeled_scores(file_list):
        X, kept = infer_matrix(d2v, file_list)
        y = [1 if p.resolve() in ws_set else 0 for p in kept]
        return reconstruction_mse(ae, X), y

    # Validation: pick the threshold from the ROC curve (score = MSE;
    # higher = more suspicious).
    val_mse, val_y_aligned = labeled_scores(val_f)
    threshold, auc_val = pick_threshold_roc(val_y_aligned, val_mse)
    print(f"Validation ROC-AUC={auc_val:.4f}, threshold (MSE)={threshold:.6f}")

    # Test: evaluate on fully unseen data.
    test_mse, test_y_aligned = labeled_scores(test_f)
    pred = (test_mse >= threshold).astype(int)
    metrics = {
        "precision": float(precision_score(test_y_aligned, pred, zero_division=0)),
        "recall": float(recall_score(test_y_aligned, pred, zero_division=0)),
        "f1": float(f1_score(test_y_aligned, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_y_aligned, test_mse)),
        "threshold": threshold,
    }
    print("\nTest-set evaluation (threshold came from validation, not tuned here):")
    for k, v in metrics.items():
        print(f"  {k:<10}= {v:.4f}")
    print("  confusion_matrix [[TN FP] [FN TP]] =")
    print(f"  {confusion_matrix(test_y_aligned, pred).tolist()}")

    model_dir.mkdir(parents=True, exist_ok=True)
    d2v.save(str(model_dir / "doc2vec.model"))
    ae.save(str(model_dir / "autoencoder.keras"))
    config = {
        "threshold": threshold,
        "vector_size": args.vector_size,
        "encoding_dim": args.encoding_dim,
        "seed": SEED,
        "metrics_test": {k: v for k, v in metrics.items() if k != "threshold"},
        "note": "anomaly score = reconstruction MSE; MSE >= threshold => suspicious.",
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nModels saved to {model_dir}/ "
          "(doc2vec.model, autoencoder.keras, config.json)")


# ----------------------------------------------------------------------------
# scan: score unknown files; higher MSE = more suspicious
# ----------------------------------------------------------------------------
def cmd_scan(args):
    from tensorflow.keras.models import load_model

    model_dir = Path(args.model_dir)
    d2v = Doc2Vec.load(str(model_dir / "doc2vec.model"))
    ae = load_model(str(model_dir / "autoencoder.keras"))
    threshold = float(json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["threshold"])

    scan_dir = Path(args.scan_dir)
    files = collect_files(scan_dir)
    if not files:
        sys.exit(f"[ERROR] No scannable source files in {scan_dir}.")
    X, kept = infer_matrix(d2v, files)
    mse = reconstruction_mse(ae, X)

    df = pd.DataFrame({
        "file": [p.relative_to(scan_dir).as_posix() for p in kept],
        "mse": np.round(mse, 6),
        "verdict": np.where(mse >= threshold, "suspicious", "likely_benign"),
    }).sort_values("mse", ascending=False).reset_index(drop=True)

    out = Path(args.out)
    df.to_excel(out, index=False)

    k = min(args.top_k, len(df))
    print(f"\nTop {k} most suspicious (highest MSE = least like a normal file):")
    for _, r in df.head(k).iterrows():
        print(f"  {r['file']:<50} mse={r['mse']:.6f}  {r['verdict']}")
    n_sus = int((df["verdict"] == "suspicious").sum())
    print(f"\nReport written to {out} "
          f"({len(df)} files, {n_sus} suspicious / threshold={threshold:.6f})")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WebShell detection: Doc2Vec + AutoEncoder (one-class)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gen-variants", help="Generate mutated samples (isolated output dir)")
    p.add_argument("--source", required=True, help="Template directory (training set)")
    p.add_argument("--out", required=True,
                   help="Output directory for variants (must differ from training set)")
    p.add_argument("--per-file", type=int, default=2, help="Variants per template")
    p.add_argument("--seed", type=int, default=SEED)
    p.set_defaults(func=cmd_gen_variants)

    p = sub.add_parser("train", help="Train Doc2Vec + AutoEncoder (with evaluation)")
    p.add_argument("--webshell-dir", required=True)
    p.add_argument("--benign-dir", required=True)
    p.add_argument("--model-dir", default="models")
    p.add_argument("--vector-size", type=int, default=100)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--d2v-epochs", type=int, default=40)
    p.add_argument("--encoding-dim", type=int, default=64)
    p.add_argument("--ae-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("scan", help="Scan unknown files and write an Excel report")
    p.add_argument("--scan-dir", required=True)
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out", default="scan_report.xlsx")
    p.add_argument("--top-k", type=int, default=10)
    p.set_defaults(func=cmd_scan)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
