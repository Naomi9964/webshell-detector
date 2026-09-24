#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# webshell_detector_v2.py — WebShell detection v2: supervised binary classification.
#   Features = hand-crafted security features (suspicious APIs, tainted sinks,
#   string entropy, encoded blobs, ...) + Doc2Vec embeddings.
#   Classifier = LightGBM (falls back to sklearn's HistGradientBoostingClassifier
#   when lightgbm is not installed).
# See README.md for usage and methodology.

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
from collections import Counter
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

TOKEN_RE = re.compile(
    r"\$[A-Za-z_][A-Za-z0-9_]*"
    r"|[A-Za-z_][A-Za-z0-9_]*"
    r"|\\x[0-9A-Fa-f]{2}"
    r"|[^\sA-Za-z0-9_$]+"
)

# Suspicious API -> weight (used only for the weighted total; each API also
# gets its own count feature).
SUSPICIOUS_APIS = {
    # code execution
    "eval": 3, "assert": 3, "create_function": 3, "pcntl_exec": 3,
    "system": 2, "exec": 2, "passthru": 2, "shell_exec": 2,
    "popen": 2, "proc_open": 2, "call_user_func": 2, "call_user_func_array": 2,
    # decoding / deobfuscation
    "base64_decode": 2, "gzinflate": 2, "gzuncompress": 2, "str_rot13": 2,
    "hex2bin": 1, "urldecode": 1, "rawurldecode": 1,
    # file / network (common webshell behavior)
    "move_uploaded_file": 1, "file_put_contents": 1, "fwrite": 1,
    "curl_exec": 1, "fsockopen": 1, "socket_create": 1,
}

STRING_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
BLOB_RE = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")  # long base64-style blob


# ----------------------------------------------------------------------------
# Utilities (same file handling as v1)
# ----------------------------------------------------------------------------
def tokenize_code(text: str) -> list:
    return TOKEN_RE.findall(text)


def read_text_safe(path: Path, max_bytes: int = 2_000_000):
    try:
        data = path.read_bytes()[:max_bytes]
    except OSError:
        return None
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="ignore")


def collect_files(folder, exts=CODE_EXTS) -> list:
    folder = Path(folder)
    out = []
    for root, _, files in os.walk(folder):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in exts:
                out.append(p)
    return sorted(out)


def build_tagged(files, ws_dir, be_dir):
    """Return (tagged_docs, raws); raws[i] = (path, text), aligned with docs."""
    docs, raws = [], []
    for p in files:
        text = read_text_safe(p)
        toks = tokenize_code(text) if text is not None else []
        if not toks:
            continue
        try:
            rel = p.relative_to(ws_dir)
            prefix = "ws"
        except ValueError:
            rel = p.relative_to(be_dir)
            prefix = "be"
        docs.append(TaggedDocument(words=toks, tags=[f"{prefix}/{rel.as_posix()}"]))
        raws.append((p, text))
    return docs, raws


# ----------------------------------------------------------------------------
# Hand-crafted security features
# ----------------------------------------------------------------------------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def extract_features(text: str) -> dict:
    """Extract hand-crafted security features from one file.
    Returns a dict whose keys follow FEATURE_NAMES order."""
    f = {}
    low = text.lower()
    f["n_chars"] = len(text)
    f["n_lines"] = text.count("\n") + 1

    # Suspicious API occurrence counts (PHP function names are
    # case-insensitive, hence matched on the lowered text).
    for api in SUSPICIOUS_APIS:
        f[f"api_{api}"] = len(re.findall(r"\b" + re.escape(api) + r"\b", low))
    f["api_suspicious_weighted"] = sum(
        f[f"api_{a}"] * w for a, w in SUSPICIOUS_APIS.items())

    # External input (superglobals)
    for sg in ("get", "post", "cookie", "request", "files"):
        f[f"sg_{sg}"] = low.count(f"$_{sg}")

    # Tainted sink: dangerous function consuming external input directly —
    # the classic webshell pattern.
    f["tainted_sink"] = len(re.findall(
        r"(eval|assert|system|exec|passthru|shell_exec|popen|proc_open|"
        r"call_user_func)\s*\([^)]*\$_(get|post|request|cookie)", low))
    f["backtick"] = text.count("`")                      # `cmd` execution
    f["var_func_call"] = len(re.findall(r"\$\w+\s*\(", text))  # $fn(...)
    f["var_variable"] = len(re.findall(r"\$\$", text))        # $$x
    f["preg_e"] = len(re.findall(r"preg_replace\s*\([^)]*/e", low))
    f["short_tags"] = len(re.findall(r"<\?(?!php)", low))     # <? / <?= short tags

    # String analysis: obfuscated / packed webshells tend to carry long,
    # high-entropy strings.
    lits = [m.group(0)[1:-1] for m in STRING_RE.finditer(text)]
    entropies = [shannon_entropy(s) for s in lits if s]
    f["str_max_entropy"] = max(entropies) if entropies else 0.0
    f["str_mean_entropy"] = float(np.mean(entropies)) if entropies else 0.0
    f["str_longest"] = max((len(s) for s in lits), default=0)
    f["str_n_long200"] = sum(1 for s in lits if len(s) > 200)
    blob_chars = sum(len(m.group(0)) for m in BLOB_RE.finditer(text))
    f["blob_ratio"] = blob_chars / max(len(text), 1)
    f["n_strings"] = len(lits)
    return f


def _feature_names():
    names = ["n_chars", "n_lines"]
    names += [f"api_{a}" for a in SUSPICIOUS_APIS]
    names += ["api_suspicious_weighted"]
    names += ["sg_get", "sg_post", "sg_cookie", "sg_request", "sg_files"]
    names += ["tainted_sink", "backtick", "var_func_call",
              "var_variable", "preg_e", "short_tags"]
    names += ["str_max_entropy", "str_mean_entropy", "str_longest",
              "str_n_long200", "blob_ratio", "n_strings"]
    return names


FEATURE_NAMES = _feature_names()


def all_column_names(vector_size: int) -> list:
    return FEATURE_NAMES + [f"emb_{i}" for i in range(vector_size)]


def matrix_for(d2v_model, items, tags=None):
    """Assemble the feature matrix. items = [(path, text)]. When tags are
    given, training vectors come from model.dv[tag]; otherwise unseen files
    are embedded with infer_vector. Returns (X, kept_paths)."""
    rows, kept = [], []
    for (p, text), tag in zip(items, tags if tags is not None else [None] * len(items)):
        hand = extract_features(text)
        hand_vec = np.array([hand[n] for n in FEATURE_NAMES], dtype=np.float32)
        if tag is not None:
            emb = np.asarray(d2v_model.dv[tag], dtype=np.float32)
        else:
            toks = tokenize_code(text)
            if not toks:
                continue
            emb = np.asarray(d2v_model.infer_vector(toks), dtype=np.float32)
        rows.append(np.concatenate([hand_vec, emb]))
        kept.append(p)
    return np.array(rows, dtype=np.float32), kept


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


def build_classifier():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                  random_state=SEED, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("[INFO] lightgbm not installed, "
              "falling back to sklearn HistGradientBoostingClassifier")
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              random_state=SEED)


def pick_threshold_roc(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best = int(np.argmax(tpr - fpr))  # Youden's J
    return float(thresholds[best]), float(roc_auc_score(y_true, scores))


def read_labeled_items(files):
    """Read files and drop empty-token ones. Returns [(path, text)]."""
    items = []
    for p in files:
        text = read_text_safe(p)
        if text is not None and tokenize_code(text):
            items.append((p, text))
    return items


# ----------------------------------------------------------------------------
# train: split -> Doc2Vec -> feature matrix -> LightGBM -> ROC threshold -> test
# ----------------------------------------------------------------------------
def cmd_train(args):
    ws_dir, be_dir, model_dir = Path(args.webshell_dir), Path(args.benign_dir), Path(args.model_dir)

    ws_files = collect_files(ws_dir)
    be_files = collect_files(be_dir)
    if not ws_files:
        sys.exit(f"[ERROR] No usable samples in {ws_dir}.")
    if not be_files:
        sys.exit(f"[ERROR] No usable samples in {be_dir}. "
                 "Supervised training needs a benign set.")

    files = ws_files + be_files
    labels = [1] * len(ws_files) + [0] * len(be_files)  # 1 = webshell
    train_f, tmp_f, _, tmp_y = train_test_split(
        files, labels, test_size=0.40, random_state=SEED, stratify=labels)
    val_f, test_f, _, _ = train_test_split(
        tmp_f, tmp_y, test_size=0.50, random_state=SEED, stratify=tmp_y)
    print(f"Split: train={len(train_f)}  val={len(val_f)}  test={len(test_f)}")
    ws_set = {p.resolve() for p in ws_files}
    label_of = lambda ps: [1 if p.resolve() in ws_set else 0 for p in ps]

    # Train Doc2Vec on the train split only.
    train_docs, train_raws = build_tagged(train_f, ws_dir, be_dir)
    d2v = train_doc2vec(train_docs, vector_size=args.vector_size, window=args.window,
                        min_count=args.min_count, epochs=args.d2v_epochs)
    print(f"Doc2Vec trained ({len(train_docs)} documents, dim={args.vector_size})")

    # Feature matrix: hand-crafted features + Doc2Vec vectors (dv[tag] for train).
    X_train, kept_train = matrix_for(d2v, train_raws, [d.tags[0] for d in train_docs])
    y_train = label_of(kept_train)
    clf = build_classifier()
    clf.fit(X_train, y_train)
    print(f"Classifier trained ({X_train.shape[0]} samples x {X_train.shape[1]} features)")

    # Validation: pick the threshold from the ROC curve.
    X_val, kept_val = matrix_for(d2v, read_labeled_items(val_f))
    y_val = label_of(kept_val)
    proba_val = clf.predict_proba(X_val)[:, 1]
    threshold, auc_val = pick_threshold_roc(y_val, proba_val)
    print(f"Validation ROC-AUC={auc_val:.4f}, threshold={threshold:.4f}")
    if auc_val < 0.6:
        print("[WARN] Low validation AUC: check sample size, add hard negatives "
              "to the benign set, or revisit the features.")

    # Test: evaluate on fully unseen data.
    X_test, kept_test = matrix_for(d2v, read_labeled_items(test_f))
    y_test = label_of(kept_test)
    proba_test = clf.predict_proba(X_test)[:, 1]
    pred = (proba_test >= threshold).astype(int)
    metrics = {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "threshold": threshold,
    }
    print("\nTest-set evaluation:")
    for k, v in metrics.items():
        print(f"  {k:<10}= {v:.4f}")
    print("  confusion_matrix [[TN FP] [FN TP]] =")
    print(f"  {confusion_matrix(y_test, pred).tolist()}")

    # Feature importance: see what the model actually learned
    # (hand-crafted features vs. embeddings).
    col_names = all_column_names(args.vector_size)
    try:
        importances = np.asarray(clf.feature_importances_, dtype=float)
    except AttributeError:
        # sklearn's HGB has no feature_importances_; use permutation importance.
        from sklearn.inspection import permutation_importance
        r = permutation_importance(clf, X_val, y_val, n_repeats=5, random_state=SEED)
        importances = r.importances_mean
    top = sorted(zip(col_names, importances), key=lambda x: -x[1])[:15]
    print("\nTop 15 features by importance:")
    for name, imp in top:
        print(f"  {name:<28} {imp:.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    d2v.save(str(model_dir / "doc2vec.model"))
    with open(model_dir / "classifier.pkl", "wb") as fh:
        pickle.dump(clf, fh)
    config = {
        "threshold": threshold,
        "vector_size": args.vector_size,
        "feature_names": col_names,
        "seed": SEED,
        "metrics_test": {k: v for k, v in metrics.items() if k != "threshold"},
        "note": "score = P(webshell); score >= threshold => suspicious.",
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nModels saved to {model_dir}/ "
          "(doc2vec.model, classifier.pkl, config.json)")


# ----------------------------------------------------------------------------
# scan: score unknown files; higher P(webshell) = more suspicious
# ----------------------------------------------------------------------------
def cmd_scan(args):
    model_dir = Path(args.model_dir)
    d2v = Doc2Vec.load(str(model_dir / "doc2vec.model"))
    with open(model_dir / "classifier.pkl", "rb") as fh:
        clf = pickle.load(fh)
    threshold = float(json.loads(
        (model_dir / "config.json").read_text(encoding="utf-8"))["threshold"])

    scan_dir = Path(args.scan_dir)
    files = collect_files(scan_dir)
    if not files:
        sys.exit(f"[ERROR] No scannable source files in {scan_dir}.")
    X, kept = matrix_for(d2v, read_labeled_items(files))
    proba = clf.predict_proba(X)[:, 1]

    df = pd.DataFrame({
        "file": [p.relative_to(scan_dir).as_posix() for p in kept],
        "p_webshell": np.round(proba, 4),
        "verdict": np.where(proba >= threshold, "suspicious", "likely_benign"),
    }).sort_values("p_webshell", ascending=False).reset_index(drop=True)

    out = Path(args.out)
    df.to_excel(out, index=False)

    k = min(args.top_k, len(df))
    print(f"\nTop {k} most suspicious (highest P(webshell)):")
    for _, r in df.head(k).iterrows():
        print(f"  {r['file']:<50} p={r['p_webshell']:.4f}  {r['verdict']}")
    n_sus = int((df["verdict"] == "suspicious").sum())
    print(f"\nReport written to {out} "
          f"({len(df)} files, {n_sus} suspicious / threshold={threshold:.4f})")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WebShell detection v2: supervised binary classification")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Train (hand-crafted features + Doc2Vec + LightGBM)")
    p.add_argument("--webshell-dir", required=True)
    p.add_argument("--benign-dir", required=True)
    p.add_argument("--model-dir", default="models_v2")
    p.add_argument("--vector-size", type=int, default=100)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--d2v-epochs", type=int, default=40)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("scan", help="Scan unknown files and write an Excel report")
    p.add_argument("--scan-dir", required=True)
    p.add_argument("--model-dir", default="models_v2")
    p.add_argument("--out", default="scan_report_v2.xlsx")
    p.add_argument("--top-k", type=int, default=10)
    p.set_defaults(func=cmd_scan)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
