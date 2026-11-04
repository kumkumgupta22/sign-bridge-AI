"""Train and evaluate the sign classifier.

Run after recording + splitting:   python -m ml.train
Writes ml/models/model.joblib, ml/models/metrics.json, ml/models/confusion_matrix.png
Only report the numbers in metrics.json - they come from the held-out TEST signers.
"""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_VERSION, seq_to_features

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "ml" / "models"
THRESHOLD = 0.70


def load_split(meta_dir: Path, raw_dir: Path):
    split_csv = meta_dir / "split.csv"
    if not split_csv.exists():
        raise FileNotFoundError("split.csv missing. Run: python -m vision.split_dataset")
    data = {s: ([], []) for s in ("train", "val", "test")}
    with open(split_csv, newline="") as f:
        for row in csv.DictReader(f):
            seq = np.load(raw_dir / row["label"] / row["file"])
            X, y = data[row["split"]]
            X.append(seq_to_features(seq))
            y.append(row["label"])
    for name, (X, y) in data.items():
        if not X:
            raise ValueError(f"No samples in the '{name}' split. Record more signers/sessions.")
    return {k: (np.stack(X), np.array(y)) for k, (X, y) in data.items()}


def candidates():
    return {
        "svm": make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=0)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=0),
        ),
    }


def save_confusion_png(cm, labels, path: Path):
    from matplotlib.figure import Figure  # imported here so tests/CI do not need matplotlib

    n = len(labels)
    fig = Figure(figsize=(max(6, n * 0.5), max(5, n * 0.5)))
    ax = fig.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test signers)")
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def run(meta_dir=META, raw_dir=RAW, out_dir=OUT, make_plot=True, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_split(Path(meta_dir), Path(raw_dir))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]

    best_name, best_model, best_f1, val_scores = None, None, -1.0, {}
    for name, model in candidates().items():
        model.fit(Xtr, ytr)
        score = f1_score(yva, model.predict(Xva), average="macro", zero_division=0)
        val_scores[name] = round(float(score), 4)
        if verbose:
            print(f"  {name:<14} validation macro-F1 = {score:.3f}")
        if score > best_f1:
            best_name, best_model, best_f1 = name, model, score

    labels = sorted(set(ytr) | set(yva) | set(yte))
    pred = best_model.predict(Xte)
    proba = best_model.predict_proba(Xte).max(axis=1)
    confident = proba >= THRESHOLD
    report = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=labels)

    metrics = {
        "model": best_name,
        "feature_version": FEATURE_VERSION,
        "validation_macro_f1": val_scores,
        "test_accuracy": round(float(accuracy_score(yte, pred)), 4),
        "test_macro_precision": round(report["macro avg"]["precision"], 4),
        "test_macro_recall": round(report["macro avg"]["recall"], 4),
        "test_macro_f1": round(report["macro avg"]["f1-score"], 4),
        "confidence_threshold": THRESHOLD,
        "coverage_at_threshold": round(float(confident.mean()), 4),
        "accuracy_when_confident": round(float(accuracy_score(yte[confident], pred[confident])), 4)
        if confident.any()
        else None,
        "per_class": {k: report[k] for k in labels},
        "samples": {"train": len(ytr), "val": len(yva), "test": len(yte)},
    }
    joblib.dump(
        {"model": best_model, "classes": list(best_model.classes_), "meta": metrics},
        out_dir / "model.joblib",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if make_plot:
        save_confusion_png(cm, labels, out_dir / "confusion_matrix.png")
    if verbose:
        print(f"\nBest: {best_name}")
        print(f"TEST accuracy {metrics['test_accuracy']:.3f}  macro-F1 {metrics['test_macro_f1']:.3f}")
        print(f"Saved to {out_dir}")
    return metrics


if __name__ == "__main__":
    run()
