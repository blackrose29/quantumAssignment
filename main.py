"""Hybrid quantum-classical banknote authentication - single end-to-end run.

    python main.py                 # default: 3 layers, 300 Adam steps, seed 42
    python main.py --seed 7 --layers 2 --steps 200

Steps performed
    1. load the UCI Banknote Authentication dataset
    2. stratified 80/20 train-test split
    3. scale the features (standardisation for classical models, angles for the circuit)
    4. train five classical baselines
    5. build the 4-qubit circuit and train the hybrid model with Qiskit Aer
    6. evaluate everything on the held-out test notes
    7. save tables (results/) and plots (figures/)

Typical runtime: about two minutes on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.classical_models import build_baselines, count_parameters
from src.data_preprocessing import AngleScaler, get_arrays, load_banknote, make_split
from src.evaluation import compute_metrics, model_outputs
from src.hybrid_model import HybridQuantumClassifier
from src.quantum_circuit import circuit_summary

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def run(seed: int, layers: int, steps: int, verbose: bool = True) -> dict:
    # 1-2. data and split --------------------------------------------------
    df = load_banknote()
    X, y = get_arrays(df)
    X_train, X_test, y_train, y_test = make_split(X, y, seed)
    if verbose:
        print(f"Dataset: {len(df)} notes, {int((y == 0).sum())} genuine / {int((y == 1).sum())} forged")
        print(f"Train: {len(y_train)}  Test: {len(y_test)}  (seed {seed})\n")

    rows, scores = [], {}

    # 4. classical baselines (each pipeline standardises the features itself)
    for name, model in build_baselines(seed).items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred, score = model_outputs(model, X_test)
        predict_time = time.perf_counter() - t0
        scores[name] = score
        rows.append({"model": name, **compute_metrics(y_test, y_pred, score),
                     "train_seconds": train_time, "predict_seconds": predict_time,
                     "parameters": count_parameters(name, model)})

    # 3 + 5. hybrid model: features -> angles -> circuit -> logistic layer --
    scaler = AngleScaler().fit(X_train)
    A_train, A_test = scaler.transform(X_train), scaler.transform(X_test)
    hybrid = HybridQuantumClassifier(n_layers=layers, steps=steps, seed=seed, verbose=verbose)
    if verbose:
        print(f"Training hybrid model: {circuit_summary(hybrid.circuit)}")
    hybrid.fit(A_train, y_train)
    t0 = time.perf_counter()
    prob = hybrid.predict_proba(A_test)
    predict_time = time.perf_counter() - t0
    name = f"Hybrid VQC ({config.N_QUBITS} qubits, {layers} layers)"
    scores[name] = prob
    rows.append({"model": name, **compute_metrics(y_test, (prob >= 0.5).astype(int), prob),
                 "train_seconds": hybrid.fit_seconds, "predict_seconds": predict_time,
                 "parameters": str(hybrid.n_parameters)})

    table = pd.DataFrame(rows).set_index("model")
    return {"table": table, "scores": scores, "y_test": y_test,
            "hybrid": hybrid, "hybrid_name": name}


def save_outputs(out: dict) -> None:
    config.FIGURE_DIR.mkdir(exist_ok=True)
    config.RESULT_DIR.mkdir(exist_ok=True)
    table, hybrid, name = out["table"], out["hybrid"], out["hybrid_name"]
    table.to_csv(config.RESULT_DIR / "main_run_metrics.csv")
    (config.RESULT_DIR / "main_run_history.json").write_text(json.dumps(hybrid.history, indent=1))

    # confusion matrix of the hybrid model
    row = table.loc[name]
    matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Hybrid VQC - test set")
    fig.tight_layout(); fig.savefig(config.FIGURE_DIR / "main_confusion_matrix.png", dpi=200); plt.close(fig)

    # training curve
    history = pd.DataFrame(hybrid.history)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(history["step"], history["loss"], color="tab:blue")
    ax.set_xlabel("Adam step"); ax.set_ylabel("Training loss (cross-entropy)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "main_training_loss.png", dpi=200); plt.close(fig)

    # accuracy / F1 comparison
    fig, ax = plt.subplots(figsize=(7, 3.6))
    (table[["accuracy", "f1"]] * 100).plot.bar(ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_ylim(90, 100.5); ax.set_ylabel("%"); ax.set_xlabel("")
    ax.set_xticklabels([t.get_text().replace(" (", "\n(") for t in ax.get_xticklabels()], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "main_model_comparison.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=config.SHOWCASE_SEED)
    parser.add_argument("--layers", type=int, default=config.N_LAYERS)
    parser.add_argument("--steps", type=int, default=config.TRAIN_STEPS)
    args = parser.parse_args()

    result = run(args.seed, args.layers, args.steps)
    save_outputs(result)

    shown = result["table"][METRIC_COLUMNS + ["train_seconds"]].copy()
    shown[METRIC_COLUMNS] = (shown[METRIC_COLUMNS] * 100).round(2)
    shown["train_seconds"] = shown["train_seconds"].round(3)
    print("\nTest-set results (percent):")
    print(shown.to_string())
    print(f"\nCircuit executions used for training: {result['hybrid'].history[-1]['circuits']:,}")
    print(f"Outputs written to {config.RESULT_DIR} and {config.FIGURE_DIR}")
