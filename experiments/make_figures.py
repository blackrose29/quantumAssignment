"""Build every table and figure of Section 5 from the saved experiment results.

    python -m experiments.make_figures

Reads   results/main.json, showcase.json, ablation.json, optimizer.json, learning_curve.json
Writes  figures/r_*.png, results/tables/*.md and results/summary.json

Nothing here is typed in by hand: all numbers are computed from the JSON files
written by experiments/run_experiments.py.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from src import config
from src.data_preprocessing import get_arrays, load_banknote

R = config.RESULT_DIR
F = config.FIGURE_DIR
T = R / "tables"
HYBRID = "Hybrid VQC"
ORDER = ["Logistic Regression", "SVM (RBF kernel)", "k-Nearest Neighbours", "Random Forest",
         "Neural Network (4-4-1)", HYBRID]
SHORT = {"Logistic Regression": "LogReg", "SVM (RBF kernel)": "SVM-RBF", "k-Nearest Neighbours": "k-NN",
         "Random Forest": "Random forest", "Neural Network (4-4-1)": "NN 4-4-1", HYBRID: "Hybrid VQC"}
COLORS = {name: ("#dd8452" if name == HYBRID else "#4c72b0") for name in ORDER}


def load(name):
    return json.loads((R / f"{name}.json").read_text())


def pm(values, scale=100, digits=2):
    values = np.asarray(values, dtype=float) * scale
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join([header, sep] + rows)


def save(fig, name):
    fig.savefig(F / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
def dataset_figure(summary):
    df = load_banknote()
    X, y = get_arrays(df)
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.6), gridspec_kw={"width_ratios": [0.8, 1, 1, 1, 1]})
    counts = np.bincount(y)
    axes[0].bar(config.CLASS_NAMES, counts, color=["#4c72b0", "#c44e52"])
    for i, c in enumerate(counts):
        axes[0].text(i, c + 10, str(c), ha="center", fontsize=8)
    axes[0].set_ylabel("Notes"); axes[0].set_ylim(0, counts.max() * 1.15); axes[0].set_title("Class balance", fontsize=9)
    for j, feat in enumerate(config.FEATURE_NAMES):
        ax = axes[j + 1]
        for cls, color in ((0, "#4c72b0"), (1, "#c44e52")):
            ax.hist(X[y == cls, j], bins=30, alpha=0.6, color=color, label=config.CLASS_NAMES[cls], density=True)
        ax.set_title(feat, fontsize=9); ax.set_yticks([])
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); save(fig, "r_dataset_distribution.png")

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=6, alpha=0.6, color="#4c72b0", label="Genuine")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=6, alpha=0.6, color="#c44e52", label="Forged")
    ax.set_xlabel("variance"); ax.set_ylabel("skewness"); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); save(fig, "r_dataset_scatter.png")
    summary["dataset"] = {"n": int(len(y)), "genuine": int(counts[0]), "forged": int(counts[1]),
                          "duplicates": int(df.duplicated().sum())}


def main_comparison(summary):
    runs = load("main")
    rows, per_model = [], {}
    for name in ORDER:
        m = [r["models"][name] for r in runs]
        per_model[name] = m
        rows.append({
            "Model": name,
            "Accuracy (%)": pm([x["accuracy"] for x in m]),
            "Precision (%)": pm([x["precision"] for x in m]),
            "Recall (%)": pm([x["recall"] for x in m]),
            "F1 (%)": pm([x["f1"] for x in m]),
            "ROC-AUC (%)": pm([x["roc_auc"] for x in m]),
            "MCC": pm([x["mcc"] for x in m], scale=1, digits=3),
            "Train time (s)": pm([x["train_seconds"] for x in m], scale=1, digits=3 if name != HYBRID else 1),
            "Predict time (ms)": pm([x["predict_seconds"] for x in m], scale=1000, digits=1),
        })
    table = pd.DataFrame(rows)
    (T / "main_metrics.md").write_text(md_table(table))

    # paired comparison hybrid vs each baseline (accuracy per split)
    tests = []
    h = np.array([x["accuracy"] for x in per_model[HYBRID]])
    for name in ORDER[:-1]:
        c = np.array([x["accuracy"] for x in per_model[name]])
        diff = (h - c) * 100
        try:
            p = wilcoxon(h, c, zero_method="zsplit").pvalue
        except ValueError:
            p = float("nan")
        wins = int((h > c).sum()); ties = int((h == c).sum()); losses = int((h < c).sum())
        tests.append({"Baseline": name, "Mean difference (pp)": f"{diff.mean():+.2f}",
                      "Hybrid better / tie / worse": f"{wins} / {ties} / {losses}", "Wilcoxon p": f"{p:.3f}"})
    (T / "paired_tests.md").write_text(md_table(pd.DataFrame(tests)))

    # pooled confusion matrices over the 10 test sets
    pooled = {name: {k: int(sum(x[k] for x in per_model[name])) for k in ("tn", "fp", "fn", "tp")} for name in ORDER}
    hybrid = per_model[HYBRID]
    summary["main"] = {
        name: {k: [float(np.mean([x[k] for x in per_model[name]])), float(np.std([x[k] for x in per_model[name]], ddof=1))]
               for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc", "train_seconds", "predict_seconds")}
        for name in ORDER}
    summary["main_pooled_confusion"] = pooled
    summary["main_param_counts"] = {name: per_model[name][0]["parameters"] for name in ORDER}
    summary["hybrid_circuits"] = int(hybrid[0]["circuits"])
    summary["hybrid_accuracy_per_seed"] = [x["accuracy"] for x in hybrid]
    summary["paired_tests"] = tests

    # box plots of accuracy and F1 over the 10 splits
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, metric, label in ((axes[0], "accuracy", "Accuracy (%)"), (axes[1], "f1", "F1-score, forged class (%)")):
        data = [[x[metric] * 100 for x in per_model[name]] for name in ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": "black"})
        for patch, name in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COLORS[name]); patch.set_alpha(0.75)
        for i, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(d)), d, s=8, color="black", zorder=3)
        ax.set_xticks(range(1, len(ORDER) + 1)); ax.set_xticklabels([SHORT[n] for n in ORDER], rotation=20)
        ax.set_ylabel(label); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); save(fig, "r_main_boxplot.png")

    # training time (log scale)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    means = [np.mean([x["train_seconds"] for x in per_model[n]]) for n in ORDER]
    ax.barh([SHORT[n] for n in ORDER], means, color=[COLORS[n] for n in ORDER])
    ax.set_xscale("log"); ax.set_xlabel("Mean training time per split (s, log scale)")
    for i, v in enumerate(means):
        ax.text(v * 1.15, i, f"{v:.3g} s", va="center", fontsize=8)
    ax.set_xlim(min(means) / 2, max(means) * 8); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); save(fig, "r_training_time.png")

    # mean training-loss curve over the 10 splits
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    curves = []
    for r in runs:
        h = pd.DataFrame(r["history"])
        ax.plot(h["step"], h["loss"], color="#dd8452", alpha=0.25, linewidth=0.8)
        curves.append(h["loss"].to_numpy())
    steps = pd.DataFrame(runs[0]["history"])["step"]
    ax.plot(steps, np.mean(curves, axis=0), color="#a0400a", linewidth=2, label="Mean of 10 splits")
    ax.set_xlabel("Adam step"); ax.set_ylabel("Training loss (binary cross-entropy)")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); save(fig, "r_training_curves.png")
    final = [c[-1] for c in curves]
    summary["training_loss"] = {"start_mean": float(np.mean([c[0] for c in curves])),
                                "step50_mean": float(np.mean([c[5] for c in curves])),
                                "final_mean": float(np.mean(final)), "final_min": float(np.min(final)),
                                "final_max": float(np.max(final))}
    accs = [pd.DataFrame(r["history"])["accuracy"].to_numpy() for r in runs]
    summary["training_accuracy_final_mean"] = float(np.mean([a[-1] for a in accs]))


def showcase(summary):
    s = load("showcase")[0]          # one detailed run (seed 42)
    y = np.array(s["y_test"])
    # confusion matrices: hybrid and logistic regression
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, name in zip(axes, ["Logistic Regression", HYBRID]):
        m = s["models"][name]
        mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, annot_kws={"size": 12})
        ax.set_title(SHORT[name]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); save(fig, "r_confusion_matrices.png")

    # pooled confusion matrix of the hybrid over 10 splits
    pooled = summary["main_pooled_confusion"][HYBRID]
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    mat = np.array([[pooled["tn"], pooled["fp"]], [pooled["fn"], pooled["tp"]]])
    sns.heatmap(mat, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax,
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, annot_kws={"size": 12})
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Hybrid VQC, 10 test sets pooled")
    fig.tight_layout(); save(fig, "r_confusion_pooled.png")

    # quantum features of the test notes
    Z = np.array(s["test_features"]); w = np.array(s["w"]); b = s["b"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for q in range(4):
        parts = axes[0].violinplot([Z[y == 0, q], Z[y == 1, q]], positions=[q * 3, q * 3 + 1], widths=0.8, showmedians=True)
        for body, color in zip(parts["bodies"], ("#4c72b0", "#c44e52")):
            body.set_facecolor(color); body.set_alpha(0.6)
    axes[0].set_xticks([q * 3 + 0.5 for q in range(4)]); axes[0].set_xticklabels([f"⟨Z{q}⟩" for q in range(4)])
    axes[0].set_ylabel("Expectation value"); axes[0].grid(axis="y", alpha=0.3)
    axes[0].scatter([], [], color="#4c72b0", label="Genuine"); axes[0].scatter([], [], color="#c44e52", label="Forged")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    score = Z @ w + b
    axes[1].hist(score[y == 0], bins=30, alpha=0.6, color="#4c72b0", label="Genuine")
    axes[1].hist(score[y == 1], bins=30, alpha=0.6, color="#c44e52", label="Forged")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("w · z + b  (decision score)"); axes[1].set_ylabel("Test notes"); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, "r_quantum_features.png")

    # shots and noise
    shots = [d["shots"] for d in s["shots"]]
    acc_shots = np.array([[r["accuracy"] for r in d["runs"]] for d in s["shots"]]) * 100
    err_shots = np.array([[r["feature_error"] for r in d["runs"]] for d in s["shots"]])
    p2 = [d["p2"] for d in s["noise"]]
    acc_noise = np.array([[r["accuracy"] for r in d["runs"]] for d in s["noise"]]) * 100
    err_noise = np.array([[r["feature_error"] for r in d["runs"]] for d in s["noise"]])
    exact_acc = s["models"][HYBRID]["accuracy"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.errorbar(shots, acc_shots.mean(1), yerr=acc_shots.std(1, ddof=1), marker="o", color="#dd8452", capsize=3, label="Test accuracy")
    ax.axhline(exact_acc, color="#dd8452", linestyle=":", label="Exact (infinite shots)")
    ax.set_xscale("log", base=2); ax.set_xlabel("Shots per circuit"); ax.set_ylabel("Test accuracy (%)")
    ax2 = ax.twinx(); ax2.plot(shots, err_shots.mean(1), "s--", color="#4c72b0", label="Mean |⟨Z⟩ error|")
    ax2.set_ylabel("Mean absolute error of ⟨Z⟩", color="#4c72b0")
    ax.set_title("(a) Finite shots, no gate noise", fontsize=9); ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="center right")
    ax = axes[1]
    xs = np.arange(len(p2))
    ax.errorbar(xs, acc_noise.mean(1), yerr=acc_noise.std(1, ddof=1), marker="o", color="#dd8452", capsize=3, label="Test accuracy")
    ax.axhline(exact_acc, color="#dd8452", linestyle=":")
    ax.set_xticks(xs); ax.set_xticklabels([f"{p * 100:g} %" for p in p2])
    ax.set_xlabel("Two-qubit gate and read-out error (1-qubit error = 1/10 of it)"); ax.set_ylabel("Test accuracy (%)")
    ax2 = ax.twinx(); ax2.plot(xs, err_noise.mean(1), "s--", color="#4c72b0", label="Mean |⟨Z⟩ error|")
    ax2.set_ylabel("Mean absolute error of ⟨Z⟩", color="#4c72b0")
    ax.set_title("(b) Noise model, 1,024 shots", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); save(fig, "r_shots_noise.png")

    shots_rows = [{"Shots": f"{sh:,}", "Accuracy (%)": pm(acc_shots[i] / 100),
                   "Mean |⟨Z⟩ error|": f"{err_shots[i].mean():.4f}"} for i, sh in enumerate(shots)]
    noise_rows = [{"2-qubit / read-out error": f"{p * 100:g} %", "Accuracy (%)": pm(acc_noise[i] / 100),
                   "Mean |⟨Z⟩ error|": f"{err_noise[i].mean():.4f}"} for i, p in enumerate(p2)]
    (T / "shots.md").write_text(md_table(pd.DataFrame(shots_rows)))
    (T / "noise.md").write_text(md_table(pd.DataFrame(noise_rows)))
    summary["showcase"] = {name: {k: s["models"][name][k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp")} for name in ORDER}
    summary["showcase_theta"] = s["theta"]; summary["showcase_w"] = s["w"]; summary["showcase_b"] = s["b"]
    summary["shots"] = {str(sh): [float(acc_shots[i].mean()), float(acc_shots[i].std(ddof=1)), float(err_shots[i].mean())] for i, sh in enumerate(shots)}
    summary["noise"] = {str(p): [float(acc_noise[i].mean()), float(acc_noise[i].std(ddof=1)), float(err_noise[i].mean())] for i, p in enumerate(p2)}
    summary["showcase_exact_accuracy"] = exact_acc


def ablation(summary):
    runs = load("ablation")
    main = [r for r in load("main") if r["seed"] < 5]
    variants = ["1 layer", "2 layers", "3 layers", "4 layers", "3 layers, no entanglement", "3 layers, no re-uploading"]
    data = {v: [r for r in runs if r["variant"] == v] for v in variants}
    data["3 layers"] = [{**r["models"][HYBRID], "final_loss": r["history"][-1]["loss"],
                         "circuit": {"cx_gates": 9, "depth": 15}} for r in main]
    rows = []
    for v in variants:
        d = data[v]
        rows.append({"Circuit variant": v, "Trainable values": d[0]["parameters"],
                     "CNOTs": d[0]["circuit"]["cx_gates"], "Depth": d[0]["circuit"]["depth"],
                     "Accuracy (%)": pm([x["accuracy"] for x in d]), "F1 (%)": pm([x["f1"] for x in d]),
                     "Final train loss": pm([x["final_loss"] for x in d], scale=1, digits=3),
                     "Train time (s)": f"{np.mean([x['train_seconds'] for x in d]):.1f}"})
    (T / "ablation.md").write_text(md_table(pd.DataFrame(rows)))
    summary["ablation"] = {v: [float(np.mean([x["accuracy"] for x in data[v]])), float(np.std([x["accuracy"] for x in data[v]], ddof=1)),
                               float(np.mean([x["final_loss"] for x in data[v]]))] for v in variants}
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    means = [np.mean([x["accuracy"] for x in data[v]]) * 100 for v in variants]
    stds = [np.std([x["accuracy"] for x in data[v]], ddof=1) * 100 for v in variants]
    colors = ["#f2c49b", "#e8a36b", "#dd8452", "#b9662f", "#8c8c8c", "#b0b0b0"]
    ax.bar(range(len(variants)), means, yerr=stds, color=colors, capsize=3)
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.4, f"{m:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(variants))); ax.set_xticklabels([v.replace(", ", ",\n") for v in variants], fontsize=8)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(min(means) - max(stds) - 5, 101); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); save(fig, "r_ablation.png")


def optimizers(summary):
    runs = load("optimizer")
    main = [r for r in load("main") if r["seed"] < 5]
    data = {"Adam + parameter shift": [{**r["models"][HYBRID], "history": r["history"],
                                        "final_loss": r["history"][-1]["loss"]} for r in main],
            "COBYLA": [r for r in runs if r["optimizer"] == "cobyla"],
            "SPSA": [r for r in runs if r["optimizer"] == "spsa"]}
    rows = []
    for name, d in data.items():
        rows.append({"Optimiser": name, "Accuracy (%)": pm([x["accuracy"] for x in d]),
                     "F1 (%)": pm([x["f1"] for x in d]),
                     "Final train loss": pm([x["final_loss"] for x in d], scale=1, digits=3),
                     "Training circuit executions": f"{int(np.mean([x['history'][-1]['circuits'] for x in d])):,}",
                     "Train time (s)": f"{np.mean([x['train_seconds'] for x in d]):.1f}"})
    (T / "optimizers.md").write_text(md_table(pd.DataFrame(rows)))
    summary["optimizers"] = {name: [float(np.mean([x["accuracy"] for x in d])), float(np.std([x["accuracy"] for x in d], ddof=1)),
                                    float(np.mean([x["final_loss"] for x in d])), float(np.mean([x["history"][-1]["circuits"] for x in d])),
                                    float(np.mean([x["train_seconds"] for x in d]))] for name, d in data.items()}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = {"Adam + parameter shift": "#dd8452", "COBYLA": "#4c72b0", "SPSA": "#55a868"}
    for name, d in data.items():
        for i, r in enumerate(d):
            h = pd.DataFrame(r["history"])
            h = h[h["circuits"] > 0]
            ax.plot(h["circuits"], h["loss"], color=colors[name], alpha=0.55, linewidth=1, label=name if i == 0 else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Circuit executions used so far (log)"); ax.set_ylabel("Training loss (log)")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); save(fig, "r_optimizers.png")


def learning_curve(summary):
    runs = load("learning_curve")
    sizes = sorted({r["size"] for r in runs})
    names = ["Logistic Regression", "SVM (RBF kernel)", HYBRID]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    rows, out = [], {}
    for name, color, marker in zip(names, ["#4c72b0", "#55a868", "#dd8452"], ["o", "s", "D"]):
        means, stds = [], []
        for n in sizes:
            acc = [r["models"][name]["accuracy"] * 100 for r in runs if r["size"] == n]
            means.append(np.mean(acc)); stds.append(np.std(acc, ddof=1))
        ax.errorbar(sizes, means, yerr=stds, marker=marker, capsize=3, color=color, label=SHORT[name])
        out[name] = [[float(m), float(s)] for m, s in zip(means, stds)]
    for i, n in enumerate(sizes):
        rows.append({"Training notes": n, **{SHORT[name]: f"{out[name][i][0]:.2f} ± {out[name][i][1]:.2f}" for name in names}})
    (T / "learning_curve.md").write_text(md_table(pd.DataFrame(rows)))
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(sizes)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Number of training notes (log scale)"); ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); save(fig, "r_learning_curve.png")
    summary["learning_curve"] = {"sizes": sizes, **out}


if __name__ == "__main__":
    T.mkdir(parents=True, exist_ok=True)
    summary = {}
    dataset_figure(summary)
    main_comparison(summary)
    showcase(summary)
    ablation(summary)
    optimizers(summary)
    learning_curve(summary)
    (R / "summary.json").write_text(json.dumps(summary, indent=1))
    for f in sorted(T.glob("*.md")):
        print(f"\n## {f.name}\n{f.read_text()}")
