"""Explanatory diagrams for the report (drawn with matplotlib, no external tools).

    python -m src.diagrams

Creates in figures/:
    d_classical_vs_quantum.png   block diagram of the two computing stacks
    d_bloch_sphere.png           Bloch-sphere picture of a qubit and an RY rotation
    d_grover_geometry.png        Grover iteration as a rotation in a plane
    d_shor_flowchart.png         classical and quantum steps of Shor's algorithm
    d_complexity.png             growth of classical and quantum costs
    d_system_architecture.png    overall system architecture of the mini project
    d_model_architecture.png     hybrid model: classical and quantum parts
    d_workflow_flowchart.png     complete implementation workflow
    d_use_case.png               UML-style use case diagram
    d_hybrid_circuit_layer.png   one layer of the circuit (Qiskit drawing)
    d_hybrid_circuit_full.png    the full three-layer circuit (Qiskit drawing)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from . import config
from .quantum_circuit import build_circuit

CLASSICAL = "#dce9f5"     # classical processing
QUANTUM = "#fde2b3"       # quantum processing
DATA = "#e2f0d9"          # data and results
LOOP = "#ead9f2"          # optimisation loop
EDGE = "#3b3b3b"


def _box(ax, x, y, w, h, text, color=CLASSICAL, fontsize=8.5, bold=False, rounding=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={rounding}",
                                facecolor=color, edgecolor=EDGE, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, start, end, text=None, style="-|>", connection="arc3", dashed=False, text_offset=(0, 0.08),
           fontsize=7.5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11, color=EDGE,
                                 linewidth=1.0, connectionstyle=connection,
                                 linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))
    if text:
        ax.text((start[0] + end[0]) / 2 + text_offset[0], (start[1] + end[1]) / 2 + text_offset[1], text,
                ha="center", va="center", fontsize=fontsize)


def _diamond(ax, cx, cy, w, h, text, color="#fff2cc", fontsize=8.5):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)],
                         closed=True, facecolor=color, edgecolor=EDGE, linewidth=1.0))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)


def _canvas(width, height, xlim, ylim):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def _legend(ax, x, y, items, fontsize=7.5, dx=2.3):
    for i, (color, label) in enumerate(items):
        ax.add_patch(Rectangle((x + i * dx, y), 0.28, 0.2, facecolor=color, edgecolor=EDGE, linewidth=0.8))
        ax.text(x + i * dx + 0.38, y + 0.1, label, va="center", fontsize=fontsize)


def _save(fig, name):
    fig.savefig(config.FIGURE_DIR / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
def classical_vs_quantum():
    fig, ax = _canvas(7.4, 4.3, (0, 10), (0, 6))
    ax.text(2.4, 5.7, "Classical computer", ha="center", fontsize=10, fontweight="bold")
    ax.text(7.6, 5.7, "Quantum computer (as used today)", ha="center", fontsize=10, fontweight="bold")
    left = ["Program (high-level language)", "Compiler: machine instructions",
            "CPU / GPU: logic gates on bits\n(AND, OR, NOT)", "Memory: bits, each 0 or 1", "Output: read bits directly\n(deterministic)"]
    right = ["Program (Qiskit circuit) on a classical host", "Transpiler: native gates, qubit layout",
             "Control electronics: microwave / laser pulses", "QPU: qubits in superposition\nand entanglement (cryostat / trap)",
             "Measurement: random bit strings,\nrepeated for many shots"]
    for i, text in enumerate(left):
        _box(ax, 0.3, 4.6 - i * 1.05, 4.2, 0.8, text, CLASSICAL)
        if i:
            _arrow(ax, (2.4, 4.6 - (i - 1) * 1.05), (2.4, 4.6 - i * 1.05 + 0.8))
    for i, text in enumerate(right):
        _box(ax, 5.2, 4.6 - i * 1.05, 4.8, 0.8, text, QUANTUM if i >= 2 else CLASSICAL, fontsize=8)
        if i:
            _arrow(ax, (7.6, 4.6 - (i - 1) * 1.05), (7.6, 4.6 - i * 1.05 + 0.8))
    ax.plot([10.0, 10.35, 10.35], [0.8, 0.8, 5.0], color=EDGE, linewidth=1.0, linestyle="--")
    _arrow(ax, (10.35, 5.0), (10.0, 5.0), dashed=True)
    ax.text(10.75, 2.9, "results return to\nthe classical host", rotation=90, ha="center", va="center", fontsize=7.5)
    ax.set_xlim(0, 11.1)
    _save(fig, "d_classical_vs_quantum.png")


def bloch_sphere():
    fig, ax = _canvas(4.4, 4.2, (-1.55, 1.75), (-1.45, 1.5))
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), 1, facecolor="#f4f7fb", edgecolor=EDGE, linewidth=1.0))
    ax.add_patch(Ellipse((0, 0), 2, 0.55, facecolor="none", edgecolor="#8a8a8a", linewidth=0.8, linestyle="--"))
    ax.plot([0, 0], [-1, 1], color="#8a8a8a", linewidth=0.8)
    ax.plot([-1, 1], [0, 0], color="#8a8a8a", linewidth=0.8, linestyle=":")
    ax.text(0, 1.1, r"$|0\rangle$", ha="center", fontsize=12)
    ax.text(0, -1.25, r"$|1\rangle$", ha="center", fontsize=12)
    ax.text(1.08, -0.05, r"$|+\rangle$", ha="left", va="center", fontsize=11)
    ax.text(-1.08, -0.05, r"$|-\rangle$", ha="right", va="center", fontsize=11)
    theta = np.deg2rad(55)
    tip = (np.sin(theta), np.cos(theta))
    ax.add_patch(FancyArrowPatch((0, 0), (0, 1), arrowstyle="-|>", mutation_scale=12, color="#7f7f7f", linewidth=1.3))
    ax.add_patch(FancyArrowPatch((0, 0), tip, arrowstyle="-|>", mutation_scale=13, color="#c44e52", linewidth=1.8))
    arc = np.linspace(0, theta, 40)
    ax.plot(0.38 * np.sin(arc), 0.38 * np.cos(arc), color="#c44e52", linewidth=1.2)
    ax.text(0.2, 0.52, r"$\theta$", color="#c44e52", fontsize=12)
    ax.text(tip[0] + 0.06, tip[1] + 0.04, r"$R_Y(\theta)|0\rangle$", color="#c44e52", fontsize=10.5)
    ax.text(0.1, -1.42, r"$R_Y(\theta)|0\rangle=\cos\frac{\theta}{2}|0\rangle+\sin\frac{\theta}{2}|1\rangle$,"
                        r"  $\langle Z\rangle=\cos\theta$", ha="center", fontsize=9)
    _save(fig, "d_bloch_sphere.png")


def grover_geometry():
    fig, ax = _canvas(4.6, 3.9, (-0.25, 1.35), (-0.2, 1.25))
    ax.set_aspect("equal")
    ax.add_patch(FancyArrowPatch((0, 0), (1.25, 0), arrowstyle="-|>", mutation_scale=11, color=EDGE, linewidth=1.0))
    ax.add_patch(FancyArrowPatch((0, 0), (0, 1.15), arrowstyle="-|>", mutation_scale=11, color=EDGE, linewidth=1.0))
    ax.text(1.27, -0.09, r"$|s'\rangle$ (all unmarked items)", ha="right", fontsize=9)
    ax.text(0.03, 1.17, r"$|w\rangle$ (marked item)", fontsize=9)
    theta = np.arcsin(1 / np.sqrt(8))
    colors = ["#7f7f7f", "#4c72b0", "#c44e52"]
    labels = [r"$|s\rangle$ start", "after 1 iteration", "after 2 iterations"]
    for k in range(3):
        angle = (2 * k + 1) * theta
        ax.add_patch(FancyArrowPatch((0, 0), (np.cos(angle), np.sin(angle)), arrowstyle="-|>",
                                     mutation_scale=12, color=colors[k], linewidth=1.7))
        ax.text(np.cos(angle) + 0.03, np.sin(angle) + 0.02, labels[k], color=colors[k], fontsize=8.5)
    arc = np.linspace(0, theta, 20)
    ax.plot(0.45 * np.cos(arc), 0.45 * np.sin(arc), color=EDGE, linewidth=0.9)
    ax.text(0.48, 0.07, r"$\theta$", fontsize=10)
    arc = np.linspace(theta, 3 * theta, 30)
    ax.plot(0.6 * np.cos(arc), 0.6 * np.sin(arc), color=EDGE, linewidth=0.9)
    ax.text(0.56, 0.42, r"$2\theta$", fontsize=10)
    ax.text(0.55, -0.17, r"$\sin\theta = 1/\sqrt{N}$, here $N=8$", ha="center", fontsize=9)
    _save(fig, "d_grover_geometry.png")


def shor_flowchart():
    fig, ax = _canvas(5.6, 7.2, (0, 8), (0, 11.2))
    w, x = 5.2, 1.4
    steps = [
        (10.2, "Input: odd composite N (not a prime power)", DATA),
        (9.0, "Pick a random a with 1 < a < N", CLASSICAL),
        (6.3, "QUANTUM: phase estimation of\n" + r"$U|y\rangle=|a\,y\ \mathrm{mod}\ N\rangle$, inverse QFT, measure $y$", QUANTUM),
        (5.1, r"Continued fractions on $y/2^t$ give the order $r$", CLASSICAL),
        (1.3, r"Factors: $\gcd(a^{r/2}\pm 1,\,N)$", DATA),
    ]
    for y, text, color in steps:
        _box(ax, x, y, w, 0.8, text, color)
    _diamond(ax, x + w / 2, 7.95, 4.6, 1.0, r"$\gcd(a, N) = 1$ ?")
    _diamond(ax, x + w / 2, 3.45, 5.2, 1.3, "$r$ even and\n" + r"$a^{r/2} \neq -1\ (\mathrm{mod}\ N)$ ?")
    cx = x + w / 2
    _arrow(ax, (cx, 10.2), (cx, 9.8)); _arrow(ax, (cx, 9.0), (cx, 8.45))
    _arrow(ax, (cx, 7.45), (cx, 7.1), "yes", text_offset=(0.35, 0))
    _arrow(ax, (cx, 6.3), (cx, 5.9)); _arrow(ax, (cx, 5.1), (cx, 4.1))
    _arrow(ax, (cx, 2.8), (cx, 2.1), "yes", text_offset=(0.35, 0))
    # gcd != 1: factor found by luck
    _arrow(ax, (cx + 2.3, 7.95), (7.6, 7.95)); ax.plot([7.6, 7.6], [7.95, 1.7], color=EDGE, linewidth=1.0)
    _arrow(ax, (7.6, 1.7), (x + w, 1.7))
    ax.text(7.25, 8.15, "no", fontsize=7.5); ax.text(7.75, 4.8, "gcd(a, N) is\nalready a factor", fontsize=7, rotation=90, va="center")
    # failed test: choose a new a
    ax.plot([cx - 2.6, 0.5], [3.45, 3.45], color=EDGE, linewidth=1.0); ax.plot([0.5, 0.5], [3.45, 9.4], color=EDGE, linewidth=1.0)
    _arrow(ax, (0.5, 9.4), (x, 9.4)); ax.text(0.8, 3.65, "no", fontsize=7.5)
    ax.text(0.28, 6.4, "try another a", fontsize=7, rotation=90, va="center")
    _legend(ax, 1.0, 0.3, [(CLASSICAL, "classical step"), (QUANTUM, "quantum step"), (DATA, "input / output")], dx=2.3)
    _save(fig, "d_shor_flowchart.png")


def complexity_plot():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    n = np.arange(2, 41)
    axes[0].semilogy(n, 2.0 ** n / 2, label="Classical search: N/2 queries", color="#4c72b0")
    axes[0].semilogy(n, np.pi / 4 * np.sqrt(2.0 ** n), label=r"Grover: $(\pi/4)\sqrt{N}$ queries", color="#c44e52")
    axes[0].semilogy(n, 2.0 ** (n - 1) + 1, ":", label=r"Deutsch-Jozsa, classical worst case: $N/2+1$", color="#7f7f7f")
    axes[0].axhline(1, color="#55a868", linestyle="--", label="Deutsch-Jozsa, quantum: 1 query")
    axes[0].set_xlabel("Number of qubits n  (N = 2^n items)"); axes[0].set_ylabel("Oracle queries")
    axes[0].legend(fontsize=6.5, frameon=False); axes[0].grid(alpha=0.3)
    bits = np.linspace(64, 4096, 200)
    ln_n = bits * np.log(2)
    gnfs = np.exp((64 / 9) ** (1 / 3) * ln_n ** (1 / 3) * np.log(ln_n) ** (2 / 3))
    axes[1].semilogy(bits, gnfs, label="Number field sieve (classical)", color="#4c72b0")
    axes[1].semilogy(bits, bits ** 3, label=r"Shor: about $(\log N)^3$ gates", color="#c44e52")
    axes[1].set_xlabel("Size of N in bits"); axes[1].set_ylabel("Operations (constants omitted)")
    axes[1].legend(fontsize=7, frameon=False); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "d_complexity.png")


def system_architecture():
    fig, ax = _canvas(7.6, 5.0, (0, 12), (0, 8))
    lanes = [(6.4, "Data layer", DATA), (4.75, "Pre-processing", CLASSICAL), (2.3, "Model layer", "#f7f7f7"),
             (0.5, "Evaluation layer", DATA)]
    for y, label, _ in lanes:
        ax.text(0.15, y + 0.55, label, fontsize=8, fontweight="bold", rotation=90, va="center", ha="center")
    _box(ax, 0.6, 6.4, 3.3, 1.1, "UCI Banknote Authentication\n1,372 notes, 4 wavelet features", DATA, fontsize=7.6)
    _box(ax, 4.5, 6.4, 3.0, 1.1, "Stratified split\n80 % train / 20 % test", DATA)
    _box(ax, 8.1, 6.4, 3.5, 1.1, "Random seeds 0-9\n(repeated experiments)", DATA)
    _box(ax, 0.6, 4.75, 4.6, 1.0, "StandardScaler\n(zero mean, unit variance)", CLASSICAL)
    _box(ax, 6.2, 4.75, 5.4, 1.0, "AngleScaler: min-max to [0, π]\n(one rotation angle per feature)", CLASSICAL)
    _box(ax, 0.6, 2.3, 4.6, 1.75, "Classical baselines (scikit-learn)\nLogistic regression, SVM-RBF, k-NN,\nrandom forest, small neural network", CLASSICAL)
    ax.add_patch(FancyBboxPatch((6.2, 2.05), 5.4, 2.0, boxstyle="round,pad=0.0,rounding_size=0.06",
                                facecolor="#fffaf0", edgecolor=EDGE, linewidth=1.0, linestyle="--"))
    ax.text(8.9, 3.78, "Hybrid quantum-classical model", fontsize=8, fontweight="bold", ha="center", va="center")
    _box(ax, 6.35, 2.5, 2.0, 1.05, "4-qubit circuit\nQiskit Aer\nsimulator", QUANTUM, fontsize=7.3)
    _box(ax, 8.6, 2.5, 1.3, 1.05, "Logistic\noutput\nlayer", CLASSICAL, fontsize=7.3)
    _box(ax, 10.15, 2.5, 1.3, 1.05, "Adam +\nparameter\nshift", LOOP, fontsize=7.3)
    _arrow(ax, (8.35, 3.02), (8.6, 3.02)); _arrow(ax, (9.9, 3.02), (10.15, 3.02))
    ax.plot([10.8, 10.8, 7.35], [2.5, 2.25, 2.25], color=EDGE, linewidth=1.0, linestyle="--")
    _arrow(ax, (7.35, 2.25), (7.35, 2.5), dashed=True)
    _box(ax, 0.6, 0.5, 11.0, 1.0, "Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, training time, circuit executions\n"
                                   "Ablation, optimiser comparison, learning curve, shot-noise and gate-noise study", DATA)
    _arrow(ax, (3.9, 6.95), (4.5, 6.95)); _arrow(ax, (8.1, 6.95), (7.5, 6.95))
    _arrow(ax, (5.2, 6.4), (2.9, 5.75)); _arrow(ax, (6.8, 6.4), (8.9, 5.75))
    _arrow(ax, (2.9, 4.75), (2.9, 4.05)); _arrow(ax, (8.9, 4.75), (8.9, 4.05))
    _arrow(ax, (2.9, 2.3), (2.9, 1.5)); _arrow(ax, (8.9, 2.05), (8.9, 1.5))
    _legend(ax, 0.6, -0.15, [(DATA, "data / results"), (CLASSICAL, "classical processing"),
                              (QUANTUM, "quantum processing"), (LOOP, "optimisation loop")], dx=2.85)
    ax.set_ylim(-0.3, 7.7)
    _save(fig, "d_system_architecture.png")


def model_architecture():
    fig, ax = _canvas(7.8, 3.5, (0, 13.6), (0, 5.5))
    _box(ax, 0.2, 3.1, 1.8, 1.5, "Banknote\nfeatures\n$x_1 \\ldots x_4$", DATA)
    _box(ax, 2.4, 3.1, 1.8, 1.5, "Scale to\nangles\n$\\phi_i \\in [0,\\pi]$", CLASSICAL)
    ax.add_patch(FancyBboxPatch((4.6, 2.75), 4.4, 2.4, boxstyle="round,pad=0.0,rounding_size=0.06",
                                facecolor=QUANTUM, edgecolor=EDGE, linewidth=1.0))
    ax.text(6.8, 4.8, "Quantum circuit: 4 qubits, 3 layers", fontsize=7.6, fontweight="bold", ha="center", va="center")
    for i, text in enumerate(["Encode\n$R_Y(\\phi_i)$", "Train\n$R_Y(\\theta_k)$", "Entangle\nCNOTs"]):
        _box(ax, 4.75 + i * 1.43, 3.0, 1.22, 1.3, text, "#fff6e3", fontsize=7.4)
        if i:
            _arrow(ax, (4.75 + i * 1.43 - 0.21, 3.65), (4.75 + i * 1.43, 3.65))
    _box(ax, 9.4, 3.1, 1.75, 1.5, "Measure\n$\\langle Z_1\\rangle \\ldots \\langle Z_4\\rangle$", QUANTUM, fontsize=7.8)
    _box(ax, 11.55, 3.1, 1.9, 1.5, "Logistic layer\n$p=\\sigma(w\\cdot z+b)$", CLASSICAL, fontsize=7.3)
    for start, end in [((2.0, 3.85), (2.4, 3.85)), ((4.2, 3.85), (4.6, 3.85)), ((9.0, 3.85), (9.4, 3.85)),
                       ((11.15, 3.85), (11.55, 3.85))]:
        _arrow(ax, start, end)
    _box(ax, 9.7, 0.75, 3.75, 1.15, "Binary cross-entropy loss\ncompared with true label $y$", LOOP, fontsize=8)
    _box(ax, 3.9, 0.75, 4.8, 1.15, "Classical optimiser (Adam)\nparameter-shift gradients for $\\theta$,\nordinary gradients for $w$, $b$", LOOP, fontsize=7.8)
    _arrow(ax, (12.9, 3.1), (12.9, 1.9)); _arrow(ax, (9.7, 1.32), (8.7, 1.32))
    _arrow(ax, (5.6, 1.9), (5.6, 2.75), "update $\\theta$", text_offset=(-0.8, 0), dashed=True)
    ax.plot([8.0, 8.0, 12.0], [1.9, 2.35, 2.35], color=EDGE, linewidth=1.0, linestyle="--")
    _arrow(ax, (12.0, 2.35), (12.0, 3.1), dashed=True)
    ax.text(10.0, 2.55, "update $w$, $b$", fontsize=7.5, ha="center")
    _legend(ax, 0.2, 0.05, [(CLASSICAL, "classical"), (QUANTUM, "quantum"), (LOOP, "training loop"), (DATA, "data")], dx=2.3)
    _save(fig, "d_model_architecture.png")


def workflow_flowchart():
    fig, ax = _canvas(6.2, 8.6, (0, 9), (0, 14))
    cx, w, h = 4.5, 5.6, 0.72
    ys = [13.1, 12.1, 11.1, 10.1, 9.1, 8.1, 7.1, 6.1]
    texts = [("Start: load banknote dataset (1,372 x 4)", DATA),
             ("Stratified 80/20 split with seed s", CLASSICAL),
             ("Fit scalers on the training notes only", CLASSICAL),
             ("Train classical baselines, store test metrics", CLASSICAL),
             ("Build 4-qubit circuit; initialise θ, w, b", QUANTUM),
             ("Draw a mini-batch of 32 training notes", CLASSICAL),
             ("Run circuit: ⟨Z⟩ for the batch, and again\nwith every θ shifted by ±π/2", QUANTUM),
             ("Loss, gradients, Adam update of θ, w, b", LOOP)]
    for y, (text, color) in zip(ys, texts):
        _box(ax, cx - w / 2, y, w, h, text, color, fontsize=7.6 if "\n" in text else 8)
    for upper, lower in zip(ys[:-1], ys[1:]):
        _arrow(ax, (cx, upper), (cx, lower + h))
    _diamond(ax, cx, 5.05, 4.0, 1.0, "300 steps done?", fontsize=8)
    _arrow(ax, (cx, 6.1), (cx, 5.55))
    ax.plot([cx + 2.0, 8.3], [5.05, 5.05], color=EDGE, linewidth=1.0); ax.plot([8.3, 8.3], [5.05, 8.46], color=EDGE, linewidth=1.0)
    _arrow(ax, (8.3, 8.46), (cx + w / 2, 8.46)); ax.text(7.3, 5.25, "no", fontsize=7.5)
    tail = [(3.55, "Predict the test notes: p ≥ 0.5 means forged", QUANTUM),
            (2.55, "Metrics, confusion matrix, comparison with baselines", DATA),
            (1.55, "Robustness: finite shots and noise model", QUANTUM)]
    _arrow(ax, (cx, 4.55), (cx, 3.55 + h), "yes", text_offset=(0.35, 0))
    for y, text, color in tail:
        _box(ax, cx - w / 2, y, w, h, text, color, fontsize=8)
    _arrow(ax, (cx, 3.55), (cx, 2.55 + h)); _arrow(ax, (cx, 2.55), (cx, 1.55 + h))
    _diamond(ax, cx, 0.6, 4.0, 0.9, "more seeds?", fontsize=8)
    _arrow(ax, (cx, 1.55), (cx, 1.05))
    ax.plot([cx - 2.0, 0.6], [0.6, 0.6], color=EDGE, linewidth=1.0); ax.plot([0.6, 0.6], [0.6, 12.46], color=EDGE, linewidth=1.0)
    _arrow(ax, (0.6, 12.46), (cx - w / 2, 12.46)); ax.text(1.4, 0.8, "yes", fontsize=7.5)
    _arrow(ax, (cx + 2.0, 0.6), (8.2, 0.6), "no", text_offset=(0, 0.2)); ax.text(8.3, 0.6, "End", fontsize=8.5, va="center", fontweight="bold")
    _legend(ax, 0.9, -0.55, [(CLASSICAL, "classical"), (QUANTUM, "quantum circuit"), (LOOP, "optimiser"), (DATA, "data")], dx=2.0)
    ax.set_ylim(-0.8, 14)
    _save(fig, "d_workflow_flowchart.png")


def _actor(ax, x, y, label):
    ax.add_patch(plt.Circle((x, y + 0.55), 0.18, facecolor="white", edgecolor=EDGE, linewidth=1.1))
    ax.plot([x, x], [y + 0.37, y - 0.1], color=EDGE, linewidth=1.1)
    ax.plot([x - 0.28, x + 0.28], [y + 0.18, y + 0.18], color=EDGE, linewidth=1.1)
    ax.plot([x, x - 0.24], [y - 0.1, y - 0.5], color=EDGE, linewidth=1.1)
    ax.plot([x, x + 0.24], [y - 0.1, y - 0.5], color=EDGE, linewidth=1.1)
    ax.text(x, y - 0.72, label, ha="center", va="top", fontsize=8)


def use_case_diagram():
    fig, ax = _canvas(7.0, 5.0, (0, 12), (0, 8.4))
    ax.add_patch(Rectangle((2.7, 0.4), 6.7, 7.6, facecolor="white", edgecolor=EDGE, linewidth=1.1))
    ax.text(6.05, 7.65, "Hybrid banknote authentication system", ha="center", fontsize=8.5, fontweight="bold")
    cases = [(6.0, 6.85, "Submit wavelet features of a note"), (6.0, 5.95, "Receive genuine / forged decision"),
             (6.0, 5.05, "Prepare data (split, scale to angles)"), (6.0, 4.15, "Train hybrid model"),
             (6.0, 3.25, "Compare with classical baselines"), (6.0, 2.35, "Test robustness (shots, noise)"),
             (6.0, 1.3, "Execute parameterised circuits")]
    for x, y, text in cases:
        ax.add_patch(Ellipse((x, y), 4.7, 0.7, facecolor=QUANTUM if "circuits" in text else CLASSICAL,
                             edgecolor=EDGE, linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.8)
    _actor(ax, 1.2, 6.2, "Cash-handling\noperator"); _actor(ax, 1.2, 3.3, "ML engineer\n(student)"); _actor(ax, 10.8, 3.4, "Quantum backend\n(Qiskit Aer\nsimulator)")
    for y in (6.85, 5.95):
        ax.plot([1.55, 3.65], [6.35, y], color=EDGE, linewidth=0.9)
    for y in (5.05, 4.15, 3.25, 2.35):
        ax.plot([1.55, 3.65], [3.45, y], color=EDGE, linewidth=0.9)
    ax.plot([10.4, 8.35], [3.5, 1.3], color=EDGE, linewidth=0.9)
    for y in (5.95, 4.15, 2.35):                       # use cases that include circuit execution
        ax.plot([8.35, 8.85], [y, y], color="#6d6d6d", linewidth=0.8, linestyle="--")
    ax.plot([8.85, 8.85], [5.95, 1.95], color="#6d6d6d", linewidth=0.8, linestyle="--")
    ax.add_patch(FancyArrowPatch((8.85, 1.95), (7.7, 1.58), arrowstyle="->", mutation_scale=9, color="#6d6d6d",
                                 linewidth=0.8, linestyle="--"))
    ax.text(9.08, 4.6, "«include»", fontsize=7, color="#555555", rotation=90, va="center")
    _save(fig, "d_use_case.png")


def hybrid_circuit_drawings():
    for layers, name, scale in ((1, "d_hybrid_circuit_layer.png", 0.95), (3, "d_hybrid_circuit_full.png", 0.62)):
        circuit, _, _ = build_circuit(n_layers=layers)
        if layers == 1:
            circuit.measure_all()
        figure = circuit.draw(output="mpl", scale=scale, fold=-1, style="bw")
        figure.savefig(config.FIGURE_DIR / name, dpi=200, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    config.FIGURE_DIR.mkdir(exist_ok=True)
    for draw in (classical_vs_quantum, bloch_sphere, grover_geometry, shor_flowchart, complexity_plot,
                 system_architecture, model_architecture, workflow_flowchart, use_case_diagram,
                 hybrid_circuit_drawings):
        draw()
        print("drawn:", draw.__name__)


if __name__ == "__main__":
    main()
