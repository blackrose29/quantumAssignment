"""Qiskit demonstrations for Section 2 of the report.

    python -m src.quantum_algorithms

Builds and simulates small circuits for
    * superposition and a Bell state (entanglement), with and without noise
    * the Deutsch-Jozsa algorithm (3 input qubits, four different oracles)
    * Grover's search (3 qubits, marked item |110>)
    * the quantum part of Shor's algorithm: order finding for a = 2, N = 15

Circuit drawings go to figures/, measured numbers to results/section2_demos.json.
"""
from __future__ import annotations

import json
from fractions import Fraction
from math import gcd, pi, sqrt, asin, sin

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from . import config

SHOTS = 4096
SEED = 2026


def run_counts(circuit: QuantumCircuit, noise_model=None, shots: int = SHOTS) -> dict:
    """Simulate a circuit with measurements and return {bitstring: count}."""
    simulator = AerSimulator(noise_model=noise_model, seed_simulator=SEED)
    compiled = transpile(circuit, simulator, optimization_level=0)
    return dict(sorted(simulator.run(compiled, shots=shots).result().get_counts().items()))


def save_drawing(circuit: QuantumCircuit, filename: str, scale: float = 0.8, fold: int = -1) -> None:
    figure = circuit.draw(output="mpl", scale=scale, fold=fold, style="bw")
    figure.savefig(config.FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 1. Superposition and entanglement
# ---------------------------------------------------------------------------
def superposition_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc.h(0)                     # |0> -> (|0> + |1>)/sqrt(2)
    qc.measure(0, 0)
    return qc


def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)                     # superposition on qubit 0
    qc.cx(0, 1)                 # copy the "which branch" information to qubit 1
    qc.measure([0, 1], [0, 1])
    return qc


def simple_noise_model(p2: float = 0.05) -> NoiseModel:
    """Depolarising gate errors plus read-out bit flips (a rough NISQ device)."""
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["h", "x", "ry"])
    model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    model.add_all_qubit_readout_error(ReadoutError([[1 - p2 / 2, p2 / 2], [p2 / 2, 1 - p2 / 2]]))
    return model


# ---------------------------------------------------------------------------
# 2. Deutsch-Jozsa
# ---------------------------------------------------------------------------
def dj_oracle(kind: str, n: int = 3) -> QuantumCircuit:
    """Oracle |x>|y> -> |x>|y XOR f(x)> for four example functions."""
    oracle = QuantumCircuit(n + 1, name=f"U_f ({kind})")
    if kind == "constant 0":
        pass                                    # f(x) = 0: do nothing
    elif kind == "constant 1":
        oracle.x(n)                             # f(x) = 1: always flip the output qubit
    elif kind == "balanced x1":
        oracle.cx(1, n)                         # f(x) = x1
    elif kind == "balanced x0 XOR x2":
        oracle.cx(0, n)
        oracle.cx(2, n)                         # f(x) = x0 XOR x2
    else:
        raise ValueError(kind)
    return oracle


def deutsch_jozsa_circuit(kind: str, n: int = 3) -> QuantumCircuit:
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)                                     # output qubit to |1>
    qc.h(range(n + 1))                          # superposition of all inputs, output in |->
    qc.barrier()
    qc.compose(dj_oracle(kind, n), inplace=True)
    qc.barrier()
    qc.h(range(n))                              # interference
    qc.measure(range(n), range(n))
    return qc


# ---------------------------------------------------------------------------
# 3. Grover
# ---------------------------------------------------------------------------
def grover_oracle(marked: str) -> QuantumCircuit:
    """Flip the phase of the basis state |marked> (string written as q2 q1 q0)."""
    n = len(marked)
    oracle = QuantumCircuit(n, name="Oracle")
    zeros = [q for q, bit in enumerate(reversed(marked)) if bit == "0"]
    if zeros:
        oracle.x(zeros)
    oracle.h(n - 1); oracle.mcx(list(range(n - 1)), n - 1); oracle.h(n - 1)   # multi-controlled Z
    if zeros:
        oracle.x(zeros)
    return oracle


def grover_diffuser(n: int) -> QuantumCircuit:
    """Reflection about the uniform superposition: 2|s><s| - I."""
    diffuser = QuantumCircuit(n, name="Diffuser")
    diffuser.h(range(n)); diffuser.x(range(n))
    diffuser.h(n - 1); diffuser.mcx(list(range(n - 1)), n - 1); diffuser.h(n - 1)
    diffuser.x(range(n)); diffuser.h(range(n))
    return diffuser


def grover_circuit(marked: str, iterations: int, measure: bool = True, boxed: bool = False) -> QuantumCircuit:
    n = len(marked)
    qc = QuantumCircuit(n, n) if measure else QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        if boxed:                                # draw oracle and diffuser as labelled boxes
            qc.append(grover_oracle(marked).to_gate(), range(n))
            qc.append(grover_diffuser(n).to_gate(), range(n))
        else:
            qc.compose(grover_oracle(marked), inplace=True)
            qc.compose(grover_diffuser(n), inplace=True)
    if measure:
        qc.measure(range(n), range(n))
    return qc


def grover_theory(n: int, k: int) -> float:
    """P(success) = sin^2((2k + 1) * theta) with sin(theta) = 1/sqrt(N)."""
    theta = asin(1 / sqrt(2 ** n))
    return sin((2 * k + 1) * theta) ** 2


# ---------------------------------------------------------------------------
# 4. Shor: order finding for a = 2 modulo N = 15
# ---------------------------------------------------------------------------
def times_two_mod15() -> QuantumCircuit:
    """U|y> = |2y mod 15> on four qubits.

    Doubling modulo 15 is a cyclic shift of the four bits (1 -> 2 -> 4 -> 8 -> 1),
    which three SWAP gates implement.
    """
    u = QuantumCircuit(4, name="x2 mod 15")
    u.swap(2, 3); u.swap(1, 2); u.swap(0, 1)
    return u


def order_finding_circuit(counting: int = 4) -> QuantumCircuit:
    """Quantum phase estimation of U (multiplication by 2 mod 15)."""
    qc = QuantumCircuit(counting + 4, counting)
    qc.h(range(counting))                        # counting register in superposition
    qc.x(counting)                               # work register starts in |1>
    for j in range(counting):                    # controlled U^(2^j)
        power = QuantumCircuit(4, name=f"2^{2 ** j} mod 15")
        for _ in range(2 ** j % 4):              # U^4 = identity, so only the remainder matters
            power.compose(times_two_mod15(), inplace=True)
        qc.append(power.to_gate().control(1), [j] + list(range(counting, counting + 4)))
    qc.append(QFTGate(counting).inverse(), range(counting))
    qc.measure(range(counting), range(counting))
    return qc


def shor_postprocess(counts: dict, counting: int, a: int = 2, N: int = 15) -> list[dict]:
    """Turn each measured value y into a candidate order r and candidate factors."""
    rows = []
    for bits, count in counts.items():
        y = int(bits, 2)
        phase = y / 2 ** counting                           # y / 2^t is close to s / r
        r = Fraction(phase).limit_denominator(N).denominator  # continued fractions
        # r is accepted only if it really is a period (a^r = 1 mod N) and is even;
        # otherwise the run is discarded and the circuit would be repeated.
        valid = pow(a, r, N) == 1 and r % 2 == 0
        factors = sorted({gcd(a ** (r // 2) - 1, N), gcd(a ** (r // 2) + 1, N)} - {1, N}) if valid else []
        rows.append({"y": y, "count": count, "phase": str(Fraction(phase).limit_denominator(N)),
                     "r": r, "valid": valid, "factors": factors})
    return rows


# ---------------------------------------------------------------------------
def main() -> None:
    config.FIGURE_DIR.mkdir(exist_ok=True)
    config.RESULT_DIR.mkdir(exist_ok=True)
    out = {"shots": SHOTS}

    # --- superposition, entanglement and noise ------------------------------
    out["superposition"] = run_counts(superposition_circuit())
    bell = bell_circuit()
    save_drawing(bell, "s2_bell_circuit.png", scale=1.0)
    out["bell_ideal"] = run_counts(bell)
    out["bell_noisy"] = run_counts(bell, simple_noise_model(0.05))
    labels = ["00", "01", "10", "11"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    width = 0.38
    ax.bar(np.arange(4) - width / 2, [out["bell_ideal"].get(k, 0) / SHOTS for k in labels], width,
           label="Ideal simulator", color="#4c72b0")
    ax.bar(np.arange(4) + width / 2, [out["bell_noisy"].get(k, 0) / SHOTS for k in labels], width,
           label="With gate and read-out noise", color="#c44e52")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labels)
    ax.set_xlabel("Measured outcome (q1 q0)"); ax.set_ylabel("Relative frequency")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "s2_bell_counts.png", dpi=200); plt.close(fig)

    # --- Deutsch-Jozsa -------------------------------------------------------
    kinds = ["constant 0", "constant 1", "balanced x1", "balanced x0 XOR x2"]
    out["deutsch_jozsa"] = {kind: run_counts(deutsch_jozsa_circuit(kind)) for kind in kinds}
    save_drawing(deutsch_jozsa_circuit("balanced x0 XOR x2"), "s2_deutsch_jozsa_circuit.png")

    # --- Grover --------------------------------------------------------------
    marked, n = "110", 3
    save_drawing(grover_circuit(marked, 2, boxed=True), "s2_grover_circuit.png")
    save_drawing(grover_oracle(marked), "s2_grover_oracle.png")
    save_drawing(grover_diffuser(n), "s2_grover_diffuser.png")
    exact = []
    for k in range(7):
        state = Statevector(grover_circuit(marked, k, measure=False))
        exact.append(float(state.probabilities_dict().get(marked, 0.0)))
    out["grover"] = {"marked": marked, "exact_probability": exact,
                     "theory": [grover_theory(n, k) for k in range(7)],
                     "counts_k2": run_counts(grover_circuit(marked, 2)),
                     "optimal_k": int(np.floor(pi / 4 * sqrt(2 ** n)))}
    ks = np.linspace(0, 6, 300)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(ks, [grover_theory(n, k) for k in ks], color="#4c72b0", label=r"Theory $\sin^2((2k+1)\theta)$")
    ax.plot(range(7), exact, "o", color="#c44e52", label="Qiskit statevector simulation")
    ax.set_xlabel("Number of Grover iterations k"); ax.set_ylabel(f"Probability of |{marked}>")
    ax.legend(frameon=False, loc="lower right", fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "s2_grover_probability.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    counts = out["grover"]["counts_k2"]
    keys = [format(i, "03b") for i in range(8)]
    ax.bar(keys, [counts.get(k, 0) / SHOTS for k in keys],
           color=["#c44e52" if k == marked else "#4c72b0" for k in keys])
    ax.set_xlabel("Measured outcome (q2 q1 q0)"); ax.set_ylabel("Relative frequency")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "s2_grover_counts.png", dpi=200); plt.close(fig)

    # --- Shor / order finding --------------------------------------------------
    u = Operator(times_two_mod15()).data
    for y in range(1, 15):                         # check the multiplier on every basis state
        assert abs(u[(2 * y) % 15, y] - 1) < 1e-9
    counting = 4
    order_circuit = order_finding_circuit(counting)
    save_drawing(order_circuit, "s2_shor_circuit.png", scale=0.7)
    shor_counts = run_counts(order_circuit)
    out["shor"] = {"a": 2, "N": 15, "counting_qubits": counting, "counts": shor_counts,
                   "postprocessing": shor_postprocess(shor_counts, counting)}
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    values = {int(k, 2): v / SHOTS for k, v in shor_counts.items()}
    ax.bar(range(16), [values.get(i, 0) for i in range(16)], color="#55a868")
    ax.set_xticks(range(16)); ax.set_xlabel("Measured counting-register value y (out of 16)")
    ax.set_ylabel("Relative frequency"); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "s2_shor_counts.png", dpi=200); plt.close(fig)

    (config.RESULT_DIR / "section2_demos.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
