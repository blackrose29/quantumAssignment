"""Construction of the parameterised quantum circuit used by the hybrid model.

One layer of the circuit has three parts:

    1. Encoding   RY(x_q)      on every qubit q  (x_q = scaled feature q)
    2. Training   RY(theta)    on every qubit    (one trainable angle each)
    3. Entangling CNOT(q, q+1) along the chain 0-1-2-3

The layer is repeated `n_layers` times. Feeding the data in again in every
layer is called data re-uploading; it lets the circuit represent more complex
functions of the features than a single encoding step could.
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_circuit(
    n_qubits: int = 4,
    n_layers: int = 3,
    entangle: bool = True,
    reupload: bool = True,
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Return (circuit, feature parameters x, trainable parameters theta).

    entangle=False removes the CNOT gates (used for the ablation study).
    reupload=False encodes the data only in the first layer.
    """
    x = ParameterVector("x", n_qubits)
    theta = ParameterVector("θ", n_layers * n_qubits)
    circuit = QuantumCircuit(n_qubits, name="hybrid_vqc")

    for layer in range(n_layers):
        # 1. data encoding: one feature per qubit
        if reupload or layer == 0:
            for q in range(n_qubits):
                circuit.ry(x[q], q)
        # 2. trainable rotations
        for q in range(n_qubits):
            circuit.ry(theta[layer * n_qubits + q], q)
        # 3. entanglement along a chain of neighbouring qubits
        if entangle:
            for q in range(n_qubits - 1):
                circuit.cx(q, q + 1)
        if layer < n_layers - 1:
            circuit.barrier()
    return circuit, x, theta


def z_observables(n_qubits: int = 4) -> list[SparsePauliOp]:
    """Pauli-Z on each qubit in turn: Z_0, Z_1, ... (one expectation value each).

    Qiskit writes Pauli strings with qubit 0 on the right, so Z on qubit 0 of
    a four-qubit register is the string "IIIZ".
    """
    observables = []
    for q in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - 1 - q] = "Z"
        observables.append(SparsePauliOp("".join(label)))
    return observables


def circuit_summary(circuit: QuantumCircuit) -> dict:
    """Gate counts and depth, used for the cost table in the report."""
    ops = circuit.count_ops()
    return {
        "qubits": circuit.num_qubits,
        "depth": circuit.depth(filter_function=lambda inst: inst.operation.name != "barrier"),
        "ry_gates": int(ops.get("ry", 0)),
        "cx_gates": int(ops.get("cx", 0)),
        "parameters": circuit.num_parameters,
    }


if __name__ == "__main__":
    qc, _, _ = build_circuit()
    print(qc.draw(output="text", fold=120))
    print(circuit_summary(qc))
