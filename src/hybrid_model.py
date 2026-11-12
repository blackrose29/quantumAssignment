"""Hybrid quantum-classical classifier for banknote authentication.

The model has a quantum part and a classical part that are trained together:

    features x (4 angles)
        |
        v
    QUANTUM   parameterised circuit U(x, theta) on 4 qubits   (simulated with Qiskit Aer)
        |     measure <Z_0>, <Z_1>, <Z_2>, <Z_3>  -> 4 numbers in [-1, 1]
        v
    CLASSICAL logistic output layer  p = sigmoid(w . <Z> + b)
        |
        v
    probability that the banknote is forged

Training minimises the binary cross-entropy loss. The gradient with respect to
the circuit angles theta is obtained with the parameter-shift rule (two extra
circuit runs per angle), the gradient with respect to w and b is ordinary
calculus, and the Adam optimiser updates all of them.
"""
from __future__ import annotations

import time

import numpy as np
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy.optimize import minimize

from . import config
from .quantum_circuit import build_circuit, z_observables


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, clipped so that exp() cannot overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def binary_cross_entropy(y_true: np.ndarray, p: np.ndarray) -> float:
    """Average of -[y log p + (1 - y) log(1 - p)] over the samples."""
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


class HybridQuantumClassifier:
    """Variational quantum circuit followed by a classical logistic layer."""

    def __init__(
        self,
        n_qubits: int = config.N_QUBITS,
        n_layers: int = config.N_LAYERS,
        entangle: bool = True,
        reupload: bool = True,
        optimizer: str = "adam",          # "adam", "cobyla" or "spsa"
        steps: int = config.TRAIN_STEPS,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        final_lr_fraction: float = config.FINAL_LR_FRACTION,
        log_every: int = config.LOG_EVERY,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.final_lr_fraction = final_lr_fraction
        self.log_every = log_every
        self.seed = seed
        self.verbose = verbose

        # ---- quantum part -------------------------------------------------
        self.circuit, self._x, self._theta = build_circuit(n_qubits, n_layers, entangle, reupload)
        self._params = tuple(self._x) + tuple(self._theta)       # binding order
        self._observables = [[obs] for obs in z_observables(n_qubits)]
        # Exact (noise-free, infinite-shot) expectation values from Qiskit Aer.
        self._estimator = AerEstimator(
            options={"default_precision": 0.0, "backend_options": {"max_parallel_threads": 1}}
        )
        self.n_theta = len(self._theta)

        # ---- trainable parameters ------------------------------------------
        rng = np.random.default_rng(seed)
        self.theta = rng.uniform(-np.pi, np.pi, self.n_theta)     # circuit angles
        self.w = np.ones(n_qubits)                                # classical weights
        self.b = 0.0                                              # classical bias

        # ---- bookkeeping ---------------------------------------------------
        self.history: list[dict] = []
        self.circuit_evaluations = 0
        self.fit_seconds = 0.0

    # ------------------------------------------------------------------
    # Quantum part
    # ------------------------------------------------------------------
    def quantum_features(self, X: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        """Run the circuit for every row of X and return <Z_q> for each qubit.

        X has shape (samples, n_qubits) and already contains rotation angles.
        The result has shape (samples, n_qubits) with values in [-1, 1].
        """
        theta = self.theta if theta is None else theta
        rows = np.hstack([X, np.tile(theta, (len(X), 1))])
        return self._expectations(rows)

    def _expectations(self, rows: np.ndarray) -> np.ndarray:
        """rows[i] = (x_0..x_3, theta_0..theta_P-1) for circuit run i."""
        job = self._estimator.run([(self.circuit, self._observables, {self._params: rows})])
        self.circuit_evaluations += len(rows)
        return np.asarray(job.result()[0].data.evs).T            # (runs, qubits)

    def sampled_features(self, X: np.ndarray, shots: int, noise_model=None, seed: int = 0) -> np.ndarray:
        """Estimate <Z_q> from a finite number of measurement shots.

        This is what a real device does: prepare the state, measure all qubits,
        repeat `shots` times and average. If bit q is 1 in a fraction f of the
        shots then <Z_q> is estimated as 1 - 2f. An optional Aer noise model
        adds gate and read-out errors.
        """
        backend_options = {"max_parallel_threads": 1}
        if noise_model is not None:
            backend_options["noise_model"] = noise_model
        sampler = AerSampler(seed=seed, options={"backend_options": backend_options})
        measured = self.circuit.copy()
        measured.measure_all()
        rows = np.hstack([X, np.tile(self.theta, (len(X), 1))])
        result = sampler.run([(measured, {self._params: rows}, shots)]).result()[0]
        raw = result.data.meas.array[..., -1]                    # (runs, shots) integers 0..15
        bits = (raw[..., None] >> np.arange(self.n_qubits)) & 1   # bit q of each outcome
        return 1.0 - 2.0 * bits.mean(axis=1)

    # ------------------------------------------------------------------
    # Classical part
    # ------------------------------------------------------------------
    def _head(self, features: np.ndarray, w=None, b=None) -> np.ndarray:
        w = self.w if w is None else w
        b = self.b if b is None else b
        return sigmoid(features @ w + b)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability that each banknote is forged (class 1)."""
        return self._head(self.quantum_features(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_proba_from_features(self, features: np.ndarray) -> np.ndarray:
        """Apply the classical layer to already measured (e.g. noisy) features."""
        return self._head(features)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridQuantumClassifier":
        start = time.perf_counter()
        y = np.asarray(y, dtype=float)
        if self.optimizer == "adam":
            self._fit_adam(X, y)
        elif self.optimizer == "cobyla":
            self._fit_cobyla(X, y)
        elif self.optimizer == "spsa":
            self._fit_spsa(X, y)
        else:
            raise ValueError(f"unknown optimizer: {self.optimizer}")
        self.fit_seconds = time.perf_counter() - start
        return self

    # -- helpers shared by the optimisers ---------------------------------
    def _pack(self) -> np.ndarray:
        return np.concatenate([self.theta, self.w, [self.b]])

    def _unpack(self, vector: np.ndarray) -> None:
        self.theta = np.array(vector[: self.n_theta])
        self.w = np.array(vector[self.n_theta: self.n_theta + self.n_qubits])
        self.b = float(vector[-1])

    def _loss_of(self, vector: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        theta = vector[: self.n_theta]
        w = vector[self.n_theta: self.n_theta + self.n_qubits]
        return binary_cross_entropy(y, self._head(self.quantum_features(X, theta), w, vector[-1]))

    def _log(self, step: int, X: np.ndarray, y: np.ndarray, loss: float | None = None) -> None:
        """Store the loss and accuracy on the whole training set."""
        used = self.circuit_evaluations
        if loss is None:
            p = self.predict_proba(X)
            self.circuit_evaluations = used        # monitoring is not counted as training cost
            loss = binary_cross_entropy(y, p)
            accuracy = float(np.mean((p >= 0.5) == y))
        else:
            accuracy = float("nan")
        self.history.append({"step": step, "circuits": used, "loss": loss, "accuracy": accuracy})
        if self.verbose:
            print(f"  step {step:4d} | circuits {used:8d} | loss {loss:.4f} | train acc {accuracy:.4f}")

    # -- Adam with parameter-shift gradients --------------------------------
    def _gradient(self, X: np.ndarray, y: np.ndarray):
        """Gradient of the mini-batch loss with respect to theta, w and b."""
        B, P = len(X), self.n_theta
        features = self.quantum_features(X)                      # (B, qubits)
        p = self._head(features)
        error = p - y                                            # dLoss/dz for each sample

        # Parameter-shift rule: d<Z>/dtheta_k = [ <Z>(theta_k + pi/2) - <Z>(theta_k - pi/2) ] / 2
        shifts = np.vstack([np.eye(P), -np.eye(P)]) * (np.pi / 2)             # (2P, P)
        thetas = self.theta + shifts                                           # (2P, P)
        rows = np.hstack([np.tile(X, (2 * P, 1)), np.repeat(thetas, B, axis=0)])
        shifted = self._expectations(rows).reshape(2 * P, B, self.n_qubits)
        d_features = (shifted[:P] - shifted[P:]) / 2.0                         # (P, B, qubits)

        # Chain rule: dLoss/dtheta_k = mean_b [ error_b * (w . d<Z>_b/dtheta_k) ]
        grad_theta = np.mean(error * (d_features @ self.w), axis=1)
        grad_w = (error[:, None] * features).mean(axis=0)
        grad_b = error.mean()
        return np.concatenate([grad_theta, grad_w, [grad_b]]), binary_cross_entropy(y, p)

    def _fit_adam(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed + 1)
        vector = self._pack()
        m = np.zeros_like(vector)                 # running mean of the gradient
        v = np.zeros_like(vector)                 # running mean of the squared gradient
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._log(0, X, y)
        for step in range(1, self.steps + 1):
            batch = rng.choice(len(X), size=min(self.batch_size, len(X)), replace=False)
            gradient, _ = self._gradient(X[batch], y[batch])
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            # the step size shrinks linearly so that late updates only fine-tune
            progress = (step - 1) / max(self.steps - 1, 1)
            lr = self.learning_rate * (1.0 - (1.0 - self.final_lr_fraction) * progress)
            vector = vector - lr * m_hat / (np.sqrt(v_hat) + eps)
            self._unpack(vector)
            if step % self.log_every == 0 or step == self.steps:
                self._log(step, X, y)

    # -- COBYLA: gradient-free, full training set ---------------------------
    def _fit_cobyla(self, X: np.ndarray, y: np.ndarray) -> None:
        counter = {"n": 0}

        def objective(vector: np.ndarray) -> float:
            loss = self._loss_of(vector, X, y)
            counter["n"] += 1
            self.history.append({"step": counter["n"], "circuits": self.circuit_evaluations,
                                 "loss": loss, "accuracy": float("nan")})
            return loss

        result = minimize(objective, self._pack(), method="COBYLA",
                          options={"maxiter": self.steps, "rhobeg": 0.5})
        self._unpack(result.x)

    # -- SPSA: two loss evaluations per step, whatever the number of parameters
    def _fit_spsa(self, X: np.ndarray, y: np.ndarray, a: float = 5.0, c: float = 0.2) -> None:
        rng = np.random.default_rng(self.seed + 1)
        vector = self._pack()
        stability = 0.1 * self.steps
        for step in range(1, self.steps + 1):
            a_k = a / (step + stability) ** 0.602          # step size (Spall's gain sequence)
            c_k = c / step ** 0.101                        # perturbation size
            delta = rng.choice([-1.0, 1.0], size=len(vector))
            loss_plus = self._loss_of(vector + c_k * delta, X, y)
            loss_minus = self._loss_of(vector - c_k * delta, X, y)
            vector = vector - a_k * (loss_plus - loss_minus) / (2 * c_k) * delta
            self.history.append({"step": step, "circuits": self.circuit_evaluations,
                                 "loss": (loss_plus + loss_minus) / 2, "accuracy": float("nan")})
        self._unpack(vector)

    # ------------------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        """Trainable values: circuit angles + classical weights + bias."""
        return self.n_theta + self.n_qubits + 1
