"""Full experimental study reported in Section 5 of the report.

    python -m experiments.run_experiments            # everything (about 25 min on 4 CPU cores)
    python -m experiments.run_experiments --quick    # 2 seeds per study, for a fast check

Studies
    main            hybrid model vs five classical baselines, 10 random stratified splits
    showcase        one fixed split (seed 42): confusion matrices, ROC data, training curve,
                    finite-shot and noise study on the trained circuit
    ablation        number of layers, no entanglement, no data re-uploading (5 splits)
    optimizer       Adam + parameter shift vs COBYLA vs SPSA (5 splits)
    learning_curve  accuracy against the number of training notes (5 splits)

Every task is independent, so the tasks are spread over the CPU cores. All
numbers are written to results/*.json; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src import config
from src.classical_models import build_baselines, count_parameters
from src.data_preprocessing import AngleScaler, get_arrays, load_banknote, make_split
from src.evaluation import compute_metrics, model_outputs
from src.hybrid_model import HybridQuantumClassifier
from src.quantum_circuit import circuit_summary

SHOTS_GRID = [16, 64, 256, 1024, 4096]
NOISE_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]      # two-qubit gate error rate
TRAIN_SIZES = [20, 50, 100, 200, 500]


def _data(seed: int):
    X, y = get_arrays(load_banknote())
    return make_split(X, y, seed)


def _hybrid_metrics(model: HybridQuantumClassifier, A_test, y_test) -> dict:
    t0 = time.perf_counter()
    prob = model.predict_proba(A_test)
    predict_seconds = time.perf_counter() - t0
    return {**compute_metrics(y_test, (prob >= 0.5).astype(int), prob),
            "train_seconds": model.fit_seconds, "predict_seconds": predict_seconds,
            "parameters": model.n_parameters, "circuits": model.circuit_evaluations}


def _train_hybrid(seed, X_train, y_train, X_test, **kwargs):
    scaler = AngleScaler().fit(X_train)
    model = HybridQuantumClassifier(seed=seed, **kwargs).fit(scaler.transform(X_train), y_train)
    return model, scaler.transform(X_test)


# ---------------------------------------------------------------------------
# Tasks (each returns a JSON-serialisable dictionary)
# ---------------------------------------------------------------------------
def task_main(seed: int) -> dict:
    X_train, X_test, y_train, y_test = _data(seed)
    out = {"study": "main", "seed": seed, "models": {}}
    for name, model in build_baselines(seed).items():
        t0 = time.perf_counter(); model.fit(X_train, y_train); train_seconds = time.perf_counter() - t0
        t0 = time.perf_counter(); y_pred, score = model_outputs(model, X_test); predict_seconds = time.perf_counter() - t0
        out["models"][name] = {**compute_metrics(y_test, y_pred, score), "train_seconds": train_seconds,
                               "predict_seconds": predict_seconds, "parameters": count_parameters(name, model)}
    hybrid, A_test = _train_hybrid(seed, X_train, y_train, X_test)
    out["models"]["Hybrid VQC"] = _hybrid_metrics(hybrid, A_test, y_test)
    out["history"] = hybrid.history
    return out


def task_showcase(seed: int) -> dict:
    """One fixed split with everything needed for the detailed figures."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    X_train, X_test, y_train, y_test = _data(seed)
    out = {"study": "showcase", "seed": seed, "y_test": y_test.tolist(), "models": {}, "scores": {}}
    for name, model in build_baselines(seed).items():
        model.fit(X_train, y_train)
        y_pred, score = model_outputs(model, X_test)
        out["models"][name] = compute_metrics(y_test, y_pred, score)
        out["scores"][name] = np.asarray(score).tolist()

    hybrid, A_test = _train_hybrid(seed, X_train, y_train, X_test)
    exact_features = hybrid.quantum_features(A_test)
    exact_prob = hybrid.predict_proba_from_features(exact_features)
    out["models"]["Hybrid VQC"] = _hybrid_metrics(hybrid, A_test, y_test)
    out["scores"]["Hybrid VQC"] = exact_prob.tolist()
    out["history"] = hybrid.history
    out["theta"] = hybrid.theta.tolist(); out["w"] = hybrid.w.tolist(); out["b"] = hybrid.b
    out["circuit"] = circuit_summary(hybrid.circuit)
    out["test_features"] = exact_features.tolist()

    def evaluate(features):
        prob = hybrid.predict_proba_from_features(features)
        return {"accuracy": float(np.mean((prob >= 0.5) == y_test)),
                "feature_error": float(np.mean(np.abs(features - exact_features))),
                "prob_error": float(np.mean(np.abs(prob - exact_prob)))}

    # (a) finite number of shots, no gate noise: 10 repetitions per setting
    out["shots"] = []
    for shots in SHOTS_GRID:
        runs = [evaluate(hybrid.sampled_features(A_test, shots, seed=100 + r)) for r in range(10)]
        out["shots"].append({"shots": shots, "runs": runs})

    # (b) gate and read-out noise at 1024 shots: 5 repetitions per setting
    out["noise"] = []
    for p2 in NOISE_GRID:
        noise_model = None
        if p2 > 0:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["ry"])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
            noise_model.add_all_qubit_readout_error(ReadoutError([[1 - p2, p2], [p2, 1 - p2]]))
        runs = [evaluate(hybrid.sampled_features(A_test, 1024, noise_model, seed=200 + r)) for r in range(5)]
        out["noise"].append({"p2": p2, "runs": runs})
    return out


def task_ablation(variant: str, seed: int) -> dict:
    settings = {
        "1 layer": dict(n_layers=1), "2 layers": dict(n_layers=2),
        "3 layers": dict(n_layers=3), "4 layers": dict(n_layers=4),
        "3 layers, no entanglement": dict(n_layers=3, entangle=False),
        "3 layers, no re-uploading": dict(n_layers=3, reupload=False),
    }[variant]
    X_train, X_test, y_train, y_test = _data(seed)
    hybrid, A_test = _train_hybrid(seed, X_train, y_train, X_test, **settings)
    return {"study": "ablation", "variant": variant, "seed": seed,
            **_hybrid_metrics(hybrid, A_test, y_test), "final_loss": hybrid.history[-1]["loss"],
            "circuit": circuit_summary(hybrid.circuit)}


def task_optimizer(optimizer: str, seed: int) -> dict:
    steps = {"adam": config.TRAIN_STEPS, "cobyla": 600, "spsa": 300}[optimizer]
    X_train, X_test, y_train, y_test = _data(seed)
    hybrid, A_test = _train_hybrid(seed, X_train, y_train, X_test, optimizer=optimizer, steps=steps)
    history = hybrid.history
    if optimizer == "cobyla":            # keep every 5th evaluation to limit file size
        history = history[::5] + history[-1:]
    scaler = AngleScaler().fit(X_train)
    train_prob = hybrid.predict_proba(scaler.transform(X_train))
    final_loss = float(-np.mean(y_train * np.log(np.clip(train_prob, 1e-7, 1)) +
                                (1 - y_train) * np.log(np.clip(1 - train_prob, 1e-7, 1))))
    metrics = _hybrid_metrics(hybrid, A_test, y_test)
    return {"study": "optimizer", "optimizer": optimizer, "seed": seed, **metrics,
            "final_loss": final_loss, "history": history}


def task_learning_curve(size: int, seed: int) -> dict:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = _data(seed)
    X_small, _, y_small, _ = train_test_split(X_train, y_train, train_size=size, stratify=y_train,
                                              random_state=seed)
    out = {"study": "learning_curve", "size": size, "seed": seed, "models": {}}
    for name, model in build_baselines(seed).items():
        if name in ("Logistic Regression", "SVM (RBF kernel)"):
            model.fit(X_small, y_small)
            y_pred, score = model_outputs(model, X_test)
            out["models"][name] = compute_metrics(y_test, y_pred, score)
    hybrid, A_test = _train_hybrid(seed, X_small, y_small, X_test)
    out["models"]["Hybrid VQC"] = _hybrid_metrics(hybrid, A_test, y_test)
    return out


def _dispatch(task):
    kind, args = task[0], task[1:]
    return {"main": task_main, "showcase": task_showcase, "ablation": task_ablation,
            "optimizer": task_optimizer, "learning_curve": task_learning_curve}[kind](*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="2 seeds per study instead of 5-10")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main_seeds = range(2) if args.quick else range(10)
    seeds = range(2) if args.quick else range(5)
    tasks = [("showcase", config.SHOWCASE_SEED)]
    tasks += [("optimizer", opt, s) for opt in ("cobyla", "spsa") for s in seeds]
    tasks += [("main", s) for s in main_seeds]
    tasks += [("ablation", v, s) for v in ("1 layer", "2 layers", "4 layers", "3 layers, no entanglement",
                                           "3 layers, no re-uploading") for s in seeds]
    tasks += [("learning_curve", n, s) for n in TRAIN_SIZES for s in seeds]

    raw_dir = config.RESULT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def raw_path(task):
        return raw_dir / ("__".join(str(part).replace(" ", "_").replace(",", "") for part in task) + ".json")

    # Each finished task is saved at once, so an interrupted run can be resumed.
    pending = [t for t in tasks if not raw_path(t).exists()]
    print(f"{len(tasks) - len(pending)} tasks already done, {len(pending)} to run")
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_dispatch, t): t for t in pending}
        for i, future in enumerate(as_completed(futures), 1):
            raw_path(futures[future]).write_text(json.dumps(future.result()))
            print(f"[{i:3d}/{len(pending)}] {futures[future]}  ({time.time() - start:6.0f} s)", flush=True)

    grouped: dict[str, list] = {}
    for t in tasks:
        r = json.loads(raw_path(t).read_text())
        grouped.setdefault(r["study"], []).append(r)
    for study, items in grouped.items():
        items.sort(key=lambda r: json.dumps({k: r[k] for k in ("variant", "optimizer", "size", "seed") if k in r}))
        (config.RESULT_DIR / f"{study}.json").write_text(json.dumps(items, indent=1))
    print(f"Done in {time.time() - start:.0f} s. Results in {config.RESULT_DIR}")
