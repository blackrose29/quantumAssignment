# Hybrid Quantum-Classical Banknote Authentication

**Course:** NV62010 Quantum Computing with Artificial Intelligence
**Assignment:** Comprehensive Study and Development of a Hybrid Quantum Machine Learning Model
**Student:** Nausheen Parween (System ID 2023410855)

This repository contains the full assignment package:

- the technical report (DOCX and PDF);
- the source code of a hybrid quantum-classical classifier, built with Qiskit, that separates genuine from forged banknotes;
- the experiments comparing it with five classical machine-learning models;
- all generated figures and result files;
- a viva preparation guide.

> **Important:** all quantum circuits were **simulated** with Qiskit Aer on a CPU. No real quantum hardware was used. The results do **not** demonstrate a quantum advantage. On this dataset the hybrid model is competitive with, but not better than, simple classical models, and it is far more expensive to train (see Section 5 of the report).

---

## 1. Project overview

| Item | Description |
|---|---|
| Problem | Classify a banknote as genuine (0) or forged (1) from four wavelet-image features |
| Dataset | UCI Banknote Authentication (ID 267): 1,372 notes, 4 features, 762 genuine / 610 forged |
| Quantum model | 4-qubit, 3-layer variational circuit: `RY` data re-uploading, trainable `RY`, CNOT chain; `<Z>` on every qubit |
| Classical part | Logistic output layer `p = sigmoid(w·z + b)`; Adam optimiser; parameter-shift gradients |
| Trainable values | 17 (12 circuit angles + 4 weights + 1 bias) |
| Baselines | Logistic regression, SVM (RBF), k-NN, random forest, 4-4-1 neural network (scikit-learn) |
| Evaluation | 10 random stratified 80/20 splits; accuracy, precision, recall, F1, ROC-AUC, MCC, confusion matrix, time, circuit executions |
| Extra studies | Depth / entanglement / re-uploading ablation; Adam vs COBYLA vs SPSA; learning curve; shot noise; gate and read-out noise |

## 2. Folder structure

```text
mlAssignment/
├── main.py                          # one complete run on one split (about 1 minute)
├── requirements.txt                 # pinned package versions
├── data/
│   └── data_banknote_authentication.txt   # UCI dataset (downloaded automatically if missing)
├── src/
│   ├── config.py                    # all settings in one place
│   ├── data_preprocessing.py        # loading, stratified split, AngleScaler
│   ├── quantum_circuit.py           # the parameterised circuit and observables
│   ├── hybrid_model.py              # HybridQuantumClassifier (parameter shift, Adam/COBYLA/SPSA, shots, noise)
│   ├── classical_models.py          # scikit-learn baselines
│   ├── evaluation.py                # metrics
│   ├── quantum_algorithms.py        # Section 2 demos: Bell state, Deutsch-Jozsa, Grover, Shor (N = 15)
│   └── diagrams.py                  # architecture, flowchart, use-case and other diagrams
├── experiments/
│   ├── run_experiments.py           # every study of Section 5 (parallel, resumable)
│   └── make_figures.py              # builds result tables and figures from results/*.json
├── notebooks/
│   └── hybrid_qml_banknote.ipynb    # step-by-step walkthrough (Colab / Jupyter)
├── results/                         # raw and aggregated JSON results, Markdown tables, summary.json
├── figures/                         # all figures used in the report (PNG, 200 dpi)
└── docs/
    └── dataset.md                   # dataset source and download instructions
```

## 3. Prerequisites

- Python 3.10 or newer (tested with 3.13.18)
- About 1 GB of free disk space for the packages
- No GPU or quantum hardware needed

## 4. Installation

```bash
git clone https://github.com/blackrose29/quantumAssignment.git
```

```bash
cd quantumAssignment
```

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

**Google Colab:** open `notebooks/hybrid_qml_banknote.ipynb` and run its first cell. It clones the project and installs the packages.

## 5. How to run

| Command | What it does | Time (4-core CPU) |
|---|---|---|
| `python main.py` | Trains all models on one split (seed 42), prints a results table, saves `results/main_run_*.{csv,json}` and `figures/main_*.png` | about 1 min |
| `python main.py --layers 2 --steps 200 --seed 7` | Same with other settings | — |
| `python -m src.quantum_algorithms` | Section 2 demonstrations; saves circuit drawings and `results/section2_demos.json` | about 20 s |
| `python -m src.diagrams` | Redraws the architecture, flowchart, use-case and theory diagrams | about 15 s |
| `python -m experiments.run_experiments` | Full study (71 tasks): 10-split comparison, ablation, optimisers, learning curve, shots and noise | about 25 min |
| `python -m experiments.run_experiments --quick` | Same with 2 seeds per study, as a quick check | about 6 min |
| `python -m experiments.make_figures` | Builds every Section 5 table and figure from `results/*.json` | about 10 s |
| `python3 report/build/build_report.py` | Rebuilds the report DOCX and PDF (needs pandoc, LibreOffice incl. Math, pdftotext, python-docx) | about 30 s |

`run_experiments.py` saves each finished task to `results/raw/`. If it is interrupted, running it again continues where it stopped.

## 6. Expected outputs

`python main.py` prints the per-step training loss, then a table like the one below. Exact numbers depend on the seed; the values shown are from the actual run with seed 42 that is reported in the report.

```text
Test-set results (percent):
                                 accuracy  precision  recall      f1  roc_auc  train_seconds
model                                                                                       
Logistic Regression                 97.09      93.85   100.0   96.83    99.99          0.005
SVM (RBF kernel)                   100.00     100.00   100.0  100.00   100.00          0.005
k-Nearest Neighbours               100.00     100.00   100.0  100.00   100.00          0.002
Random Forest                       99.64      99.19   100.0   99.59   100.00          0.147
Neural Network (4-4-1)              97.82      95.31   100.0   97.60   100.00          0.411
Hybrid VQC (4 qubits, 3 layers)     98.91      97.60   100.0   98.79   100.00         61.296

Circuit executions used for training: 240,000
```

The headline results of the full study (mean ± standard deviation over 10 random splits) are:

| Model | Accuracy (%) | F1 (%) | Training time per split |
|---|---|---|---|
| Logistic regression | 98.11 ± 0.89 | 97.91 ± 0.97 | 0.06 s |
| SVM (RBF kernel) | 100.00 ± 0.00 | 100.00 ± 0.00 | 0.008 s |
| k-nearest neighbours | 99.93 ± 0.15 | 99.92 ± 0.17 | 0.003 s |
| Random forest | 99.31 ± 0.36 | 99.22 ± 0.41 | 0.17 s |
| Neural network (4-4-1) | 98.62 ± 1.03 | 98.47 ± 1.13 | 0.50 s |
| **Hybrid VQC (4 qubits, 3 layers)** | **99.05 ± 0.46** | **98.94 ± 0.52** | **66 s (240,000 circuit executions)** |

Other findings (see `results/tables/`):

- Removing the CNOTs drops accuracy to 80.7 %.
- Encoding the data once instead of re-uploading it gives 99.78 %.
- Adam with parameter shift beats COBYLA (96.7 %) and SPSA (95.7 %).
- About 256 shots per circuit are enough to match exact accuracy.
- Accuracy stays at about 98 % or more with up to 5 % gate and read-out error.

## 7. Dataset

UCI Machine Learning Repository, *Banknote Authentication* (V. Lohweg, 2012), DOI [10.24432/C55P57](https://doi.org/10.24432/C55P57), licence CC BY 4.0. See [`docs/dataset.md`](docs/dataset.md) for details and manual download instructions. The code downloads the file automatically if `data/data_banknote_authentication.txt` is missing.

## 8. Reproducibility notes

- All random choices are seeded: splits use seeds 0–9 (42 for the detailed run), and the circuit initialisation, mini-batches and shot sampling are seeded as well.
- Training uses exact expectation values from `qiskit_aer.primitives.EstimatorV2`, which corresponds to infinitely many shots. The shot and noise studies use `SamplerV2`, with an optional `NoiseModel`.
- Timings depend on the machine. The report's timings were measured on a 4-core Linux cloud container with 15 GB of RAM.

## 9. Report

It contains the seven prescribed sections:

1. Introduction and synopsis
2. Fundamentals of quantum computing and algorithms (with Qiskit demonstrations)
3. Critical review of 50 research articles, with a comparative matrix
4. Mini-project design
5. Results
6. Conclusion and future scope
7. IEEE references

The contents lists and page numbers were generated from the PDF layout. In Word, pagination can differ slightly from the PDF.
