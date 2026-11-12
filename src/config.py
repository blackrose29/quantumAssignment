"""Central place for every setting used by the project.

Keeping the numbers here (instead of scattering them through the code) means
the report, the scripts and the notebook all describe the same experiment.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"

DATA_FILE = DATA_DIR / "data_banknote_authentication.txt"
# Official download location (UCI Machine Learning Repository, dataset id 267)
DATA_URL = "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
FEATURE_NAMES = ["variance", "skewness", "curtosis", "entropy"]
# The UCI file stores the label as 0/1 without naming the classes. Following the
# usual reading of this dataset, class 0 is treated as genuine and class 1 as
# forged. "Forged" is the positive class for precision, recall and F1.
CLASS_NAMES = ["Genuine", "Forged"]
TEST_SIZE = 0.20          # 80 % training, 20 % testing (stratified)
SHOWCASE_SEED = 42        # split used for the confusion matrix and curves

# ----------------------------------------------------------------------------
# Hybrid quantum-classical model
# ----------------------------------------------------------------------------
N_QUBITS = 4              # one qubit per feature
N_LAYERS = 3              # number of [encode -> rotate -> entangle] blocks
TRAIN_STEPS = 300         # Adam updates
BATCH_SIZE = 32           # banknotes per update
LEARNING_RATE = 0.1       # Adam step size at the start of training
FINAL_LR_FRACTION = 0.1   # step size falls linearly to 10 % of that by the end
LOG_EVERY = 10            # record the full training loss every 10 steps
