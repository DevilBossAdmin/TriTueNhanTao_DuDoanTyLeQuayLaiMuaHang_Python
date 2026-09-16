from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

DATA_PATH = DATA_DIR / "customers.csv"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = RESULTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"

TARGET = "returned_90d"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
