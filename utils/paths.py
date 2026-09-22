"""Canonical repository paths used by PEACE scripts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
COLOCATIONS_DIR = DATA_ROOT / "colocations"
BASELINE_METRICS_DIR = DATA_ROOT / "baseline_metrics"
MODEL_DATASETS_DIR = DATA_ROOT / "model_datasets"
PREDICTIONS_DIR = REPO_ROOT / "artifacts" / "predictions"
BASELINE_EVALUATIONS_DIR = REPO_ROOT / "artifacts" / "baseline_evaluations"
PAPER_RESULTS_DIR = REPO_ROOT / "results" / "paper"
