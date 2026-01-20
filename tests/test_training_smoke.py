import json
from pathlib import Path
import subprocess
import sys

def test_smoke_train_runs():
    # Run the smoke training module
    subprocess.check_call([sys.executable, "-m", "src.training.smoke_train"])

    metrics_path = Path("artifacts/smoke_metrics.json")
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert "roc_auc" in metrics and metrics["roc_auc"] >= 0.5
    assert "pr_auc" in metrics and metrics["pr_auc"] >= 0.5

