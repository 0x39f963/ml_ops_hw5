import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "artifacts" / "images"
METRICS_PATH = ROOT / "artifacts" / "reports" / "metrics.json"
MLFLOW_PATH = ROOT / "artifacts" / "mlflow" / "mlflow_run.json"
FEATURE_YAML = ROOT / "artifacts" / "feature_store" / "feature_store.yaml"
VIDEO_STATS = ROOT / "artifacts" / "reports" / "video_processing_stats.json"


def save_card(lines: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    ax.text(0.03, 0.92, title, fontsize=17, weight="bold", color="#1d3557")
    y = 0.78
    for line in lines:
        ax.text(0.05, y, line, fontsize=12, family="monospace", color="#222222")
        y -= 0.1
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    save_card(
        [
            f"accuracy  = {metrics['accuracy']}",
            f"f1_macro  = {metrics['f1_macro']}",
            f"test_rows = {metrics['test_rows']}",
            f"seed      = {metrics['seed']}",
            f"model     = {metrics['model']}",
        ],
        IMAGE_DIR / "01_metrics_card.png",
        "HW5 Iris model metrics",
    )

    save_card(
        [
            "prepare: sklearn Iris -> data/raw/iris.csv",
            "prepare: raw csv -> data/processed/iris_clean.csv",
            "train:   clean csv -> RandomForestClassifier",
            "train:   model -> models/iris_rf.joblib",
            "train:   metrics -> artifacts/reports/metrics.json",
        ],
        IMAGE_DIR / "02_dvc_pipeline_card.png",
        "DVC pipeline: 2 stages",
    )

    mlflow_run = json.loads(MLFLOW_PATH.read_text(encoding="utf-8"))
    save_card(
        [
            f"mode         = {mlflow_run['mode']}",
            "experiment   = hw5_iris_reproducibility",
            "run_name     = iris_rf_dvc_demo",
            f"accuracy     = {metrics['accuracy']}",
            f"f1_macro     = {metrics['f1_macro']}",
        ],
        IMAGE_DIR / "03_mlflow_card.png",
        "MLflow run summary",
    )

    feature_config = yaml.safe_load(FEATURE_YAML.read_text(encoding="utf-8"))
    save_card(
        [
            f"project      = {feature_config['project']}",
            f"provider     = {feature_config['provider']}",
            f"registry     = {feature_config['registry']}",
            f"online_store = {feature_config['online_store']['type']}",
            f"path         = {feature_config['online_store']['path']}",
        ],
        IMAGE_DIR / "04_feature_store_card.png",
        "Feast local Feature Store",
    )

    video_stats = json.loads(VIDEO_STATS.read_text(encoding="utf-8"))
    save_card(
        [
            f"source frames      = {video_stats['frame_count']}",
            f"processed frames   = {video_stats['processed_frames']}",
            f"frames with faces  = {video_stats['frames_with_faces']}",
            f"total detections   = {video_stats['total_faces']}",
            f"resolution         = {video_stats['width']}x{video_stats['height']}",
            f"fps                = {video_stats['fps']}",
        ],
        IMAGE_DIR / "05_video_processing_card.png",
        "OpenCV face blur on real HW5 video",
    )
    print("report assets saved to", IMAGE_DIR)


if __name__ == "__main__":
    main()
