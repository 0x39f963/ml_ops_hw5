import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "iris_rf.joblib"
DATA_PATH = ROOT / "data" / "processed" / "iris_clean.csv"
METRICS_PATH = ROOT / "artifacts" / "reports" / "metrics.json"
RUN_PATH = ROOT / "artifacts" / "mlflow" / "mlflow_run.json"
MLRUNS_DIR = ROOT / "mlruns"


def main() -> None:
    RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature

        df = pd.read_csv(DATA_PATH)
        X = df.drop(columns=["species_id", "species"])
        model = joblib.load(MODEL_PATH)

        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        mlflow.set_experiment("hw5_iris_reproducibility")

        signature = infer_signature(X.head(), model.predict(X.head()))
        with mlflow.start_run(run_name="iris_rf_dvc_demo") as run:
            mlflow.log_params(
                {
                    "model": "RandomForestClassifier",
                    "seed": metrics["seed"],
                    "n_estimators": 80,
                    "max_depth": 4,
                }
            )
            mlflow.log_metrics(
                {
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                }
            )
            mlflow.log_artifact(str(ROOT / "dvc.yaml"))
            mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)
            run_info = {
                "mode": "real_mlflow",
                "tracking_uri": mlflow.get_tracking_uri(),
                "run_id": run.info.run_id,
                "metrics": metrics,
            }
    except Exception as exc:
        run_info = {
            "mode": "fallback_json",
            "tracking_uri": str(MLRUNS_DIR),
            "run_name": "iris_rf_dvc_demo",
            "metrics": metrics,
            "note": f"mlflow was not used here: {type(exc).__name__}: {exc}",
        }

    RUN_PATH.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(run_info)


if __name__ == "__main__":
    main()
