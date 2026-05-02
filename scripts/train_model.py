import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "iris_clean.csv"
MODEL_PATH = ROOT / "models" / "iris_rf.joblib"
METRICS_PATH = ROOT / "artifacts" / "reports" / "metrics.json"
SEED = 42


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["species_id", "species"])
    y = df["species_id"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=4,
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
        "test_rows": int(len(y_test)),
        "seed": SEED,
        "model": "RandomForestClassifier",
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics)


if __name__ == "__main__":
    main()
