from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "iris.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "iris_clean.csv"


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species_id"})
    df["species"] = df["species_id"].map(dict(enumerate(iris.target_names)))

    df.to_csv(RAW_PATH, index=False)

    clean_df = df.drop_duplicates().dropna().reset_index(drop=True)
    clean_df.to_csv(CLEAN_PATH, index=False)

    print(
        {
            "raw_path": str(RAW_PATH),
            "clean_path": str(CLEAN_PATH),
            "rows": len(clean_df),
            "columns": len(clean_df.columns),
        }
    )


if __name__ == "__main__":
    main()
