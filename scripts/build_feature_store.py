from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "iris_clean.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "feature_store"
FEATURE_YAML = ARTIFACT_DIR / "feature_store.yaml"
REPO_PY = ARTIFACT_DIR / "iris_repo.py"
FEATURES_PARQUET = ARTIFACT_DIR / "iris_features.parquet"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    feature_store_text = dedent(
        """
        project: hw5_iris
        project_description: >
          HW5 demo feature store for Iris model.
          Local-вариант из семинара: registry в файле + sqlite online store.
        registry: data/registry.db
        provider: local
        online_store:
            type: sqlite
            path: data/online_store.db
        entity_key_serialization_version: 3
        auth:
            type: no_auth
        """
    ).strip() + "\n"
    FEATURE_YAML.write_text(feature_store_text, encoding="utf-8")

    parsed = yaml.safe_load(feature_store_text)
    if parsed["provider"] != "local":
        raise ValueError("Feature Store provider should be local for this homework")

    df = pd.read_csv(DATA_PATH)
    df["iris_id"] = np.arange(len(df))
    df["event_timestamp"] = pd.Timestamp.now(tz="UTC").floor("s")
    df.to_parquet(FEATURES_PARQUET, index=False)

    repo_text = dedent(
        f"""
        from datetime import timedelta
        from feast import Entity, FeatureService, FeatureView, Field, FileSource
        from feast.types import Float32, Int64

        iris = Entity(name="iris", join_keys=["iris_id"])

        iris_source = FileSource(
            name="iris_source",
            path="{FEATURES_PARQUET.as_posix()}",
            timestamp_field="event_timestamp",
        )

        iris_features_view = FeatureView(
            name="iris_features_view",
            entities=[iris],
            ttl=timedelta(days=365),
            schema=[
                Field(name="sepal length (cm)", dtype=Float32),
                Field(name="sepal width (cm)", dtype=Float32),
                Field(name="petal length (cm)", dtype=Float32),
                Field(name="petal width (cm)", dtype=Float32),
                Field(name="species_id", dtype=Int64),
            ],
            source=iris_source,
            online=True,
        )

        iris_service_v1 = FeatureService(
            name="iris_service_v1",
            features=[iris_features_view],
        )
        """
    ).strip() + "\n"
    REPO_PY.write_text(repo_text, encoding="utf-8")
    print({"feature_store_yaml": str(FEATURE_YAML), "repo_py": str(REPO_PY)})


if __name__ == "__main__":
    main()
