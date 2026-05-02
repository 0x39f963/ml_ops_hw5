from datetime import timedelta
from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Int64

iris = Entity(name="iris", join_keys=["iris_id"])

iris_source = FileSource(
    name="iris_source",
    path="/home/x39963/web/niki/mo-dz/DP/HomeWork_5_ML_Ops_Novik_F1/artifacts/feature_store/iris_features.parquet",
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
