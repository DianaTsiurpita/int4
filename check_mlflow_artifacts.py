import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

versions = client.search_model_versions("name = 'house-price'")

print("Found versions:", len(versions))

for version in sorted(versions, key=lambda v: int(v.version)):
    print("\nMODEL VERSION:", version.version)
    print("RUN ID:", version.run_id)

    for artifact in client.list_artifacts(version.run_id):
        print("ARTIFACT:", artifact.path)
