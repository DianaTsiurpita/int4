import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

source_run = "3825d56533054f3db20954216efcdff8"
target_run = "095bad146b8049ec835fbd92fef3ce1c"

local_path = client.download_artifacts(
    source_run,
    "data_stats.json"
)

client.log_artifact(target_run, local_path)

print("SUCCESS: data_stats.json copied to model version 1")

print("Version 1 artifacts:")
for artifact in client.list_artifacts(target_run):
    print(artifact.path)
