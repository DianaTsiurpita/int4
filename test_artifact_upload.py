import mlflow
from pathlib import Path

mlflow.set_tracking_uri("http://localhost:5000")

client = mlflow.tracking.MlflowClient()
run_id = "095bad146b8049ec835fbd92fef3ce1c"

stats_path = Path("/tmp/data_stats_test.json")
stats_path.write_text('{"test": true}', encoding="utf-8")

client.log_artifact(run_id, str(stats_path))

print("Artifact upload test: SUCCESS")
