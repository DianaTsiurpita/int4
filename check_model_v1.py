import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

client = mlflow.tracking.MlflowClient()
run = client.get_run("095bad146b8049ec835fbd92fef3ce1c")

print("STATUS:", run.info.status)
print("ARTIFACT URI:", run.info.artifact_uri)
print("PARAMS:", run.data.params)
print("TAGS:", run.data.tags)
