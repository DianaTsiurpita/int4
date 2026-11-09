import sys
from mlflow.tracking import MlflowClient
if len(sys.argv) != 2: raise SystemExit("Usage: python scripts/model_alias.py VERSION")
client = MlflowClient()
client.set_registered_model_alias("house-price", "prod", sys.argv[1])
print("prod ->", client.get_model_version_by_alias("house-price", "prod").version)
