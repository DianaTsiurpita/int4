import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from mlflow.tracking import MlflowClient

from app.database import SessionLocal
from app.models import Order

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_MAP = {
    "gr_liv_area": "GrLivArea",
    "total_bsmt_sf": "TotalBsmtSF",
    "garage_area": "GarageArea",
    "year_built": "YearBuilt",
    "overall_qual": "OverallQual",
    "ms_zoning": "MSZoning",
}

NUMERIC = [
    "GrLivArea", "TotalBsmtSF", "GarageArea",
    "YearBuilt", "OverallQual",
]
CATEGORICAL = ["MSZoning"]


def run_monitoring():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", "http://mlflow:5000"
    )
    model_name = os.getenv("MODEL_NAME", "house-price")
    model_alias = os.getenv("MODEL_ALIAS", "prod")
    threshold = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
    window_hours = float(os.getenv("MONITOR_WINDOW_HOURS", "24"))

    if not 0 <= threshold <= 1:
        raise ValueError("DRIFT_THRESHOLD must be between 0 and 1")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model = client.get_model_version_by_alias(
        model_name, model_alias
    )
    logger.info(
        "Monitoring model %s version %s",
        model_name, model.version
    )

    reference_path = mlflow.artifacts.download_artifacts(
        run_id=model.run_id,
        artifact_path="reference_data.parquet",
    )
    reference = pd.read_parquet(reference_path)

    since = datetime.now(timezone.utc) - timedelta(
        hours=window_hours
    )

    db = SessionLocal()
    try:
        rows = (
            db.query(Order)
            .filter(Order.created_at >= since)
            .order_by(Order.created_at.desc())
            .all()
        )
        records = [
            {
                source: getattr(order, source)
                for source in FEATURE_MAP
            }
            for order in rows
        ]
    finally:
        db.close()

    if not records:
        logger.warning(
            "No orders in the last %s hours; monitoring skipped",
            window_hours,
        )
        return {"status": "no_data", "orders": 0}

    current = pd.DataFrame(records).rename(
        columns=FEATURE_MAP
    )
    columns = NUMERIC + CATEGORICAL

    missing = set(columns) - set(reference.columns)
    if missing:
        raise ValueError(
            f"Reference data missing columns: {sorted(missing)}"
        )

    reference = reference[columns].copy()
    current = current[columns].copy()

    for column in NUMERIC:
        reference[column] = pd.to_numeric(
            reference[column], errors="coerce"
        )
        current[column] = pd.to_numeric(
            current[column], errors="coerce"
        )

    for column in CATEGORICAL:
        reference[column] = reference[column].astype("string")
        current[column] = current[column].astype("string")

    definition = DataDefinition(
        numerical_columns=NUMERIC,
        categorical_columns=CATEGORICAL,
    )

    report = Report([DataDriftPreset()])
    snapshot = report.run(
        reference_data=Dataset.from_pandas(
            reference, data_definition=definition
        ),
        current_data=Dataset.from_pandas(
            current, data_definition=definition
        ),
    )

    metrics = snapshot.dict().get("metrics", [])
    drift_metric = next(
        (
            metric
            for metric in metrics
            if metric.get("metric_id", "").startswith(
                "DriftedColumnsCount("
            )
        ),
        None,
    )
    if drift_metric is None:
        raise RuntimeError(
            "Evidently did not return DriftedColumnsCount"
        )

    drift_count = int(drift_metric["value"]["count"])
    drift_share = float(drift_metric["value"]["share"])
    exceeded = drift_share > threshold

    mlflow.set_experiment("monitoring")
    with mlflow.start_run(
        run_name=f"data-drift-model-v{model.version}"
    ) as run:
        mlflow.log_params({
            "model_name": model_name,
            "model_version": str(model.version),
            "model_alias": model_alias,
            "reference_run_id": model.run_id,
            "window_hours": window_hours,
            "drift_threshold": threshold,
        })
        mlflow.log_metrics({
            "drift_share": drift_share,
            "drifted_features": drift_count,
            "total_features": len(columns),
            "current_rows": len(current),
            "reference_rows": len(reference),
        })
        mlflow.set_tags({
            "monitoring_status": (
                "warning" if exceeded else "ok"
            ),
            "drift_threshold_exceeded": str(exceeded).lower(),
        })

        with tempfile.TemporaryDirectory() as directory:
            html_path = Path(directory) / "data_drift.html"
            json_path = Path(directory) / "data_drift.json"

            snapshot.save_html(str(html_path))
            snapshot.save_json(str(json_path))

            mlflow.log_artifact(
                str(html_path), artifact_path="reports"
            )
            mlflow.log_artifact(
                str(json_path), artifact_path="reports"
            )

        run_id = run.info.run_id

    if exceeded:
        logger.warning(
            "DATA DRIFT WARNING: share=%.3f exceeds threshold=%.3f",
            drift_share, threshold
        )
    else:
        logger.info(
            "Data drift share=%.3f, threshold=%.3f",
            drift_share, threshold
        )

    return {
        "status": "warning" if exceeded else "ok",
        "model_version": str(model.version),
        "orders": len(current),
        "drift_share": drift_share,
        "drifted_features": drift_count,
        "threshold": threshold,
        "mlflow_run_id": run_id,
    }


if __name__ == "__main__":
    print(run_monitoring())
