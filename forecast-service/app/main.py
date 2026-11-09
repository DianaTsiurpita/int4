"""Registry-driven forecast serving; no training data or training code."""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
from fastapi import FastAPI
from .schemas import PredictRequest,PredictResponse
logger=logging.getLogger('forecast-api')
NAME=os.getenv('MODEL_NAME','house-price')
ALIAS=os.getenv('MODEL_ALIAS','prod')
FEATURES=['GrLivArea','TotalBsmtSF','GarageArea','YearBuilt','OverallQual','MSZoning']
@asynccontextmanager
async def lifespan(app):
    try:
        mv=MlflowClient().get_model_version_by_alias(NAME,ALIAS)
        app.state.model=mlflow.pyfunc.load_model(f'models:/{NAME}@{ALIAS}')
        app.state.model_version=str(mv.version)
        stats_path=mlflow.artifacts.download_artifacts(run_id=mv.run_id,artifact_path='data_stats.json')
        app.state.data_stats=json.loads(Path(stats_path).read_text(encoding='utf-8'))
    except Exception as exc:
        logger.exception('Cannot resolve registry model %s@%s: %s',NAME,ALIAS,exc)
        raise RuntimeError(f'Model alias {NAME}@{ALIAS} unavailable; run trainer first') from exc
    yield
app=FastAPI(title='Forecast API',lifespan=lifespan)
@app.get('/health')
def health():return {'status':'ok','model_version':app.state.model_version}
@app.get('/model/data-stats')
def data_stats():return app.state.data_stats
@app.post('/predict',response_model=PredictResponse)
def predict(req:PredictRequest):
    t=time.perf_counter()
    df=pd.DataFrame([dict(GrLivArea=req.gr_liv_area,TotalBsmtSF=req.total_bsmt_sf,
       GarageArea=req.garage_area,YearBuilt=req.year_built,OverallQual=req.overall_qual,MSZoning=req.ms_zoning)])
    for feature in ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'YearBuilt', 'OverallQual']:
        df[feature] = df[feature].astype('int64')

    price=float(app.state.model.predict(df[FEATURES])[0])
    return PredictResponse(predicted_price=price,model_version=app.state.model_version,
                           latency_ms=round((time.perf_counter()-t)*1000,2))
