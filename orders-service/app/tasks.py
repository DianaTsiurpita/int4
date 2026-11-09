"""Durable Redis/RQ pricing worker; separate from Orders API process."""
import logging
import os
import httpx
from tenacity import retry,retry_if_exception_type,stop_after_attempt,wait_exponential
from .database import SessionLocal
from . import crud
logger=logging.getLogger(__name__)
@retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=.3,max=2),
       retry=retry_if_exception_type((httpx.ConnectError,httpx.TimeoutException)),reraise=True)
def fetch(payload):
    resp=httpx.post(os.getenv('FORECAST_URL','http://forecast-api:8001')+'/predict',
                    json=payload,timeout=float(os.getenv('FORECAST_TIMEOUT','5')))
    resp.raise_for_status()
    return resp.json()
def price_order(oid,payload):
    try:
        prediction=fetch(payload)
        with SessionLocal() as db:
            crud.update_prediction(db,oid,float(prediction['predicted_price']),'priced',str(prediction['model_version']))
    except Exception:
        logger.exception('Prediction failed for order %s',oid)
        with SessionLocal() as db:
            crud.update_prediction(db,oid,None,'failed',None)
