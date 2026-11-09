import logging
import os
import asyncio
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import auth, crud, schemas, models
from redis import Redis
from rq import Queue
from .database import Base, SessionLocal, engine, get_db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("orders-api")

FORECAST_URL = os.getenv(
    "FORECAST_URL",
    "http://forecast-api:8001",
)

BATCH_MAX_ITEMS = int(os.getenv("BATCH_MAX_ITEMS", "50"))

FORECAST_TIMEOUT = float(
    os.getenv("FORECAST_TIMEOUT", "3.0")
)

# Tables are created automatically for the initial version.
# In production, replace this with Alembic migrations.
# Importing auth registers the User model before table creation.
# Upgrade databases created by the previous laboratory without dropping data.
with engine.begin() as connection:
    connection.execute(text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'pending'")) if engine.dialect.has_table(connection, "orders") else None
    # Existing rows receive the correct status; new rows are not modified.
    if engine.dialect.has_table(connection, "orders"):
        connection.execute(text("""UPDATE orders SET status = CASE WHEN predicted_price IS NULL THEN 'failed' ELSE 'priced' END WHERE status IS NULL"""))
Base.metadata.create_all(bind=engine)
with engine.begin() as connection:
    connection.execute(text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS model_version VARCHAR(32)"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(FORECAST_TIMEOUT)
    )

    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(
    title="Orders API",
    version="0.1.0",
    lifespan=lifespan,
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, max=2),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.TimeoutException)
    ),
    reraise=True,
)
def _request_prediction(payload: dict) -> dict:
    """Request a price prediction from the forecasting service.

    Retry only transient network failures such as connection errors
    and timeouts. HTTP errors are not retried.
    """
    response = httpx.post(
        f"{FORECAST_URL}/predict",
        json=payload,
        timeout=FORECAST_TIMEOUT,
    )
    response.raise_for_status()

    return response.json()

def update_prediction_separately(oid: int, price: float | None, status: str, model_version: str | None = None) -> None:
    with SessionLocal() as db:
        crud.update_prediction(db, oid, price, status, model_version)


async def request_prediction_async(payload: dict) -> dict:
    """Request a prediction using the shared asynchronous HTTP client."""
    response = await app.state.http_client.post(
        f"{FORECAST_URL}/predict",
        json=payload,
    )

    response.raise_for_status()

    return response.json()

def enqueue_prediction(oid: int, payload: dict) -> None:
    queue = Queue('pricing', connection=Redis.from_url(os.getenv('REDIS_URL','redis://redis:6379/0')))
    queue.enqueue('app.tasks.price_order', oid, payload, job_timeout=120)

@app.post(
    "/orders/batch",
    response_model=schemas.OrderBatchResponse,
)
async def create_orders_batch(
    batch: schemas.OrderBatchCreate,
    db: Session = Depends(get_db),
    current_user: str = Depends(auth.require_user),
):
    """Create orders and request predictions concurrently."""
    if len(batch.items) > BATCH_MAX_ITEMS:
        raise HTTPException(status_code=422, detail=f"Maximum {BATCH_MAX_ITEMS} items allowed")

    orders = [
        crud.create_order(db, item)
        for item in batch.items
    ]

    async def predict_one(index, order, item):
        try:
            result = await request_prediction_async(item.model_dump())
            price = float(result["predicted_price"])
            version = str(result["model_version"])

            await run_in_threadpool(update_prediction_separately, order.id, price, "priced", version)

            return schemas.OrderBatchItemResult(
                index=index,
                order_id=order.id,
                status="priced",
                predicted_price=price,
                model_version=version,
            )

        except Exception as exc:
            logger.exception(
                "Batch prediction failed: order_id=%s",
                order.id,
            )

            await run_in_threadpool(update_prediction_separately, order.id, None, "failed")

            return schemas.OrderBatchItemResult(
                index=index,
                order_id=order.id,
                status="failed",
                predicted_price=None,
                error=str(exc),
            )

    results = await asyncio.gather(
        *(
            predict_one(index, order, item)
            for index, (order, item) in enumerate(zip(orders, batch.items))
        )
    )

    return schemas.OrderBatchResponse(
        total=len(results),
        results=list(results)
    )

@app.post("/auth/register", status_code=201)
def register_user(data: schemas.UserCredentials):
    with SessionLocal() as db:
        existing = db.query(auth.User).filter(
            auth.User.username == data.username
        ).first()

        if existing:
            raise HTTPException(
                status_code=409,
                detail="Username already exists",
            )

        user = auth.User(
            username=data.username,
            password_hash=auth.hash_password(data.password),
        )

        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=409, detail="Username already exists")

    return {"message": "User registered successfully"}


@app.post(
    "/auth/login",
    response_model=schemas.TokenResponse,
)
def login_user(data: schemas.UserCredentials):
    with SessionLocal() as db:
        user = db.query(auth.User).filter(
            auth.User.username == data.username
        ).first()

        if user is None or not auth.verify_password(
            data.password, user.password_hash
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password",
            )

    return schemas.TokenResponse(
        access_token=auth.create_token(data.username),
        token_type="bearer",
    )

@app.get("/health")
def health() -> dict[str, str]:
    """Return the health status of the Orders API."""
    return {"status": "ok"}


@app.post(
    "/orders",
    response_model=schemas.OrderRead,
    status_code=202,
)
def create_order(
    data: schemas.OrderCreate,
    response: Response,
    db: Session = Depends(get_db),
    current_user: str = Depends(auth.require_user),
):
    """Create an order and schedule background price prediction."""
    order = crud.create_order(db, data)

    response.headers["Location"] = f"/orders/{order.id}"

    enqueue_prediction(order.id, data.model_dump())

    logger.info(
        "Order created: order_id=%s status=pending",
        order.id,
    )

    return order


@app.get(
    "/orders",
    response_model=list[schemas.OrderRead],
)
def list_orders(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: str = Depends(auth.require_user),
):
    """Return a paginated list of orders."""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100.",
        )

    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="Offset must be non-negative.",
        )

    return crud.list_orders(
        db,
        limit,
        offset,
    )


@app.get(
    "/orders/{oid}",
    response_model=schemas.OrderRead,
)
def get_order(
    oid: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(auth.require_user),
):
    """Return a single order by ID."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    return order


@app.patch("/orders/{oid}", response_model=schemas.OrderRead)
def update_order(
    oid: int,
    data: schemas.OrderUpdate,
    db: Session = Depends(get_db),
    current_user: str = Depends(auth.require_user),
):
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(status_code=404, detail="Order not found")

    changes = data.model_dump(exclude_unset=True)

    property_fields = {
        "gr_liv_area",
        "total_bsmt_sf",
        "garage_area",
        "year_built",
        "overall_qual",
        "ms_zoning",
    }

    property_changed = any(
        field in changes and changes[field] != getattr(order, field)
        for field in property_fields
    )

    order = crud.update_order(db, order, data)

    if property_changed:
        order = crud.update_prediction(
            db,
            oid,
            None,
            "pending",
        )

        payload = {
            "gr_liv_area": order.gr_liv_area,
            "total_bsmt_sf": order.total_bsmt_sf,
            "garage_area": order.garage_area,
            "year_built": order.year_built,
            "overall_qual": order.overall_qual,
            "ms_zoning": order.ms_zoning,
        }

        enqueue_prediction(oid, payload)

        logger.info(
            "Prediction rescheduled: order_id=%s",
            oid,
        )

    return order



@app.delete(
    "/orders/{oid}",
    status_code=204,
)
def delete_order(
    oid: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(auth.require_user),
) -> None:
    """Delete an existing order."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    crud.delete_order(db, order)
