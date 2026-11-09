# Integration of Information Systems — Practice 4

Continuation of Practice 3. Includes PostgreSQL, Redis/RQ queue and worker, Orders API,
MLflow registry, on-demand training service and registry-based Forecast API.

## First start (PowerShell)

1. Create `secrets/postgres_password.txt` and `secrets/jwt_secret.txt` with strong distinct values. Copy `.env.example` to `.env`.
2. `docker compose build`
3. `docker compose up -d postgres redis mlflow`
4. If reusing the Practice 3 PostgreSQL volume, create the MLflow database **once**:
   `docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname=\x27mlflow\x27"'`
   If missing, run `docker compose exec postgres psql -U orders -d postgres -c "CREATE DATABASE mlflow;"`.
   The init SQL executes **only for a fresh PostgreSQL volume**. Do not delete the volume to rerun it.
5. `docker compose --profile train run --rm trainer` (20 Optuna trials × 5-fold CV; can take a while).
6. `docker compose up -d forecast-api orders-api orders-worker`
7. `docker compose ps`

UI: http://localhost:5000 (operator-only). Orders API: http://localhost:8000/docs.
The MLflow UI is an operator management interface, not a client prediction API;
Forecast API and Redis expose no host ports.

## Training / promotion / rollback

Training registers a Ridge baseline and XGBoost final model as versions of `house-price`.
The final version is assigned the `prod` alias. Use MLflow UI to inspect metrics,
20 nested Optuna trials, dataset digest, data_stats.json, reference_data.parquet,
target_distributions.png and the model signature.

To retrain: `docker compose --profile train run --rm trainer`.
To switch aliases from a Python shell in the trainer container:
`MlflowClient().set_registered_model_alias('house-price','prod','VERSION')`.
Then `docker compose restart forecast-api`. Check `/health` from inside the network.
For rollback, set the previous version and restart; no forecast image rebuild.
The model is resolved once at service startup; a missing alias causes startup failure.

Internal checks:
`docker compose exec forecast-api python -c "import urllib.request as u; print(u.urlopen('http://localhost:8001/health').read().decode())"`
`docker compose exec forecast-api python -c "import urllib.request as u; print(u.urlopen('http://localhost:8001/model/data-stats').read().decode()[:500])"`

Orders: POST `/auth/register`, POST `/auth/login`, POST `/orders` (202 + Location),
GET `/orders`, GET `/orders/{id}`, PATCH `/orders/{id}`, DELETE `/orders/{id}`,
POST `/orders/batch`. Bearer JWT required for orders. `model_version` is persisted.

## Limitations

Compose and Python sources have not been exercised against Docker in the generation environment.
Test on your machine and capture genuine results before submitting the report.
MLflow and Optuna training can require substantial RAM and time. The optional drift-monitoring
bonus is intentionally not implemented.

## Важливе обмеження відтворюваності
Версії залежностей закріплені в pyproject.toml, MLflow 3.1.0 — у Dockerfile. Спільний uv.lock у середовищі без доступу до мережі не згенеровано; вимога R1.8 ще потребує `uv lock` на комп'ютері з інтернетом, після чого Dockerfile слід перевести на `uv sync --frozen`.

## Promotion / rollback
`docker compose --profile train run --rm trainer python scripts/model_alias.py VERSION`
`docker compose restart forecast-api`
Фіксуйте версію /health та ціну прогнозу до/після.
