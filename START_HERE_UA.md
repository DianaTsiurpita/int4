# Лабораторна №4 — запуск у PowerShell
1. Створіть secrets/postgres_password.txt та secrets/jwt_secret.txt (сильні різні паролі); скопіюйте .env.example у .env.
2. docker compose config --quiet
3. docker compose build
4. docker compose up -d postgres redis mlflow
5. Якщо том PostgreSQL вже ініціалізований і база mlflow відсутня: docker compose exec postgres psql -U orders -d postgres -c "CREATE DATABASE mlflow;"
6. docker compose --profile train run --rm trainer
7. docker compose up -d forecast-api orders-api orders-worker
8. docker compose ps
MLflow UI: http://localhost:5000; Orders Swagger: http://localhost:8000/docs.
Бонусний моніторинг drift не реалізовано. Збережіть реальні скріншоти перевірок.
