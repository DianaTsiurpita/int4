import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "",
)

if os.getenv("POSTGRES_PASSWORD_FILE"):
    with open(os.environ["POSTGRES_PASSWORD_FILE"], encoding="utf-8") as secret_file:
        db_password = secret_file.read().strip()
    DATABASE_URL = URL.create(
        "postgresql+psycopg",
        username=os.getenv("POSTGRES_USER", "orders"),
        password=db_password,
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=5432,
        database=os.getenv("POSTGRES_DB", "orders"),
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


def get_db():
    """Provide a database session for a request."""
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
