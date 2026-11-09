from datetime import datetime

from pydantic import BaseModel, Field


class OrderCreate(BaseModel):
    """Request schema for creating a new order."""

    client_name: str = Field(
        min_length=2,
        max_length=120,
        description="Client name.",
    )

    gr_liv_area: float = Field(
        gt=0,
        le=10_000,
        description="Above-ground living area in square feet.",
    )

    total_bsmt_sf: float = Field(
        ge=0,
        le=10_000,
        description="Total basement area in square feet.",
    )

    garage_area: float = Field(
        ge=0,
        le=2_000,
        description="Garage area in square feet.",
    )

    year_built: int = Field(
        ge=1872,
        le=2010,
        description="Original construction year.",
    )

    overall_qual: int = Field(
        ge=1,
        le=10,
        description="Overall material and finish quality, rated from 1 to 10.",
    )

    ms_zoning: str = Field(
        min_length=1,
        max_length=10,
        description="General zoning classification.",
    )


class OrderUpdate(BaseModel):
    client_name: str | None = Field(default=None, min_length=2, max_length=120)
    gr_liv_area: float | None = Field(default=None, gt=0, le=10000)
    total_bsmt_sf: float | None = Field(default=None, ge=0, le=10000)
    garage_area: float | None = Field(default=None, ge=0, le=2000)
    year_built: int | None = Field(default=None, ge=1872, le=2010)
    overall_qual: int | None = Field(default=None, ge=1, le=10)
    ms_zoning: str | None = Field(default=None, min_length=1, max_length=10)

class OrderRead(BaseModel):
    """Response schema representing an order."""

    model_config = {"from_attributes": True}

    id: int
    client_name: str

    gr_liv_area: float
    total_bsmt_sf: float
    garage_area: float
    year_built: int
    overall_qual: int
    ms_zoning: str

    predicted_price: float | None
    model_version: str | None
    status: str
    created_at: datetime
class OrderBatchCreate(BaseModel):
    """Request schema for creating a batch of orders."""

    items: list[OrderCreate] = Field(
        min_length=1,

    )


class OrderBatchItemResult(BaseModel):
    """Result of processing one order in a batch."""

    index: int
    order_id: int
    status: str
    predicted_price: float | None = None
    model_version: str | None = None
    error: str | None = None


class OrderBatchResponse(BaseModel):
    """Response containing results for all batch items."""

    total: int
    results: list[OrderBatchItemResult]
class UserCredentials(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
