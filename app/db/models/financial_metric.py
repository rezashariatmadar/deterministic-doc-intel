"""SQLAlchemy ORM model for structured financial metrics."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class FinancialMetric(Base):
    """Stores structured annual/quarterly financial metrics for companies."""

    __tablename__ = "financial_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    company: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Company name (e.g. Apple, NVIDIA)",
    )
    ticker: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    fiscal_year: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        comment="Fiscal year (e.g. 2022, 2023, 2024)",
    )
    metric_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Metric name (e.g. revenue, net_income, gross_margin)",
    )
    metric_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Numeric metric value in specified unit",
    )
    unit: Mapped[str] = mapped_column(
        String(50),
        default="USD_MILLIONS",
        nullable=False,
        comment="Unit of measurement (e.g. USD_MILLIONS, PERCENT, USD_PER_SHARE)",
    )
    period: Mapped[str] = mapped_column(
        String(20),
        default="FY",
        nullable=False,
        comment="Reporting period (e.g. FY, Q1, Q2, Q3, Q4)",
    )
    source_filing: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Filing reference (e.g. Apple 10-K FY2023)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
