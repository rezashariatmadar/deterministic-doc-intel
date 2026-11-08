"""add financial metrics table and seed data

Revision ID: 20261108_financial_metrics
Revises: 20261107_telemetry
Create Date: 2026-11-08 12:00:00.000000

"""

import uuid
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "20261108_financial_metrics"
down_revision: Union[str, Sequence[str], None] = "20261107_telemetry"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema and insert seed financial metrics."""
    financial_metrics_table = op.create_table(
        "financial_metrics",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("company", sa.String(length=100), nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("fiscal_year", sa.Integer(), nullable=False),
        sa.Column("metric_name", sa.String(length=100), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=50), server_default="USD_MILLIONS", nullable=False),
        sa.Column("period", sa.String(length=20), server_default="FY", nullable=False),
        sa.Column("source_filing", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        op.f("ix_financial_metrics_company"), "financial_metrics", ["company"], unique=False
    )
    op.create_index(
        op.f("ix_financial_metrics_ticker"), "financial_metrics", ["ticker"], unique=False
    )
    op.create_index(
        op.f("ix_financial_metrics_fiscal_year"), "financial_metrics", ["fiscal_year"], unique=False
    )
    op.create_index(
        op.f("ix_financial_metrics_metric_name"), "financial_metrics", ["metric_name"], unique=False
    )

    # Seed data for Apple (FY22-23) and NVIDIA (FY23-24)
    seed_records = [
        # Apple FY 2022
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2022,
            "metric_name": "revenue",
            "metric_value": 394328.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2022",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2022,
            "metric_name": "net_income",
            "metric_value": 99803.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2022",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2022,
            "metric_name": "gross_margin",
            "metric_value": 170782.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2022",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2022,
            "metric_name": "operating_income",
            "metric_value": 119437.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2022",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2022,
            "metric_name": "eps",
            "metric_value": 6.11,
            "unit": "USD_PER_SHARE",
            "period": "FY",
            "source_filing": "Apple 10-K FY2022",
        },
        # Apple FY 2023
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2023,
            "metric_name": "revenue",
            "metric_value": 383285.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2023,
            "metric_name": "net_income",
            "metric_value": 96995.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2023,
            "metric_name": "gross_margin",
            "metric_value": 169148.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2023,
            "metric_name": "operating_income",
            "metric_value": 114301.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2023,
            "metric_name": "eps",
            "metric_value": 6.13,
            "unit": "USD_PER_SHARE",
            "period": "FY",
            "source_filing": "Apple 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2023,
            "metric_name": "rd_expense",
            "metric_value": 29915.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2023,
            "metric_name": "total_assets",
            "metric_value": 352583.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2023,
            "metric_name": "total_current_assets",
            "metric_value": 143566.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2023,
            "metric_name": "commercial_paper",
            "metric_value": 5985.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2023,
            "metric_name": "term_debt",
            "metric_value": 95281.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "Apple 10-K FY2023",
        },
        # NVIDIA FY 2023
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2023,
            "metric_name": "revenue",
            "metric_value": 26974.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2023,
            "metric_name": "net_income",
            "metric_value": 4368.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2023,
            "metric_name": "gross_margin",
            "metric_value": 15356.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2023,
            "metric_name": "operating_income",
            "metric_value": 4224.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2023",
        },
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2023,
            "metric_name": "eps",
            "metric_value": 1.74,
            "unit": "USD_PER_SHARE",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2023",
        },
        # NVIDIA FY 2024
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2024,
            "metric_name": "revenue",
            "metric_value": 60922.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2024",
        },
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2024,
            "metric_name": "net_income",
            "metric_value": 29760.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2024",
        },
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2024,
            "metric_name": "gross_margin",
            "metric_value": 44301.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2024",
        },
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2024,
            "metric_name": "operating_income",
            "metric_value": 32972.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2024",
        },
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2024,
            "metric_name": "eps",
            "metric_value": 11.93,
            "unit": "USD_PER_SHARE",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2024",
        },
        {
            "id": uuid.uuid4(),
            "company": "NVIDIA",
            "ticker": "NVDA",
            "fiscal_year": 2024,
            "metric_name": "rd_expense",
            "metric_value": 8675.0,
            "unit": "USD_MILLIONS",
            "period": "FY",
            "source_filing": "NVIDIA 10-K FY2024",
        },
    ]

    op.bulk_insert(financial_metrics_table, seed_records)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_financial_metrics_metric_name"), table_name="financial_metrics")
    op.drop_index(op.f("ix_financial_metrics_fiscal_year"), table_name="financial_metrics")
    op.drop_index(op.f("ix_financial_metrics_ticker"), table_name="financial_metrics")
    op.drop_index(op.f("ix_financial_metrics_company"), table_name="financial_metrics")
    op.drop_table("financial_metrics")

