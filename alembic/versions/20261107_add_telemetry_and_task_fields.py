"""add telemetry and task fields

Revision ID: 20261107_telemetry
Revises: 7040528719a0
Create Date: 2026-11-07 16:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20261107_telemetry"
down_revision: Union[str, Sequence[str], None] = "7040528719a0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Add task_id and processing_stage columns to documents
    op.add_column(
        "documents",
        sa.Column(
            "task_id",
            sa.String(length=100),
            nullable=True,
            comment="Celery background task ID",
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "processing_stage",
            sa.String(length=50),
            nullable=True,
            comment="QUEUED | PARSING | CHUNKING | EMBEDDING | INDEXING | COMPLETED | FAILED",
        ),
    )
    op.create_index(
        op.f("ix_documents_task_id"), "documents", ["task_id"], unique=False
    )

    # 2. Create query_logs table
    op.create_table(
        "query_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("search_mode", sa.String(length=20), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("is_answerable", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("confidence_score", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column("top_reranker_score", sa.Float(), nullable=True),
        sa.Column("rewrite_ms", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column("dense_search_ms", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column("sparse_search_ms", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column("rerank_ms", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column("llm_ttft_ms", sa.Float(), nullable=True),
        sa.Column("total_latency_ms", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("embedding_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("estimated_cost_usd", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_query_logs_request_id"), "query_logs", ["request_id"], unique=False
    )
    op.create_index(
        op.f("ix_query_logs_session_id"), "query_logs", ["session_id"], unique=False
    )
    op.create_index(
        op.f("ix_query_logs_cache_hit"), "query_logs", ["cache_hit"], unique=False
    )
    op.create_index(
        op.f("ix_query_logs_created_at"), "query_logs", ["created_at"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_query_logs_created_at"), table_name="query_logs")
    op.drop_index(op.f("ix_query_logs_cache_hit"), table_name="query_logs")
    op.drop_index(op.f("ix_query_logs_session_id"), table_name="query_logs")
    op.drop_index(op.f("ix_query_logs_request_id"), table_name="query_logs")
    op.drop_table("query_logs")
    op.drop_index(op.f("ix_documents_task_id"), table_name="documents")
    op.drop_column("documents", "processing_stage")
    op.drop_column("documents", "task_id")
