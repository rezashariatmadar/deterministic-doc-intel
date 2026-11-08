"""Unit tests for SafeSQLTool and ExternalSearchTool."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import SafeSQLError
from app.db.session import async_session_maker
from app.schemas.query import CitationSourceType
from app.services.agent.external_search import ExternalSearchTool
from app.services.agent.sql_tool import SafeSQLTool
from app.services.providers.base import LLMProvider


class MockSQLLLM(LLMProvider):
    def __init__(self) -> None:
        super().__init__(model_name="mock-sql-llm")
        self.generate_mock = AsyncMock()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        res = await self.generate_mock(prompt, system_prompt=system_prompt, temperature=temperature)
        return str(res)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncGenerator[str, None]:
        yield "mock"


class TestSafeSQLToolValidation:
    """Test suite for SQL sandbox validation and security checks."""

    @pytest.fixture
    def tool(self) -> SafeSQLTool:
        return SafeSQLTool(llm_provider=None)

    def test_valid_select_allowed(self, tool: SafeSQLTool) -> None:
        valid_query = (
            "SELECT company, metric_name, metric_value "
            "FROM financial_metrics WHERE fiscal_year = 2023"
        )
        cleaned = tool.validate_sql(valid_query)
        assert cleaned == f"{valid_query} LIMIT 50"

    def test_valid_select_with_trailing_semicolon_and_spaces(self, tool: SafeSQLTool) -> None:
        query = "SELECT company, metric_name FROM financial_metrics WHERE fiscal_year = 2023  ;  "
        cleaned = tool.validate_sql(query)
        expected = (
            "SELECT company, metric_name FROM financial_metrics WHERE fiscal_year = 2023 LIMIT 50"
        )
        assert cleaned == expected

    def test_valid_select_preserves_existing_limit(self, tool: SafeSQLTool) -> None:
        query = (
            "SELECT company, metric_name FROM financial_metrics WHERE fiscal_year = 2023 LIMIT 10"
        )
        cleaned = tool.validate_sql(query)
        assert cleaned == query

    def test_cte_query_allowed(self, tool: SafeSQLTool) -> None:
        cte_query = (
            "WITH apple_rev AS ("
            "SELECT company, metric_value FROM financial_metrics WHERE company = 'Apple'"
            ") SELECT company, metric_value FROM apple_rev LIMIT 5"
        )
        cleaned = tool.validate_sql(cte_query)
        assert cleaned == cte_query

    def test_cte_query_auto_limit(self, tool: SafeSQLTool) -> None:
        cte_query = (
            "WITH apple_metrics AS ("
            "SELECT * FROM financial_metrics WHERE company = 'Apple'"
            ") SELECT * FROM apple_metrics"
        )
        cleaned = tool.validate_sql(cte_query)
        assert cleaned == f"{cte_query} LIMIT 50"

    def test_cte_query_with_column_list(self, tool: SafeSQLTool) -> None:
        cte_query = (
            "WITH apple_metrics (company, metric_value) AS ("
            "SELECT company, metric_value FROM financial_metrics WHERE company = 'Apple'"
            ") SELECT company, metric_value FROM apple_metrics"
        )
        cleaned = tool.validate_sql(cte_query)
        assert cleaned == f"{cte_query} LIMIT 50"

    def test_public_schema_table_allowed(self, tool: SafeSQLTool) -> None:
        query = (
            "SELECT company, metric_value FROM public.financial_metrics WHERE fiscal_year = 2023"
        )
        cleaned = tool.validate_sql(query)
        assert cleaned == f"{query} LIMIT 50"

    def test_multiple_ctes_allowed(self, tool: SafeSQLTool) -> None:
        query = (
            "WITH cte1 AS ("
            "SELECT company, metric_value FROM financial_metrics WHERE company = 'Apple'"
            "), cte2 (company, metric_value) AS ("
            "SELECT company, metric_value FROM financial_metrics WHERE company = 'NVIDIA'"
            ") SELECT * FROM cte1 UNION ALL SELECT * FROM cte2"
        )
        cleaned = tool.validate_sql(query)
        assert cleaned == f"{query} LIMIT 50"

    @pytest.mark.parametrize(
        "malicious_sql",
        [
            "SELECT * FROM financial_metrics; DROP TABLE financial_metrics;",
            "SELECT * FROM financial_metrics WHERE fiscal_year = 2023; SELECT * FROM users;",
            "SELECT * FROM financial_metrics -- bypass checks",
            "SELECT * FROM financial_metrics /* comment block */ WHERE 1=1",
            "SELECT * FROM financial_metrics # mysql hash comment",
            "INSERT INTO financial_metrics (company) VALUES ('HackerCorp')",
            "DELETE FROM financial_metrics WHERE company = 'Apple'",
            "UPDATE financial_metrics SET metric_value = 0",
            "DROP TABLE financial_metrics",
            "TRUNCATE TABLE financial_metrics",
            "SELECT * FROM pg_catalog.pg_tables",
            "SELECT * FROM information_schema.tables",
            "SELECT * FROM documents",  # Not financial_metrics
            "SELECT * FROM financial_metrics, documents",  # Comma join attack
            "SELECT * FROM financial_metrics JOIN documents ON 1=1",  # Explicit join attack
            "SELECT * FROM financial_metrics UNION SELECT * FROM documents",  # Union attack
            "SELECT * FROM financial_metrics WHERE company = 'Apple",  # Unbalanced quote
            "SELECT * FROM financial_metrics WHERE (fiscal_year = 2023",  # Unbalanced paren
        ],
    )
    def test_rejects_unsafe_queries(self, tool: SafeSQLTool, malicious_sql: str) -> None:
        with pytest.raises(SafeSQLError):
            tool.validate_sql(malicious_sql)


class TestSafeSQLToolExecution:
    """Test suite for sandbox execution against PostgreSQL database."""

    @pytest.mark.asyncio
    async def test_execute_and_fetch_apple_nvidia_seed(self) -> None:
        tool = SafeSQLTool()
        async with async_session_maker() as session:
            # Query Apple FY23 revenue
            sql = (
                "SELECT company, fiscal_year, metric_name, metric_value "
                "FROM financial_metrics "
                "WHERE company = 'Apple' AND fiscal_year = 2023 AND metric_name = 'revenue'"
            )
            cols, rows = await tool.execute_query(session, sql)
            assert cols == ["company", "fiscal_year", "metric_name", "metric_value"]
            assert len(rows) == 1
            assert rows[0][0] == "Apple"
            assert rows[0][1] == 2023
            assert rows[0][3] == 383285.0

            # Query NVIDIA FY24 net income
            sql_nvda = (
                "SELECT metric_value FROM financial_metrics "
                "WHERE company = 'NVIDIA' AND fiscal_year = 2024 AND metric_name = 'net_income'"
            )
            cols, rows = await tool.execute_query(session, sql_nvda)
            assert len(rows) == 1
            assert rows[0][0] == 29760.0
            assert tool.last_executed_sql == f"{sql_nvda} LIMIT 50"

    @pytest.mark.asyncio
    async def test_session_rollback_on_execution_error(self) -> None:
        tool = SafeSQLTool()
        async with async_session_maker() as session:
            # Invalid column in financial_metrics
            bad_sql = "SELECT non_existent_column FROM financial_metrics"
            with pytest.raises(SafeSQLError):
                await tool.execute_query(session, bad_sql)

            # Ensure session is still usable because rollback was executed (Trap 3)
            valid_sql = "SELECT count(*) FROM financial_metrics"
            _, rows = await tool.execute_query(session, valid_sql)
            assert rows[0][0] > 0

    @pytest.mark.asyncio
    async def test_self_correction_retry(self) -> None:
        mock_llm = MockSQLLLM()
        # LLM corrects invalid column 'rev' to 'metric_value'
        mock_llm.generate_mock.return_value = (
            "SELECT company, metric_value FROM financial_metrics "
            "WHERE metric_name = 'revenue' LIMIT 1"
        )
        tool = SafeSQLTool(llm_provider=mock_llm, max_correction_retries=1)

        async with async_session_maker() as session:
            bad_initial_sql = (
                "SELECT company, rev FROM financial_metrics WHERE metric_name = 'revenue' LIMIT 1"
            )
            cols, rows = await tool.execute_query(session, bad_initial_sql)
            assert "metric_value" in cols
            assert len(rows) == 1
            mock_llm.generate_mock.assert_awaited_once()

    def test_format_as_chunk(self) -> None:
        tool = SafeSQLTool()
        cols = ["company", "metric_name", "metric_value"]
        rows = [["Apple", "revenue", 383285.0]]
        chunk = tool.format_as_chunk(cols, rows, sql_query="SELECT ...", source_id=2)

        assert chunk.source_id == 2
        assert chunk.source_type == CitationSourceType.SQL_TABLE
        assert chunk.metadata["source_type"] == "sql_table"
        assert "Apple" in chunk.text
        assert "383285.0" in chunk.text


class TestExternalSearchTool:
    """Test suite for pluggable external web search tool and citation segregation."""

    @pytest.mark.asyncio
    async def test_gated_by_allow_external_fallback(self) -> None:
        tool = ExternalSearchTool()
        # Disabled
        chunks = await tool.search("Apple latest products", allow_external_fallback=False)
        assert len(chunks) == 0

        # Enabled (uses fallback or Wikipedia)
        chunks_enabled = await tool.search("Apple Inc.", allow_external_fallback=True)
        assert len(chunks_enabled) > 0
        chunk = chunks_enabled[0]
        assert chunk.source_type == CitationSourceType.EXTERNAL_WEB
        assert chunk.metadata["source_type"] == "external_web"
