"""Safe Read-Only Text-to-SQL Sandbox with AST/regex security guards and self-correction."""

import asyncio
import logging
import re
import uuid
from typing import Any, ClassVar

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import SafeSQLError
from app.schemas.query import CitationSourceType, RetrievedChunk
from app.services.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class SafeSQLTool:
    """Validates, sanitizes, executes read-only SQL queries and converts results to chunks."""

    FORBIDDEN_KEYWORDS: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|"
        r"EXECUTE|EXEC|CALL|COPY|VACUUM|INTO|SET|BEGIN|COMMIT|ROLLBACK)\b",
        re.IGNORECASE,
    )
    SYSTEM_SCHEMAS: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:information_schema|pg_catalog|pg_|sqlite_|current_setting|"
        r"current_user|session_user)\b",
        re.IGNORECASE,
    )
    DISALLOWED_TABLES: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:documents|chunks|parent_chunks|query_logs|alembic_version)\b",
        re.IGNORECASE,
    )
    COMMENT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"(?:--|/\*|\*/|#)")

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        timeout_ms: int = 3000,
        max_correction_retries: int = 1,
    ) -> None:
        self.llm_provider = llm_provider
        self.timeout_ms = timeout_ms
        self.max_correction_retries = max_correction_retries
        self.last_executed_sql: str | None = None

    def _check_balanced_tokens(self, sql: str) -> None:
        """AST check: verify balanced parentheses and quotes to prevent injection escapes."""
        # Check single quotes balance
        if sql.count("'") % 2 != 0:
            raise SafeSQLError("Unbalanced single quotes in SQL query.", query=sql)
        # Check double quotes balance
        if sql.count('"') % 2 != 0:
            raise SafeSQLError("Unbalanced double quotes in SQL query.", query=sql)

        # Check balanced parentheses outside string literals
        in_string = False
        paren_depth = 0
        for char in sql:
            if char == "'":
                in_string = not in_string
            elif not in_string:
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1
                    if paren_depth < 0:
                        raise SafeSQLError("Mismatched parentheses in SQL query.", query=sql)
        if paren_depth != 0:
            raise SafeSQLError("Unbalanced parentheses in SQL query.", query=sql)

    def validate_sql(self, sql_query: str) -> str:
        """Validate read-only query against security sandbox rules."""
        if not sql_query or not sql_query.strip():
            raise SafeSQLError("Empty SQL query.")

        cleaned = sql_query.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"[\s;]+$", "", cleaned).strip()

        # Reject internal semicolons
        if ";" in cleaned:
            raise SafeSQLError(
                "Multiple statements or internal semicolons are prohibited.",
                query=cleaned,
            )

        # Reject SQL comments
        if self.COMMENT_PATTERN.search(cleaned):
            raise SafeSQLError(
                "SQL comments are strictly prohibited in sandbox queries.",
                query=cleaned,
            )

        # AST structural checks: balanced tokens
        self._check_balanced_tokens(cleaned)

        # Reject forbidden DDL / DML keywords
        match_forbidden = self.FORBIDDEN_KEYWORDS.search(cleaned)
        if match_forbidden:
            raise SafeSQLError(
                f"Forbidden SQL keyword detected: '{match_forbidden.group(0)}'",
                query=cleaned,
            )

        # Reject system catalog / schema references
        match_sys = self.SYSTEM_SCHEMAS.search(cleaned)
        if match_sys:
            raise SafeSQLError(
                f"Access to system catalogs/functions is prohibited: '{match_sys.group(0)}'",
                query=cleaned,
            )

        # Reject explicit access to other application tables
        match_disallowed = self.DISALLOWED_TABLES.search(cleaned)
        if match_disallowed:
            raise SafeSQLError(
                f"Access to table '{match_disallowed.group(0)}' is prohibited in SQL sandbox.",
                query=cleaned,
            )

        # Must be a SELECT or WITH statement
        if not re.match(r"^(?:SELECT|WITH)\b", cleaned, re.IGNORECASE):
            raise SafeSQLError(
                "Query must be a read-only SELECT or WITH statement.",
                query=cleaned,
            )

        # Restrict table access exclusively to financial_metrics
        if "financial_metrics" not in cleaned.lower():
            raise SafeSQLError(
                "Queries are strictly restricted to the 'financial_metrics' table.",
                query=cleaned,
            )

        # Reject multiple comma-separated tables in FROM clause (e.g. FROM table1, table2)
        if re.search(r"\bFROM\s+[^,;()]+\s*,", cleaned, re.IGNORECASE):
            raise SafeSQLError(
                "Comma-separated multi-table joins are prohibited in SQL sandbox.",
                query=cleaned,
            )

        # Extract any CTE aliases defined in WITH clauses (including optional column lists)
        cte_aliases = {
            m.lower()
            for m in re.findall(
                r"\b([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*\([^)]*\))?\s+AS\s*\(",
                cleaned,
                re.IGNORECASE,
            )
        }
        allowed_targets = (
            {"financial_metrics", "public.financial_metrics"}
            | cte_aliases
            | {f"public.{a}" for a in cte_aliases}
        )

        # Verify all FROM and JOIN table targets are financial_metrics, CTE aliases, or subqueries
        targets = re.findall(r"\b(?:FROM|JOIN)\s+([a-zA-Z0-9_.\'\"]+)", cleaned, re.IGNORECASE)
        for target in targets:
            clean_target = target.strip("'\"").lower()
            if clean_target not in allowed_targets and not clean_target.startswith("("):
                raise SafeSQLError(
                    f"Queries are restricted to 'financial_metrics', forbidden target: '{target}'",
                    query=cleaned,
                )

        # Enforce maximum result limit
        if not re.search(r"\bLIMIT\s+\d+", cleaned, re.IGNORECASE):
            cleaned = f"{cleaned} LIMIT 50"

        return cleaned

    async def generate_sql(self, natural_query: str) -> str:
        """Convert natural language inquiry to a clean read-only SQL query."""
        if self.llm_provider is not None:
            prompt = (
                "You are an expert Text-to-SQL translator for a Postgres database.\n"
                "You have access to ONE table:\n\n"
                "Table: financial_metrics\n"
                "Columns:\n"
                "  - company (VARCHAR, e.g. 'Apple', 'NVIDIA')\n"
                "  - ticker (VARCHAR, e.g. 'AAPL', 'NVDA')\n"
                "  - fiscal_year (INTEGER, e.g. 2022, 2023, 2024)\n"
                "  - metric_name (VARCHAR, e.g. 'revenue', 'net_income', 'gross_margin', 'eps')\n"
                "  - metric_value (FLOAT)\n"
                "  - unit (VARCHAR, e.g. 'USD_MILLIONS', 'USD_PER_SHARE')\n"
                "  - period (VARCHAR, e.g. 'FY')\n\n"
                "Rules:\n"
                "1. Return ONLY the raw SQL query. No explanation, no backticks, no semicolons.\n"
                "2. Only use SELECT on table 'financial_metrics'.\n"
                "3. Use case-insensitive matching for company.\n\n"
                f"User Question: {natural_query}\n"
                "SQL Query:"
            )
            try:
                raw_sql = await self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You generate strictly safe, read-only SQL queries.",
                    temperature=0.0,
                )
                return self.validate_sql(raw_sql)
            except SafeSQLError:
                raise
            except Exception as exc:
                logger.warning("LLM SQL generation failed, falling back to heuristic: %s", exc)

        # Deterministic heuristic fallback
        q_lower = natural_query.lower()
        company = (
            "Apple"
            if "apple" in q_lower or "aapl" in q_lower
            else ("NVIDIA" if "nvidia" in q_lower or "nvda" in q_lower else None)
        )
        years = re.findall(r"\b(2022|2023|2024)\b", natural_query)

        conditions = ["1=1"]
        if company:
            conditions.append(f"company = '{company}'")
        if years:
            year_list = ", ".join(years)
            conditions.append(f"fiscal_year IN ({year_list})")

        sql = (
            f"SELECT company, ticker, fiscal_year, metric_name, metric_value, unit "
            f"FROM financial_metrics WHERE {' AND '.join(conditions)} "
            f"ORDER BY fiscal_year, metric_name"
        )
        return self.validate_sql(sql)

    async def self_correct_sql(self, failed_sql: str, error_message: str) -> str:
        """Self-correct a failed SQL query using LLM error guidance (1 retry allowed)."""
        if self.llm_provider is None:
            raise SafeSQLError(f"SQL execution failed: {error_message}", query=failed_sql)

        prompt = (
            "The following SQL query failed execution against table 'financial_metrics':\n\n"
            f"Failed SQL: {failed_sql}\n"
            f"Database Error: {error_message}\n\n"
            "Correct the SQL query to resolve this error. Return ONLY the corrected SQL query.\n"
            "Corrected SQL:"
        )
        try:
            corrected_raw = await self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You fix SQL queries for PostgreSQL. Return only raw query.",
                temperature=0.0,
            )
            return self.validate_sql(corrected_raw)
        except Exception as exc:
            logger.warning("SQL self-correction failed: %s", exc)
            raise SafeSQLError(
                f"SQL execution and self-correction failed: {error_message}", query=failed_sql
            ) from exc

    async def execute_query(
        self,
        session: AsyncSession,
        sql_query: str,
    ) -> tuple[list[str], list[list[Any]]]:
        """Execute validated SQL query in read-only sandbox with rollback protection."""
        valid_sql = self.validate_sql(sql_query)

        retries_remaining = self.max_correction_retries
        current_sql = valid_sql

        while True:
            try:
                # Enforce database-level read-only transaction mode
                try:
                    await session.execute(text("SET TRANSACTION READ ONLY"))
                except Exception as ro_exc:
                    logger.debug("Could not set transaction to read only: %s", ro_exc)

                result = await asyncio.wait_for(
                    session.execute(text(current_sql)),
                    timeout=self.timeout_ms / 1000.0,
                )
                columns = list(result.keys())
                rows = [list(r) for r in result.fetchall()]
                self.last_executed_sql = current_sql
                await session.rollback()
                return columns, rows
            except Exception as exc:
                # CRITICAL Trap 3: Always rollback to prevent poisoning the async session
                await session.rollback()
                logger.warning("SQL execution failed for '%s': %s", current_sql, exc)

                if retries_remaining > 0 and self.llm_provider is not None:
                    retries_remaining -= 1
                    logger.info("Attempting SQL self-correction retry...")
                    try:
                        current_sql = await self.self_correct_sql(current_sql, str(exc))
                        continue
                    except Exception as corr_exc:
                        raise SafeSQLError(
                            f"SQL self-correction failed: {corr_exc}", query=current_sql
                        ) from corr_exc

                raise SafeSQLError(f"SQL execution error: {exc}", query=current_sql) from exc

    def format_as_chunk(
        self,
        columns: list[str],
        rows: list[list[Any]],
        sql_query: str,
        source_id: int = 1,
    ) -> RetrievedChunk:
        """Format tabular SQL results into a synthetic RetrievedChunk for citations."""
        if not rows:
            table_text = "No matching records found in financial_metrics table."
        else:
            headers = " | ".join(columns)
            separators = " | ".join(["---"] * len(columns))
            row_strs = [" | ".join(str(cell) for cell in row) for row in rows]
            table_text = f"| {headers} |\n| {separators} |\n" + "\n".join(
                f"| {r} |" for r in row_strs
            )

        return RetrievedChunk(
            source_id=source_id,
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_title="Database Table: financial_metrics",
            page_start=1,
            page_end=1,
            section_path="financial_metrics",
            element_type="table",
            text=table_text,
            contextualized_text=table_text,
            similarity_score=1.0,
            source_type=CitationSourceType.SQL_TABLE,
            metadata={
                "source_type": "sql_table",
                "sql_query": sql_query,
                "row_count": len(rows),
                "columns": columns,
            },
        )
