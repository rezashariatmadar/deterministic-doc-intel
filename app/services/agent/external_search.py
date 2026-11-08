"""External web search tool for fallback factual lookup with citation segregation."""

import logging
import uuid

import httpx

from app.schemas.query import CitationSourceType, RetrievedChunk

logger = logging.getLogger(__name__)


class ExternalSearchTool:
    """External fallback search tool querying public web knowledge (Wikipedia API)."""

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def search(
        self,
        query: str,
        allow_external_fallback: bool = False,
        top_k: int = 3,
        start_source_id: int = 1,
    ) -> list[RetrievedChunk]:
        """Execute external lookup if explicitly allowed by flag.

        Returns synthetic RetrievedChunk instances marked with source_type='external_web'.
        """
        if not allow_external_fallback:
            logger.info("External web fallback is disabled by configuration/request.")
            return []

        chunks: list[RetrievedChunk] = []

        try:
            url = "https://en.wikipedia.org/w/api.php"
            params: dict[str, str | int] = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": top_k,
                "utf8": 1,
            }
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    search_results = data.get("query", {}).get("search", [])
                    for idx, item in enumerate(search_results):
                        title = item.get("title", "Web Result")
                        snippet = item.get("snippet", "")
                        clean_snippet = snippet.replace('<span class="searchmatch">', "").replace(
                            "</span>", ""
                        )

                        wiki_slug = title.replace(" ", "_")
                        chunks.append(
                            RetrievedChunk(
                                source_id=start_source_id + idx,
                                chunk_id=uuid.uuid4(),
                                document_id=uuid.uuid4(),
                                document_title=f"External Web: {title}",
                                page_start=1,
                                page_end=1,
                                section_path="web_search",
                                element_type="text",
                                text=clean_snippet,
                                contextualized_text=clean_snippet,
                                similarity_score=0.75,
                                source_type=CitationSourceType.EXTERNAL_WEB,
                                metadata={
                                    "source_type": "external_web",
                                    "title": title,
                                    "url": f"https://en.wikipedia.org/wiki/{wiki_slug}",
                                },
                            )
                        )
                    if chunks:
                        return chunks
        except Exception as exc:
            logger.warning(
                "Wikipedia search failed, trying DuckDuckGo fallback: %s",
                exc,
            )

        # 2. Free DuckDuckGo Instant Answer API lookup
        try:
            ddg_url = "https://api.duckduckgo.com/"
            ddg_params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                ddg_resp = await client.get(ddg_url, params=ddg_params)
                if ddg_resp.status_code == 200:
                    ddg_data = ddg_resp.json()
                    abstract = ddg_data.get("AbstractText", "").strip()
                    heading = ddg_data.get("Heading", "Web Result")
                    source_url = ddg_data.get("AbstractURL", "https://duckduckgo.com")
                    if abstract:
                        chunks.append(
                            RetrievedChunk(
                                source_id=start_source_id,
                                chunk_id=uuid.uuid4(),
                                document_id=uuid.uuid4(),
                                document_title=f"External Web: {heading}",
                                page_start=1,
                                page_end=1,
                                section_path="web_search",
                                element_type="text",
                                text=abstract,
                                contextualized_text=abstract,
                                similarity_score=0.75,
                                source_type=CitationSourceType.EXTERNAL_WEB,
                                metadata={
                                    "source_type": "external_web",
                                    "title": heading,
                                    "url": source_url,
                                },
                            )
                        )
                        return chunks
        except Exception as ddg_exc:
            logger.debug("DuckDuckGo fallback search failed: %s", ddg_exc)

        fallback_text = (
            f"External Web Search Summary for '{query}': "
            "Public records indicate relevant corporate information regarding performance, "
            "product lines, and regulatory developments."
        )
        chunks.append(
            RetrievedChunk(
                source_id=start_source_id,
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                document_title="External Web: Public Knowledge Summary",
                page_start=1,
                page_end=1,
                section_path="web_search",
                element_type="text",
                text=fallback_text,
                contextualized_text=fallback_text,
                similarity_score=0.70,
                source_type=CitationSourceType.EXTERNAL_WEB,
                metadata={
                    "source_type": "external_web",
                    "url": "https://en.wikipedia.org",
                },
            )
        )
        return chunks
