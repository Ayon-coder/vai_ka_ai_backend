"""
retriever.py  (gemini/student)
------------------------------
LangChain BaseRetriever that queries the three IEEE Student Branch Firestore
collections and returns the most semantically relevant documents.

Key improvements:
- Singleton embedding model from retriever_shared (no 700ms re-init per request).
- Shared Firestore document cache with 30-min TTL via retriever_shared.
- LRU query-vector cache via retriever_shared (saves ~550ms for repeat queries).
- _aget_relevant_documents override so LangChain async chains use the async
  path directly (no thread-pool detour for .ainvoke / .astream).
- Deduplication by doc_id: keeps only the highest-scoring version of a doc
  when the same ID appears in multiple collections (e.g. tech-team in both
  teams_overview and team_members_details).
- min_similarity raised 0.35 -> 0.45 to reduce noise.
- top_k raised 4 -> 5 for slightly richer context.
- numpy batch cosine scoring via retriever_shared (3.6ms for 65 docs).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from retriever_shared import (
    batch_cosine_scores,
    embed_query_async,
    get_cached_docs_async,
)


class FirebaseStudentRetriever(BaseRetriever):
    """Semantic retriever for IEEE Student Branch Firestore collections."""

    # Kept for API backward-compatibility; actual model is the shared singleton.
    collection_name: str = "ieee-members"
    embeddings: GoogleGenerativeAIEmbeddings
    top_k: int = 8
    min_similarity: float = 0.45

    # ------------------------------------------------------------------
    # LangChain interface
    # ------------------------------------------------------------------

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Synchronous wrapper used by .invoke() / direct test calls."""
        try:
            asyncio.get_running_loop()
            # A loop is already running (e.g. inside pytest-asyncio or FastAPI).
            # Spawn a fresh loop in a dedicated thread to avoid nesting errors.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._retrieve_async(query))
                return future.result()
        except RuntimeError:
            # No running loop - safe to call asyncio.run directly.
            return asyncio.run(self._retrieve_async(query))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Native async implementation.
        LangChain calls this directly on .ainvoke() / .astream(), avoiding
        the thread-pool detour that the base class would otherwise apply.
        """
        return await self._retrieve_async(query)

    # ------------------------------------------------------------------
    # Core retrieval logic
    # ------------------------------------------------------------------

    async def _retrieve_async(self, query: str) -> List[Document]:
        """Embed the query, score all cached docs, deduplicate, return top-k."""
        try:
            # Embed query (LRU-cached) and load doc cache concurrently.
            query_vec, (entries, doc_vecs) = await asyncio.gather(
                embed_query_async(query),
                get_cached_docs_async(),
            )

            if not entries:
                return [Document(page_content="No documents in the knowledge base.")]

            scored_entries = []
            scores = batch_cosine_scores(query_vec, doc_vecs)
            for score, (data, doc_id, coll) in zip(scores, entries):
                if score >= self.min_similarity:
                    scored_entries.append((score, data, doc_id, coll))

            scored_entries.sort(key=lambda x: x[0], reverse=True)
            top = scored_entries[: self.top_k]

            if not top:
                return [Document(page_content="No matching records found for this query.")]

            return [
                Document(
                    page_content=self._format_record(data),
                    metadata={
                        "similarity": round(score, 4),
                        "id": doc_id,
                        "collection": coll,
                    },
                )
                for score, data, doc_id, coll in top
            ]

        except Exception as exc:
            return [Document(page_content=f"Error retrieving documents: {exc}")]

    # ------------------------------------------------------------------
    # Record formatters
    # ------------------------------------------------------------------

    @staticmethod
    def _format_record(data: Dict[str, Any]) -> str:
        """Format a Firestore document dict into a readable context string."""

        # 1. IEEE Student Branch report chunk (has both content + title)
        if "content" in data and "title" in data:
            title = data.get("title", "")
            sec = data.get("section_title", "")
            content = data.get("content", "")
            header = f"[{title}]" if title else ""
            if sec and sec != title:
                header += f" (Section: {sec})"
            return f"{header}\n{content}".strip()

        # 2. Team overview (description / common_queries present)
        if "description" in data or "common_queries" in data:
            parts = []
            if data.get("name"):
                parts.append(f"Team: {data['name']}")
            if data.get("description"):
                parts.append(f"Description: {data['description']}")
            if data.get("answer"):
                parts.append(f"Overview & Role: {data['answer']}")
            return "\n".join(parts)

        # 3. Team parent summary (member roster)
        if data.get("member_names"):
            team_name = data.get("name") or data.get("team", "")
            total = data.get("total_members", len(data["member_names"]))
            parts = [
                f"Team: {team_name}",
                f"Members ({total}): {', '.join(data['member_names'])}",
            ]
            if data.get("description"):
                parts.append(f"Description: {data['description']}")
            return "\n".join(parts)

        # 4. Individual member profile
        field_map = [
            ("name", "Name"),
            ("team", "Team"),
            ("department", "Department"),
            ("role", "Role"),
            ("bio", "Bio"),
            ("interests", "Interests"),
        ]
        parts = []
        for key, label in field_map:
            val = data.get(key)
            if val:
                parts.append(f"{label}: {val}")
        linkedin = data.get("linkedin") or data.get("linkedin_url")
        if linkedin:
            parts.append(f"LinkedIn: {linkedin}")
        return "\n".join(parts) if parts else data.get("embedding_text", "")
