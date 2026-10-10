"""
chain.py  (gemini/student)
--------------------------
Assembles the student-branch RAG chain.

The FirebaseStudentRetriever and its embedding model are module-level singletons
(initialised once per process) to eliminate the ~700 ms embedding-model creation
overhead measured on every previous call to get_student_branch_chain().

The LLM is still created fresh per call so that the round-robin key manager
keeps distributing load across all available Gemini API keys.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from key_manager import gemini_keys
from retriever_shared import get_embedding_model

from .prompt import student_branch_prompt
from .retriever import FirebaseStudentRetriever


def _format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Singleton retriever (embedding model + cached docs, created once)
# ---------------------------------------------------------------------------

_retriever: FirebaseStudentRetriever | None = None


def _get_retriever() -> FirebaseStudentRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FirebaseStudentRetriever(embeddings=get_embedding_model())
    return _retriever


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_student_branch_chain():
    """
    Return the student-branch RAG chain.

    Retriever is a singleton (fast).
    LLM is fresh per call (enables API-key round-robin for rate-limit handling).
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        # temperature omitted: gemini-3.5-flash-lite uses fixed sampling defaults
        # and emits a warning if temperature is passed.
        google_api_key=gemini_keys.get_next_key(),
    )

    from operator import itemgetter
    return (
        {
            "context": itemgetter("question") | _get_retriever() | _format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | student_branch_prompt
        | llm
        | StrOutputParser()
    )
