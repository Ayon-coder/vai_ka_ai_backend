"""
chain.py  (groq_backend/student)
---------------------------------
Assembles the student-branch RAG chain using Groq as the LLM backend.

Same singleton-retriever pattern as the Gemini chain:
- FirebaseStudentRetriever is a module-level singleton (shared embedding model,
  shared Firestore doc cache, no per-request overhead).
- LLM (ChatGroq) is created fresh per call for API-key round-robin.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from key_manager import gemini_keys, groq_keys
from retriever_shared import get_embedding_model

from .prompt import student_branch_prompt
from .retriever import FirebaseStudentRetriever


def _format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Singleton retriever
# ---------------------------------------------------------------------------

_retriever: FirebaseStudentRetriever | None = None


def _get_retriever() -> FirebaseStudentRetriever:
    global _retriever
    if _retriever is None:
        # Embeddings MUST use a Gemini key - data in DB was embedded with Gemini.
        _retriever = FirebaseStudentRetriever(embeddings=get_embedding_model())
    return _retriever


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_student_branch_chain():
    """
    Return the student-branch RAG chain (Groq LLM + shared Gemini retriever).
    """
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        groq_api_key=groq_keys.get_next_key(),
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
