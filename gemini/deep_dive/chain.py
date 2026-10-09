from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from .prompt import deep_dive_prompt
from key_manager import gemini_keys

def get_deep_dive_chain():
    """Generates a dynamic chain with the next round-robin API key."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.7,
        google_api_key=gemini_keys.get_next_key()
    )
    return (
        deep_dive_prompt
        | llm
        | StrOutputParser()
    )
