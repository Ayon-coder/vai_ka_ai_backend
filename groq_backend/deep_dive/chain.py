from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from .prompt import deep_dive_prompt
from key_manager import groq_keys

def get_deep_dive_chain():
    llm = ChatGroq(
        model="openai/gpt-oss-120b", 
        temperature=0.7,
        groq_api_key=groq_keys.get_next_key()
    )
    return (
        deep_dive_prompt
        | llm
        | StrOutputParser()
    )
