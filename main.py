import os
import sys
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import re

# Load environment variables
load_dotenv()

_OBVIOUS_GIBBERISH_RE = re.compile(r'(.)\1{5,}|[bcdfghjklmnpqrstvwxyz]{8,}', re.IGNORECASE)
_ABUSIVE_OR_ROLEPLAY_RE = re.compile(
    r'\b('
    # General profanity & severe insults
    r'fuck(?:ing|er|ed)?|shit(?:ty)?|bitch(?:es|y)?|asshole|bastard|dickhead|cunt|motherfucker|dumbass|stfu|twat|wanker|bullshit|jackass'
    # Slurs (racial, homophobic, ableist, etc.)
    r'|faggot|fag|tranny|retard|retarded|nigger|nigga|chink|spick|wetback|kike|dyke|gook'
    # NSFW, sexual anatomy & explicit acts
    r'|pussy|cock|dick|boobs|tits|titties|vagina|penis|anal|blowjob|handjob|cum|jizz|semen|masturbate|dildo|vibrator|rape|incest|pedophile'
    r'|porn|nude|nudes|slut|whore|hooker|prostitute|escort|sex|horny|fetish|bdsm|kink'
    # Self-harm
    r'|kill\s+(?:your|my)self|suicide'
    # Romantic/inappropriate roleplay
    r'|pretend\s+(?:you\s+are|to\s+be)\s+my\s+(?:girlfriend|boyfriend|wife|husband|lover|slave)'
    r'|act\s+as\s+my\s+(?:girlfriend|boyfriend|wife|husband|lover|slave)'
    r'|be\s+my\s+(?:girlfriend|boyfriend|wife|husband|lover|slave)'
    r'|ignore\s+(?:all\s+)?(?:previous\s+)?instructions|disregard\s+all\s+prior\s+instructions|enable\s+DAN\s+mode'
    r')\b', re.IGNORECASE
)
MODERATION_WARNING_MESSAGE = (
    "I am a strictly technical and professional AI assistant. "
    "Please refrain from using abusive language, roleplay, or non-sensical queries. "
    "Repeated violations will result in a temporary ban."
)

from search_tool import search_ieee as async_search_ieee

# Determine provider
provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
print(f"Starting ASGI Backend Server with provider: {provider.upper()}")

if provider == "groq":
    try:
        from groq_backend import get_deep_dive_chain, get_student_branch_chain
    except ImportError as e:
        print(f"Error loading Groq modules: {e}")
        sys.exit(1)
elif provider == "gemini":
    try:
        from gemini import get_deep_dive_chain, get_student_branch_chain
    except ImportError as e:
        print(f"Error loading Gemini modules: {e}")
        sys.exit(1)
else:
    print(f"Unsupported LLM_PROVIDER: '{provider}'")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(title="IEEE Chatbot Async Backend")

# Add CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    mode: str = "student_branch" # 'student_branch' or 'deep_dive'

class ChatResponse(BaseModel):
    response: str

async def process_chat(query: str, mode: str):
    """
    Processes the chat asynchronously using LangChain's .ainvoke().
    """
    if _OBVIOUS_GIBBERISH_RE.search(query) or _ABUSIVE_OR_ROLEPLAY_RE.search(query):
        return MODERATION_WARNING_MESSAGE

    internal_mode = "tech" if mode == "deep_dive" else "student"
    if internal_mode == "tech":
        results = await async_search_ieee(query)
        formatted = []
        if results:
            for idx, res in enumerate(results, 1):
                formatted.append(f"[Source {idx}] {res['title']}\nLink: {res['link']}\nSnippet: {res['snippet']}")
            context_str = "\n\n".join(formatted)
        else:
            context_str = "No results found."
            
        chain = get_deep_dive_chain()
        return await chain.ainvoke({"context": context_str, "question": query})
    else:
        chain = get_student_branch_chain()
        return await chain.ainvoke(query)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")
    
    query = request.messages[-1].get("content", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        response_text = await process_chat(query, request.mode)
        return ChatResponse(response=response_text)
    except Exception as e:
        print(f"Error during chain execution: {e}")
        err_str = str(e).lower()
        if "429" in err_str or "too many requests" in err_str or "resourceexhausted" in err_str or "rate limit" in err_str:
            return ChatResponse(response="bohot msg ho raha hein ruk ja bhai")
        raise HTTPException(status_code=500, detail=str(e))

async def chat_stream_generator(query: str, mode: str):
    if _OBVIOUS_GIBBERISH_RE.search(query) or _ABUSIVE_OR_ROLEPLAY_RE.search(query):
        meta = json.dumps({"type": "meta", "sources": [], "is_warning": True})
        yield f"data: {meta}\n\n"
        data = json.dumps({"type": "chunk", "content": MODERATION_WARNING_MESSAGE})
        yield f"data: {data}\n\n"
        done_data = json.dumps({"type": "done"})
        yield f"data: {done_data}\n\n"
        return

    internal_mode = "tech" if mode == "deep_dive" else "student"
    
    try:
        if internal_mode == "tech":
            results = await async_search_ieee(query)
            
            sources = []
            formatted = []
            if results:
                for idx, res in enumerate(results, 1):
                    formatted.append(f"[Source {idx}] {res['title']}\nLink: {res['link']}\nSnippet: {res['snippet']}")
                    sources.append({"title": res["title"], "link": res["link"]})
                context_str = "\n\n".join(formatted)
            else:
                context_str = "No results found."
                
            meta = json.dumps({"type": "meta", "sources": sources})
            yield f"data: {meta}\n\n"
            
            chain = get_deep_dive_chain()
            async for chunk in chain.astream({"context": context_str, "question": query}):
                data = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {data}\n\n"
        else:
            meta = json.dumps({"type": "meta", "sources": []})
            yield f"data: {meta}\n\n"
            
            chain = get_student_branch_chain()
            async for chunk in chain.astream(query):
                data = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {data}\n\n"
        
        done_data = json.dumps({"type": "done"})
        yield f"data: {done_data}\n\n"
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "too many requests" in err_str or "resourceexhausted" in err_str or "rate limit" in err_str:
            data = json.dumps({"type": "chunk", "content": "bohot msg ho raha hein ruk ja bhai"})
            yield f"data: {data}\n\n"
            done_data = json.dumps({"type": "done"})
            yield f"data: {done_data}\n\n"
        else:
            err_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {err_data}\n\n"

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")
    
    query = request.messages[-1].get("content", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    return StreamingResponse(
        chat_stream_generator(query, request.mode),
        media_type="text/event-stream"
    )

@app.post("/api/warmup")
async def warmup_endpoint():
    return {"status": "ok"}


@app.post("/api/cache/refresh")
async def cache_refresh_endpoint():
    """Force-refresh the Firestore document cache without restarting the server."""
    try:
        from retriever_shared import get_cached_docs_async
        entries, vecs = await get_cached_docs_async(force_refresh=True)
        return {"status": "refreshed", "docs_loaded": len(entries)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
async def health_check():
    return {
        "status": "online", 
        "provider": provider, 
    }

# Local development only — Vercel ignores this block.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
