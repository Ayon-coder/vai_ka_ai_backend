import os

def replace_in_file(path, old, new):
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {path}")
    else:
        print(f"Old content not found in {path}")

# 1. Update gemini/student/prompt.py and groq_backend/student/prompt.py
student_old = '''6. If the context does not contain the answer, reply EXACTLY:
   "I don't have enough information about that in the student branch records."
7. Never invent names, roles, or details not present in the context.'''

student_new = '''6. If the user asks a conversational question (e.g. greetings, "what was my last query") or refers to chat history, respond naturally using the chat history.
7. If the user asks a factual question about the branch and the context does not contain the answer, reply: "I don't have enough information about that in the student branch records."
8. Never invent names, roles, or details not present in the context.'''

prompt_old = '''student_branch_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{question}")
])'''

prompt_new = '''student_branch_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Chat History:\\n{chat_history}\\n\\nQuestion: {question}")
])'''

for p in ["gemini/student/prompt.py", "groq_backend/student/prompt.py"]:
    replace_in_file(p, student_old, student_new)
    replace_in_file(p, prompt_old, prompt_new)


# 2. Update gemini/deep_dive/prompt.py and groq_backend/deep_dive/prompt.py
dd_prompt_old = '''deep_dive_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{question}")
])'''

dd_prompt_new = '''deep_dive_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Chat History:\\n{chat_history}\\n\\nQuestion: {question}")
])'''

dd_sys_old = '''3. Format output clearly with code blocks for code, bold text for emphasis, etc.
4. Keep the tone professional, technical, and helpful.'''

dd_sys_new = '''3. Format output clearly with code blocks for code, bold text for emphasis, etc.
4. Keep the tone professional, technical, and helpful.
5. If the user refers to previous context or engages in conversation, answer naturally using the chat history.'''

for p in ["gemini/deep_dive/prompt.py", "groq_backend/deep_dive/prompt.py"]:
    replace_in_file(p, dd_sys_old, dd_sys_new)
    replace_in_file(p, dd_prompt_old, dd_prompt_new)


# 3. Update gemini/student/chain.py and groq_backend/student/chain.py
student_chain_old = '''    return (
        {"context": _get_retriever() | _format_docs, "question": RunnablePassthrough()}
        | student_branch_prompt'''

student_chain_new = '''    from operator import itemgetter
    return (
        {
            "context": itemgetter("question") | _get_retriever() | _format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | student_branch_prompt'''

for p in ["gemini/student/chain.py", "groq_backend/student/chain.py"]:
    replace_in_file(p, student_chain_old, student_chain_new)


# 4. Update main.py
main_old_process = '''async def process_chat(query: str, mode: str):
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
                formatted.append(f"[Source {idx}] {res['title']}\\nLink: {res['link']}\\nSnippet: {res['snippet']}")
            context_str = "\\n\\n".join(formatted)
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
        return ChatResponse(response=response_text)'''

main_new_process = '''def extract_query_and_history(messages):
    query = messages[-1].get("content", "") if messages else ""
    history_str = ""
    if len(messages) > 1:
        for msg in messages[-6:-1]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_str += f"{role}: {msg.get('content', '')}\\n"
    if not history_str:
        history_str = "No previous conversation."
    return query, history_str

async def process_chat(messages: List[Dict[str, Any]], mode: str):
    """
    Processes the chat asynchronously using LangChain's .ainvoke().
    """
    query, history_str = extract_query_and_history(messages)
    
    if _OBVIOUS_GIBBERISH_RE.search(query) or _ABUSIVE_OR_ROLEPLAY_RE.search(query):
        return MODERATION_WARNING_MESSAGE

    internal_mode = "tech" if mode == "deep_dive" else "student"
    if internal_mode == "tech":
        results = await async_search_ieee(query)
        formatted = []
        if results:
            for idx, res in enumerate(results, 1):
                formatted.append(f"[Source {idx}] {res['title']}\\nLink: {res['link']}\\nSnippet: {res['snippet']}")
            context_str = "\\n\\n".join(formatted)
        else:
            context_str = "No results found."
            
        chain = get_deep_dive_chain()
        return await chain.ainvoke({"context": context_str, "question": query, "chat_history": history_str})
    else:
        chain = get_student_branch_chain()
        return await chain.ainvoke({"question": query, "chat_history": history_str})

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")
    
    query = request.messages[-1].get("content", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        response_text = await process_chat(request.messages, request.mode)
        return ChatResponse(response=response_text)'''

main_old_stream = '''async def chat_stream_generator(query: str, mode: str):
    if _OBVIOUS_GIBBERISH_RE.search(query) or _ABUSIVE_OR_ROLEPLAY_RE.search(query):
        meta = json.dumps({"type": "meta", "sources": [], "is_warning": True})
        yield f"data: {meta}\\n\\n"
        data = json.dumps({"type": "chunk", "content": MODERATION_WARNING_MESSAGE})
        yield f"data: {data}\\n\\n"
        done_data = json.dumps({"type": "done"})
        yield f"data: {done_data}\\n\\n"
        return

    internal_mode = "tech" if mode == "deep_dive" else "student"
    
    try:
        if internal_mode == "tech":
            results = await async_search_ieee(query)
            
            sources = []
            formatted = []
            if results:
                for idx, res in enumerate(results, 1):
                    formatted.append(f"[Source {idx}] {res['title']}\\nLink: {res['link']}\\nSnippet: {res['snippet']}")
                    sources.append({"title": res["title"], "link": res["link"]})
                context_str = "\\n\\n".join(formatted)
            else:
                context_str = "No results found."
                
            meta = json.dumps({"type": "meta", "sources": sources})
            yield f"data: {meta}\\n\\n"
            
            chain = get_deep_dive_chain()
            async for chunk in chain.astream({"context": context_str, "question": query}):
                data = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {data}\\n\\n"
        else:
            meta = json.dumps({"type": "meta", "sources": []})
            yield f"data: {meta}\\n\\n"
            
            chain = get_student_branch_chain()
            async for chunk in chain.astream(query):
                data = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {data}\\n\\n"
        
        done_data = json.dumps({"type": "done"})
        yield f"data: {done_data}\\n\\n"
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "too many requests" in err_str or "resourceexhausted" in err_str or "rate limit" in err_str:
            data = json.dumps({"type": "chunk", "content": "bohot msg ho raha hein ruk ja bhai"})
            yield f"data: {data}\\n\\n"
            done_data = json.dumps({"type": "done"})
            yield f"data: {done_data}\\n\\n"
        else:
            err_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {err_data}\\n\\n"

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
    )'''

main_new_stream = '''async def chat_stream_generator(messages: List[Dict[str, Any]], mode: str):
    query, history_str = extract_query_and_history(messages)
    
    if _OBVIOUS_GIBBERISH_RE.search(query) or _ABUSIVE_OR_ROLEPLAY_RE.search(query):
        meta = json.dumps({"type": "meta", "sources": [], "is_warning": True})
        yield f"data: {meta}\\n\\n"
        data = json.dumps({"type": "chunk", "content": MODERATION_WARNING_MESSAGE})
        yield f"data: {data}\\n\\n"
        done_data = json.dumps({"type": "done"})
        yield f"data: {done_data}\\n\\n"
        return

    internal_mode = "tech" if mode == "deep_dive" else "student"
    
    try:
        if internal_mode == "tech":
            results = await async_search_ieee(query)
            
            sources = []
            formatted = []
            if results:
                for idx, res in enumerate(results, 1):
                    formatted.append(f"[Source {idx}] {res['title']}\\nLink: {res['link']}\\nSnippet: {res['snippet']}")
                    sources.append({"title": res["title"], "link": res["link"]})
                context_str = "\\n\\n".join(formatted)
            else:
                context_str = "No results found."
                
            meta = json.dumps({"type": "meta", "sources": sources})
            yield f"data: {meta}\\n\\n"
            
            chain = get_deep_dive_chain()
            async for chunk in chain.astream({"context": context_str, "question": query, "chat_history": history_str}):
                data = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {data}\\n\\n"
        else:
            meta = json.dumps({"type": "meta", "sources": []})
            yield f"data: {meta}\\n\\n"
            
            chain = get_student_branch_chain()
            async for chunk in chain.astream({"question": query, "chat_history": history_str}):
                data = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {data}\\n\\n"
        
        done_data = json.dumps({"type": "done"})
        yield f"data: {done_data}\\n\\n"
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "too many requests" in err_str or "resourceexhausted" in err_str or "rate limit" in err_str:
            data = json.dumps({"type": "chunk", "content": "bohot msg ho raha hein ruk ja bhai"})
            yield f"data: {data}\\n\\n"
            done_data = json.dumps({"type": "done"})
            yield f"data: {done_data}\\n\\n"
        else:
            err_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {err_data}\\n\\n"

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty.")
    
    query = request.messages[-1].get("content", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    return StreamingResponse(
        chat_stream_generator(request.messages, request.mode),
        media_type="text/event-stream"
    )'''

replace_in_file("main.py", main_old_process, main_new_process)
replace_in_file("main.py", main_old_stream, main_new_stream)
