import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import httpx
import re
import asyncio
import random

# Import custom modules (Deep Dive Logic)
from deep_dive.prompt import SYSTEM_PROMPT, CLASSIFIER_PROMPT, REJECTION_MESSAGE, NOT_FOUND_MESSAGE, IDENTITY_MESSAGE
from deep_dive.tool import search_ieee

# Import Student Branch Logic
from student_branch.chat import handle_student_branch_chat

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Allow cross-origin requests from the separately deployed frontend
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
CORS(app, origins=[FRONTEND_URL] if FRONTEND_URL != "*" else "*")

# Parse comma-separated API keys into lists for load balancing
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEY", "").split(",") if k.strip()]
CATEGORICAL_API_KEYS = [k.strip() for k in os.getenv("CATEGORICAL_MODEL_API_KEY", "").split(",") if k.strip()]

# Fallback: if no categorical keys set, use the main keys
if not CATEGORICAL_API_KEYS:
    CATEGORICAL_API_KEYS = GROQ_API_KEYS

GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
CATEGORICAL_MODEL = os.getenv("CATEGORICAL_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

print(f"Loaded {len(GROQ_API_KEYS)} main API key(s), {len(CATEGORICAL_API_KEYS)} categorical API key(s)")

async def call_groq(messages, model=None, temperature=0):
    """
    Utility function to call the Groq API asynchronously.
    Randomly selects an API key from the appropriate pool:
    - Categorical model → CATEGORICAL_API_KEYS
    - All other models → GROQ_API_KEYS
    """
    if model is None:
        model = GROQ_MODEL_NAME
    
    # Pick the right key pool based on which model is being used
    if model == CATEGORICAL_MODEL:
        key_pool = CATEGORICAL_API_KEYS
    else:
        key_pool = GROQ_API_KEYS
    
    api_key = random.choice(key_pool)
    print(f"[DEBUG] Model: {model} | Key: {api_key[:8]}...{api_key[-4:]} | Pool size: {len(key_pool)}")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": 1024,
        "top_p": 1,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)
            if response.status_code != 200:
                print(f"Groq API Error Status: {response.status_code}")
                print(f"Groq API Error Response: {response.text}")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error calling Groq: {e}")
        return None

@app.route('/')
def index():
    return jsonify({"status": "ok", "message": "IEEE Chatbot API is running."})

@app.route('/api/warmup', methods=['GET', 'POST'])
def warmup():
    """
    Minimal LLM call to warm up the provider's cold start.
    """
    print("Warmup requested...")
    warmup_msgs = [
        {"role": "system", "content": "You are a warmup assistant. Reply only with 'OK'."},
        {"role": "user", "content": "test"}
    ]
    # Use the fastest model for rollup warmup
    result = asyncio.run(call_groq(warmup_msgs, model=CATEGORICAL_MODEL))
    if not result:
        return jsonify({"status": "error", "message": "Failed to connect to AI provider. Check API keys."}), 503
    return jsonify({"status": "warmed_up", "message": "Backend is ready."})



@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages')
    
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "Invalid messages format"}), 400

    user_query = messages[-1].get('content')
    mode = data.get('mode', 'deep_dive') # Default to deep_dive if not provided

    # Routing based on user selected mode
    if mode == 'student_branch':
        print(f"Routing to Student Branch (Normal AI) -> Query: '{user_query}'")
        res = asyncio.run(handle_student_branch_chat(user_query, call_groq))
        if "error" in res:
             return jsonify(res), 500
        # No sources needed for student branch
        res['sources'] = [] 
        return jsonify(res)

    # ---------------------------------------------------------
    # IEEE DEEP DIVE MODE (Strict Search & Classification)
    # ---------------------------------------------------------

    # STEP 1 & 2: Classify and Search CONCURRENTLY to save time
    print(f"Processing query: '{user_query}'...")
    
    classification_msgs = [
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user", "content": user_query}
    ]

    try:
        # Run classification and search at the same time
        class_task = call_groq(classification_msgs, model=CATEGORICAL_MODEL)
        search_task = search_ieee(user_query)
        
        class_res, search_results = asyncio.run(asyncio.gather(class_task, search_task))

        if not class_res:
            print("Error: Classification task returned no result.")
            return jsonify({"error": "Classification failed"}), 500
        
        category = class_res['choices'][0]['message']['content'].strip().upper()
        print(f"Category: {category} | Search Results: {len(search_results) if search_results else 0}")

        # CASE A: GREETING
        if category == "GREETING":
            greet_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ]
            greet_res = asyncio.run(call_groq(greet_msgs, temperature=0.7))
            return jsonify(greet_res)

        # CASE B: REJECTED
        if category == "REJECTED":
            return jsonify({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": REJECTION_MESSAGE
                    }
                }],
                "is_rejected": True
            })

        # CASE C: TECHNICAL (Allowed)
        # STEP 3: Format Context
        context_parts = ["<IEEE_SOURCES>"]
        for i, r in enumerate(search_results or [], 1):
            year = "N/A"
            year_match = re.search(r'\b(19|20)\d{2}\b', r['snippet'])
            if year_match:
                year = year_match.group(0)
            context_parts.append(f"[Source {i}]\nTitle: {r['title']}\nYear: {year}\nContent: {r['snippet']}\n")
        context_parts.append("</IEEE_SOURCES>")
        
        context_str = "\n".join(context_parts)

        # STEP 4: Synthesis
        synthesis_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nUser Question: {user_query}"}
        ]

        synth_res = asyncio.run(call_groq(synthesis_msgs))
        if not synth_res:
            print("Error: Synthesis task returned no result.")
            return jsonify({"error": "Synthesis failed - AI provider did not respond"}), 500

        # Return answer along with sources metadata
        final_response = synth_res.copy()
        final_response['sources'] = search_results if search_results else []
        
        # Check if the model said it couldn't find information
        content = final_response['choices'][0]['message']['content']
        if "I could not find this in IEEE sources" in content or not search_results:
             if not search_results and "I could not find" not in content:
                 final_response['choices'][0]['message']['content'] = NOT_FOUND_MESSAGE

        return jsonify(final_response)

    except Exception as e:
        print(f"Chat execution error: {str(e)}")
        return jsonify({"error": f"Internal server error: {type(e).__name__}"}), 500

if __name__ == '__main__':
    # When running locally, Flask development server can handle some concurrency with threaded=True
    # For production, use: uvicorn app:app --workers 4
    app.run(port=5000, debug=True, threaded=True)
