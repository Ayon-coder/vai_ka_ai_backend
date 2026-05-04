import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import httpx
import re
import asyncio
import random

# Import custom modules - use sys.path since we're in api/ subfolder
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_dive.prompt import SYSTEM_PROMPT, CLASSIFIER_PROMPT, REJECTION_MESSAGE, NOT_FOUND_MESSAGE, IDENTITY_MESSAGE
from deep_dive.tool import search_ieee

# Import Student Branch Logic
from student_branch.chat import handle_student_branch_chat

# Import context builder (regex-based context vector)
from context_builder import build_context_vector, build_slim_history

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Allow cross-origin requests from the separately deployed frontend
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
CORS(app, origins=[FRONTEND_URL] if FRONTEND_URL != "*" else "*")

# Parse comma-separated API keys into lists for load balancing
GROQ_API_KEYS = [k.strip(' "\'') for k in os.getenv("GROQ_API_KEY", "").split(",") if k.strip()]
CATEGORICAL_API_KEYS = [k.strip(' "\'') for k in os.getenv("CATEGORICAL_MODEL_API_KEY", "").split(",") if k.strip()]

# Fallback: if no categorical keys set, use the main keys
if not CATEGORICAL_API_KEYS:
    CATEGORICAL_API_KEYS = GROQ_API_KEYS

GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
CATEGORICAL_MODEL = os.getenv("CATEGORICAL_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Max number of recent messages to keep in the context window sent to the LLM.
# Each user+assistant exchange = 2 messages, so 10 = up to 5 turns of memory.
CONTEXT_WINDOW_SIZE = 10

print(f"Loaded {len(GROQ_API_KEYS)} main API key(s), {len(CATEGORICAL_API_KEYS)} categorical API key(s)")

async def call_groq(messages, model=None, temperature=0, max_tokens=1024):
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
        "max_completion_tokens": max_tokens,
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
async def index():
    return jsonify({"status": "ok", "message": "IEEE Chatbot API is running."})

@app.route('/api/warmup', methods=['GET', 'POST'])
async def warmup():
    """
    Minimal LLM call to warm up the provider's cold start.
    """
    print("Warmup requested...")
    warmup_msgs = [
        {"role": "system", "content": "You are a warmup assistant. Reply only with 'OK'."},
        {"role": "user", "content": "test"}
    ]
    # Use the fastest model for rollup warmup
    result = await call_groq(warmup_msgs, model=CATEGORICAL_MODEL)
    if not result:
        return jsonify({"status": "error", "message": "Failed to connect to AI provider. Check API keys."}), 503
    return jsonify({"status": "warmed_up", "message": "Backend is ready."})

@app.route('/api/chat', methods=['POST'])
async def chat():
    data = request.json
    messages = data.get('messages')

    if not messages or not isinstance(messages, list):
        return jsonify({"error": "Invalid messages format"}), 400

    # --- Context Queue: keep only the most recent N messages ---
    # This forms the "context vector" sent to the LLM, giving it memory
    # of the last CONTEXT_WINDOW_SIZE messages (user + assistant turns).
    context_window = messages[-CONTEXT_WINDOW_SIZE:]

    # The latest user query is always the last message in the window
    user_query = context_window[-1].get('content', '')
    mode = data.get('mode', 'deep_dive')  # Default to deep_dive if not provided

    print(f"[Context] Window size: {len(context_window)} | Mode: {mode} | Query: '{user_query[:80]}'")

    # Routing based on user selected mode
    if mode == 'student_branch':
        print(f"Routing to Student Branch (Normal AI) -> Query: '{user_query}'")
        res = await handle_student_branch_chat(context_window, call_groq)
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
        
        class_res, search_results = await asyncio.gather(class_task, search_task)

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
            greet_res = await call_groq(greet_msgs, temperature=0.7, max_tokens=150)
            return jsonify(greet_res)

        # CASE B: STUDENT BRANCH → redirect without LLM call
        if category == "STUDENT_BRANCH":
            return jsonify({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "That's a Student Branch question! Please switch to **IEEE Student Branch** mode for info about events, members, schedules & more 🎓"
                    }
                }]
            })

        # CASE C: REJECTED
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

        # STEP 4: Synthesis — regex context vector + slim history
        #
        # 1. build_context_vector() runs regex patterns over all prior turns
        #    to extract IEEE standards, acronyms, topics, specs, and years.
        #    This gives the LLM a compact semantic summary of what the
        #    conversation has been about, in ~50 tokens instead of ~500.
        #
        # 2. build_slim_history() keeps the last 2 full user+assistant pairs
        #    verbatim (for coherence) and compresses older turns to first
        #    sentence — drastically cutting token usage for long conversations.
        #
        # 3. The current user query is replaced with an augmented version that
        #    embeds the live IEEE search results.

        ctx_vector = build_context_vector(context_window)
        slim_history = build_slim_history(context_window, max_prior_turns=2)

        # Augment the system prompt with the extracted context vector
        system_with_ctx = (
            SYSTEM_PROMPT
            + (f"\n\n{ctx_vector}" if ctx_vector else "")
        )

        augmented_user_msg = {
            "role": "user",
            "content": (
                f"IEEE Source Context:\n{context_str}\n\n"
                f"User Question: {user_query}"
            )
        }
        synthesis_msgs = (
            [{"role": "system", "content": system_with_ctx}]
            + slim_history
            + [augmented_user_msg]
        )

        print(f"[Synthesis] msgs={len(synthesis_msgs)} | vector={'yes' if ctx_vector else 'no'}")

        synth_res = await call_groq(synthesis_msgs, max_tokens=600)
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
