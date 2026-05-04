from .prompt import SYSTEM_PROMPT

async def handle_student_branch_chat(context_window, call_groq_func):
    """
    Handle student branch chat using a context window of recent messages.
    `context_window` is a list of {"role": ..., "content": ...} dicts,
    already trimmed to the last N messages by the caller.
    """
    # Build the full message list: system prompt + conversation history
    synthesis_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + context_window

    # Use the versatile model for general conversation
    synth_res = await call_groq_func(synthesis_msgs, temperature=0.7)

    if not synth_res:
        return {"error": "Student Branch Synthesis failed"}

    return synth_res
