from llm_utils import clean_llm_response

from .prompt import SYSTEM_PROMPT

# The student branch model is a reasoning model: its chain-of-thought is billed
# against the completion budget before any visible answer is produced. 1500
# tokens comfortably covers the trace plus a full conversational reply, so
# responses are never cut off with finish_reason="length".
SYNTHESIS_MAX_TOKENS = 1500


async def handle_student_branch_chat(context_window, call_groq_func):
    """
    Handle student branch chat using a context window of recent messages.
    `context_window` is a list of {"role": ..., "content": ...} dicts,
    already trimmed to the last N messages by the caller.
    """
    # Build the full message list: system prompt + conversation history
    synthesis_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + context_window

    # Use the versatile model for general conversation
    synth_res = await call_groq_func(
        synthesis_msgs,
        temperature=0.7,
        max_tokens=SYNTHESIS_MAX_TOKENS,
    )

    if not synth_res:
        return {"error": "Student Branch Synthesis failed"}

    # Verify the response actually carries usable text. clean_llm_response strips
    # any <think> markup and falls back to the reasoning trace, so this only
    # fails when the model genuinely returned nothing.
    if not clean_llm_response(synth_res):
        return {"error": "Student Branch Synthesis returned empty content"}

    return synth_res
