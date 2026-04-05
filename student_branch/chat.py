from .prompt import SYSTEM_PROMPT

async def handle_student_branch_chat(user_query, call_groq_func):
    synthesis_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    # Use the versatile model for general conversation
    synth_res = await call_groq_func(synthesis_msgs, temperature=0.7)
    
    if not synth_res:
        return {"error": "Student Branch Synthesis failed"}
        
    return synth_res
