from utils import callgpt

def process_question(haystack_sessions: list[list[dict]], question: str, question_date: str, haystack_dates: list[str]) -> str:
    # Concatenate all conversations into a single prompt
    prompt = f"QUESTION: {question}\nQUESTION_DATE: {question_date}\n\n"
    for date, session in sorted(zip(haystack_dates, haystack_sessions), key=lambda x: x[0]):
        prompt += f"\n[Session from {date}]\n"
        for turn in session:
            prompt += f"{turn['role'].upper()}: {turn['content']}\n"

    # Send the prompt to GPT-4o
    response = callgpt([
        {'role': 'system', 'content': f"You're a helpful assistant. Answer the question below using relevant info from prior sessions.\n{prompt}"}
    ], model='gpt-4o', max_tokens=1024)

    return response
