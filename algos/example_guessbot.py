from utils import callgpt

def hello_world_process_question(haystack_sessions: list[list[dict]], question: str, question_date:str, haystack_dates: list[str]) -> str:
    # haystack_sessions is a list of sessions, each session is a list of turns, each turn is a dict with role and content
    # question_date : '2023/08/20 (Sun) 23:59'
    # return hypothesis: str
    result = callgpt(model="gpt-4o", max_tokens=250,
                     messages=[
                         # {"role": "system", "content": "Try to make a wild guess to answer the question. Don't say you need more information, just make up a guess."},
                         {"role": "system", "content": "Try to make a guess at the mostly likely answer for the question. Don't say you need more information, just make up a SINGLE guess.  Do NOT make a list of guesses.  Just the most likely guess."},
                         {"role": "user", "content": question}])
    print(f"\nQuestion: {question}")
    print(f"Result: {result}")
    return result

