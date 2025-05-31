# This is a placeholder for YOUR function.  The API should be clear from below.

def hello_world_process_question(haystack_sessions: list[list[dict]], question: str, question_date:str, haystack_dates: list[str]) -> str:
    # haystack_sessions is a list of sessions, each session is a list of turns, each turn is a dict with role and content
    # question_date : '2023/08/20 (Sun) 23:59'
    # return hypothesis: str
    return f"{question}?  I don't know the answer to that!"
