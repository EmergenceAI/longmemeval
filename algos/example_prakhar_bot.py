from utils import callgpt

# def process_question(haystack_sessions: list[list[dict]], question: str, question_date: str, haystack_dates: list[str]) -> str:
#     # Concatenate all conversations into a single prompt
#     prompt = f"QUESTION: {question}\nQUESTION_DATE: {question_date}\n\n"
#     for date, session in sorted(zip(haystack_dates, haystack_sessions), key=lambda x: x[0]):
#         prompt += f"\n[Session from {date}]\n"
#         for turn in session:
#             prompt += f"{turn['role'].upper()}: {turn['content']}\n"

#     # Send the prompt to GPT-4o
#     response = callgpt([
#         {'role': 'system', 'content': f"You're a helpful assistant. Answer the question below using relevant info from prior sessions.\n{prompt}"}
#     ], model='gpt-4o', max_tokens=1024)

#     return response

def process_question(haystack_sessions: list[list[dict]], question: str, question_date: str, haystack_dates: list[str]) -> str:
    # Concatenate all conversations into a single prompt
    prompt = f"QUESTION: {question}\nQUESTION_DATE: {question_date}\n\n"
    for date, session in sorted(zip(haystack_dates, haystack_sessions), key=lambda x: x[0]):
        prompt += f"\n\n<Session from {date}>\n\n"
        for turn in session:
            prompt += f"{turn['role'].upper()}: {turn['content']}\n"
        prompt += f"\n\n<End Session from {date}>\n\n"

    # Send the prompt to GPT-4o
    response = callgpt([
        {'role': 'system', 'content': f"You're a helpful assistant. Answer the question below using relevant info from prior sessions.\n{prompt}"}
    ], model='gpt-4o', max_tokens=1024)
    return response


def test():
    import json
    DATA_DIR = './data/'
    FILENAME = 'longmemeval_s.json'
    # For Marc's REPL debugging.
    haystacks = json.load(open(f'{DATA_DIR}/{FILENAME}'))
    haystack = haystacks[0]
    question_id = haystack['question_id']
    question = haystack['question']
    question_date = haystack['question_date']
    haystack_dates = haystack['haystack_dates']
    haystack_sessions = haystack['haystack_sessions']
    answer = haystack['answer']
    # This is to get my linter to shut up about unused vars.
    print(len([question_id, question, question_date, haystack_dates, haystack_sessions, answer]))
