from utils import callgpt

def process_question(haystack_sessions: list[list[dict]], question: str, question_date:str, haystack_dates: list[str]) -> str:
    # Sort the haystack sessions by date
    haystack_sessions2 = list(zip(haystack_dates, haystack_sessions))
    haystack_sessions2.sort(key=lambda x: x[0])
    conv_prompt = ''
    for haydate, conv in haystack_sessions2:
        conv_prompt += f'\n\nCONVERSATION DATED: {haydate}\n\n'
        for message in conv:
            conv_prompt += f'\n\n{message["role"].upper()}: {message["content"]}\n\n'
    # Add the question to the prompt
    conv_prompt += f'\n\nQUESTION: {question}\n\nQUESTION_DATE: {question_date}\n\n'
    messages = [{'role': 'system', 'content': f'Given the dated conversation below, please succinctly answer the question. \n\nCONVERSATIONS:\n{conv_prompt}'}]
    answer = callgpt(messages, model='gpt-4o', max_tokens=2048)
    return answer
