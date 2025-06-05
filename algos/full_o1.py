from utils import immutify_messages, remutify_messages
import diskcache, openai, os
CACHE_DIR = './cache/'

cache_openai = diskcache.Cache(os.path.join(CACHE_DIR, 'openai'), eviction_policy='none')
cache_openai.reset('cull_limit', 0)
@cache_openai.memoize()
def _callgpt_helper2(immutable_messages, model:str):
    if _callgpt_helper2.client is None: _callgpt_helper2.client = openai.OpenAI()
    messages = remutify_messages(immutable_messages)
    response = _callgpt_helper2.client.chat.completions.create(model=model, messages=messages)
    message = response.choices[0].message
    return message.content
_callgpt_helper2.client = None

def callgpt2(messages, model:str):
    immutable_messages = immutify_messages(messages)
    return _callgpt_helper2(immutable_messages, model)


def process_haystack(haystack_sessions: list[list[dict]], haystack_dates: list[str]):
    # Sort the haystack sessions by date
    haystack_sessions2 = list(zip(haystack_dates, haystack_sessions))
    haystack_sessions2.sort(key=lambda x: x[0])
    conv_prompt = ''
    for haydate, conv in haystack_sessions2:
        conv_prompt += f'\n\nCONVERSATION DATED: {haydate}\n\n'
        for message in conv:
            conv_prompt += f'\n\n{message["role"].upper()}: {message["content"]}\n\n'
    return conv_prompt

def process_question(memstruct, question: str, question_date: str) -> str:
    conv_prompt = memstruct
    # Add the question to the prompt
    conv_prompt += f'\n\nQUESTION: {question}\n\nQUESTION_DATE: {question_date}\n\n'
    messages = [{'role': 'system', 'content': f'Given the dated conversation below, please succinctly answer the question. \n\nCONVERSATIONS:\n{conv_prompt}'}]
    answer = callgpt2(messages, model='o1')
    return answer
