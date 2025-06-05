from utils import callgpt

SYSPROMPT = """
You are a helpful assistant that is helping to gather data to answer a question based on a series of conversations.
Given the question, the list of previous dated facts, and the current conversation, please create new facts that are relevant to the question.
The given facts are extracted from previous conversations.  There also may be future conversations.
Please keep the facts in chronological order, when relevant.
Note the date of the question.
DO NOT REPEAT THE PREVIOUS FACTS, but do call out changes or updates to them when it might change the answer with UPDATE.
If the question is answerable, please provide the answer.  Note that this answer may change with future conversations.
The first FACT should be a summary of the conversation, even if it's not directly related to the question.
\nThe output should be of the form:\n
FACT: [conv summary]\n
FACT: [evidence to answer question]\n
FACT: [evidence to answer question]\n
etc.
ANSWER SO FAR: [answer] or [None]\n
\n
QUESTION: {question}\n\n
QUESTION_DATE: {question_date}\n
\n
PREVIOUS_FACTS:\n
{facts}\n
\n
CONVERSATION_DATE: {conversation_date}
\n\n
CONVERSATION:\n\n
{conversation}
"""

LESSON_SCHEMA = {"entries": [{"description": "str: A description of the fact."}]}

def update_facts(haydate, conv, question, question_date, old_facts):
    systemprompt = SYSPROMPT.format(
        question=question,
        question_date=question_date,
        facts=old_facts if old_facts else 'None',
        conversation_date=haydate,
        conversation='\n\n'.join([f"{turn['role']}: {turn['content']}" for turn in conv])
    )
    response = callgpt([{'role': 'system', 'content': systemprompt}], model='gpt-4o', max_tokens=2048)
    # Split on FACT: and strip the leading and trailing whitespace
    # Replace newlines in response with "\n  "
    new_facts = f"\n\nFACTS FROM {haydate}:\n\n  {response.replace('\n', '\n  ')}\n"
    return new_facts

def process_haystack(haystack_sessions: list[list[dict]], haystack_dates: list[str]):
    # Process the haystack in light of the question.
    # Sort the haystack sessions by date
    haystack_sessions2 = list(zip(haystack_dates, haystack_sessions))
    haystack_sessions2.sort(key=lambda x: x[0])
    return haystack_sessions2

def process_question(memstruct, question: str, question_date: str) -> str:
    haystack_sessions2 = memstruct
    old_facts = ''
    # Iterate over the sessions and update the facts
    for haydate, conv in haystack_sessions2:
        old_facts += update_facts(haydate, conv, question, question_date, old_facts)
    # Can add a loop ReadAgent style, where the agent can go back to specific convs for more facts.
    answer = callgpt([{'role': 'system', 'content': f'Given the facts and dates below, please succinctly answer the question. \n\n FACTS:\n{old_facts}\n\nQUESTION: {question}\n\nQUESTION_DATE: {question_date}'}], model='gpt-4o', max_tokens=512)
    return answer
