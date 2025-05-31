from utils import callgpt

SYSPROMPT = """
You are a helpful assistant that is helping to gather data to answer a question based on a series of conversations.
Given the question, the list of previous dated facts, and the current conversation, please create new facts that are relevant to the question.
The given facts are extracted from previous conversations.  There also may be future conversations.
Please keep the facts in chronological order, when relevant.
Note the date of the question.
Do not repeat the previous facts, but do call out changes or updates to them when it might change the answer with UPDATE.
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

LESSON_SCHEMA = {
    "entries": [
        {
            "description": "str: A description of the fact.",
        }
    ]
}

def update_facts(haydate, conv, question, question_date, old_facts):
    systemprompt = SYSPROMPT.format(
        # schema=json.dumps(LESSON_SCHEMA, indent=2),
        question=question,
        question_date=question_date,
        facts=old_facts if old_facts else 'None',
        conversation_date=haydate,
        conversation='\n\n'.join([f"{turn['role']}: {turn['content']}" for turn in conv])
    )
    # response = callgpt([{'role': 'system', 'content': systemprompt}], model='gpt-4o', response_format={'type':'json_object'})
    response = callgpt([{'role': 'system', 'content': systemprompt}], model='gpt-4o', max_tokens=2048)
    # Split on FACT: and strip the leading and trailing whitespace
    # Replace newlines in response with "\n  "
    new_facts = f"\n\nFACTS FROM {haydate}:\n\n  {response.replace('\n', '\n  ')}\n"
    return new_facts

def process_question(haystack_sessions: list[list[dict]], question: str, question_date:str, haystack_dates: list[str]) -> str:
    # Check sessions for duplicate sessions
    if False:
        dupes = []
        for i in range(len(haystack_sessions)):
            for j in range(i + 1, len(haystack_sessions)):
                if haystack_dates[i] == haystack_dates[j]:
                    dupes.append((i, j))
        if dupes:
            print(f"DUPLICATE SESSIONS FOUND: {dupes}")
            # Reconstruct a new haystack without the duplicates
            delenda = {j if i < j else i for i, j in dupes}
            new_haystack = []
            new_timestamps = []
            for i in range(len(haystack_sessions)):
                if i not in delenda:
                    new_haystack.append(haystack_sessions[i])
                    new_timestamps.append(haystack_dates[i])
            haystack_sessions = new_haystack
            haystack_dates = new_timestamps
            # Results when duplicates are removed:
            # Evaluated 500 hypotheses with 403 successes.  Accuracy: 0.8060
            # Accuracy: 0.806
            #         single-session-user        : 95.71% (70 obs)
            #         multi-session              : 75.19% (133 obs)
            #         temporal-reasoning         : 74.44% (133 obs)
            #         knowledge-update           : 91.03% (78 obs)
            #         single-session-assistant   : 96.43% (56 obs)
            #         single-session-preference  : 40.00% (30 obs)
    # Process the haystack in light of the question.
    # Sort the haystack sessions by date
    haystack_sessions2 = list(zip(haystack_dates, haystack_sessions))
    haystack_sessions2.sort(key=lambda x: x[0])
    old_facts = ''
    # Iterate over the sessions and update the facts
    # for haydate, conv in haystack_sessions2:
    i = 0
    while i < len(haystack_sessions2):
        haydate, conv = haystack_sessions2[i]
        new_facts = update_facts(haydate, conv, question, question_date, old_facts)
        old_facts += new_facts
        # print(f"Updated facts for {haydate}:\n{new_facts}")
        i += 1

    # Can add a loop ReadAgent style, where the agent can go back to specific convs for more facts.
    answer = callgpt([{'role': 'system', 'content': f'Given the facts and dates below, please succinctly answer the question. \n\n FACTS:\n{old_facts}\n\nQUESTION: {question}\n\nQUESTION_DATE: {question_date}'}], model='gpt-4o', max_tokens=2048)
    return answer

def test():
    import json
    DATA_DIR = './data/'
    filename = 'longmemeval_s.json' # longmemeval_m.json, longmemeval_oracle.json
    haystacks = json.load(open(f'{DATA_DIR}/{filename}'))
    haystack = haystacks[0]
    question_id = haystack['question_id']
    question = haystack['question']
    question_date = haystack['question_date']
    haystack_dates = haystack['haystack_dates']
    haystack_sessions = haystack['haystack_sessions']
    answer = haystack['answer']
