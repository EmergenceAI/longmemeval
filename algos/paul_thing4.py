#!/usr/bin/env python
# Read paul_20250523.xlsx and convert to json structure.
import pandas as pd, json
from utils import callgpt

PAUL_FILENAME = './algos/paul_thing_20250604.xlsx'
PAULKEY = 'grounded'
PAULKEY2 = 'sentence'
SHEETNAME = 'facts'

# The test data filename
DATA_DIR = './data/'
FILENAME = 'longmemeval_s.json' # longmemeval_m.json, longmemeval_oracle.json

# Making a class so we don't have to reload with each call to process_question.
class PaulThing:
    def __init__(self):
        print('Loading PaulThing...')
        excel_file = pd.ExcelFile(PAUL_FILENAME)
        # Read all sheets into a dictionary
        sheet_dict = {}
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            data_dict = df.to_dict(orient='records')
            sheet_dict[sheet_name] = data_dict
        # Get the extracted sessions.
        self.sessions = {}
        known_keys = {'session', 'sentence', 'grounded'}
        # ['items', 'sessions', 'fact', 'episodes']
        for extract in sheet_dict[SHEETNAME]:
            if extract['session'] not in self.sessions:
                self.sessions[extract['session']] = []
            assert set(extract.keys()) >= known_keys, f"Keys are not as expected: {extract.keys()}"
            new_extract = {k: extract[k] for k in extract.keys() if k != 'session'}
            self.sessions[extract['session']].append(new_extract)
        # Need to map h to list of sessions and dates.
        # Hack: use question as key.  (Assert that each question is unique.)
        questions = [item['question'] for item in sheet_dict['items']]
        assert len(questions) == len(set(questions)), "Questions are not unique!"
        # q_dict = {item['question']: item for item in sheet_dict['items']}
        # Total hack to get session ids:
        haystacks = json.load(open(f'{DATA_DIR}/{FILENAME}'))
        self.q_info = {h['question']: {'haystack_session_ids': h['haystack_session_ids'], 'haystack_dates': h['haystack_dates']} for h in haystacks}
        print(f"Loaded {len(self.sessions)} sessions.")

    def process_question(self, haystack_sessions: list[list[dict]], question: str, question_date:str, haystack_dates: list[str]) -> str:
        # Note: haystack_sessions isn't used since we get the sessions from self.
        # Get the relevant session information
        haystack_session_ids, haystack_dates_2 = self.q_info[question]['haystack_session_ids'], self.q_info[question]['haystack_dates']
        assert haystack_dates == haystack_dates_2, f"Haystack dates do not match: {haystack_dates} != {haystack_dates_2}"
        # Sort the ids by date.
        assert len(haystack_session_ids) == len(haystack_dates), f"Haystack session ids and dates do not match: {len(haystack_session_ids)} != {len(haystack_dates)}"
        ids_dates = list(zip(haystack_session_ids, haystack_dates))
        ids_dates.sort(key=lambda x: x[1])
        knowledge = []
        for idx, date in ids_dates:
            if idx not in self.sessions: continue
            knowledge.append((date, self.sessions[idx]))
        ################ Done processing..  Now let's use the knowledge.
        # OK, let's just get the grounded sentences.
        groundeds = []
        for date, session in knowledge:
            facts = []
            for extract in session:
                assert PAULKEY in extract, f"Extract does not have '{PAULKEY}' key: {extract}"
                fact = extract[PAULKEY]
                if type(fact) is not str:
                    assert PAULKEY2 in extract, f"Extract does not have '{PAULKEY2}' key: {extract}"
                    fact = extract[PAULKEY2]
                    if type(fact) is not str:
                        print(f"Fact is not a string: {fact}")
                        continue
                facts.append(extract[PAULKEY])
            groundeds.append((date, facts))
        ################
        # (A good place to refactor here.)
        # The Almighty says don't change the subject, just answer the fookin question.
        # Hack: Do a simple deduping when we have the same fact twice in a row.
        groundeds2 = []
        for date, facts in groundeds:
            facts2 = []
            for fact in facts:
                if len(facts2) == 0 or facts2[-1] != fact:
                    facts2.append(fact)
            groundeds2.append((date, facts2))
        allfacts = ''
        for date, facts in groundeds2:
            allfacts += f'FACTS FROM CONVERSATION ON {date}:'
            for fact in facts: allfacts += f'\n    {fact}'
            allfacts += '\n\n'
        answer = callgpt([{'role': 'system', 'content': f'Given the facts and dates below, please succinctly answer the question. \n\n FACTS:\n{allfacts}\n\nQUESTION: {question}\n\nQUESTION_DATE: {question_date}'}], model='gpt-4o', max_tokens=2048)
        return answer

def process_haystack(haystack_sessions: list[list[dict]], haystack_dates: list[str]):
    if process_question.pfunc is None:
        # Make my func the pfunc.  I wants to get func-ed up.
        process_question.pfunc = PaulThing()
    return haystack_sessions, haystack_dates, process_question.pfunc

def process_question(memstruct, question: str, question_date: str) -> str:
    haystack_sessions, haystack_dates, pfunc = memstruct
    return pfunc.process_question(haystack_sessions, question, question_date, haystack_dates)
process_question.pfunc = None
