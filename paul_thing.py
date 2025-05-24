#!/usr/bin/env python
# Read paul_20250523.xlsx and convert to json structure.
import pandas as pd, json
from utils import callgpt

PAUL_FILENAME = 'paul_thing_20250523.xlsx'
SHEETNAME = 'Marc'

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
        for extract in sheet_dict[SHEETNAME]:
            if extract['session'] not in self.sessions:
                self.sessions[extract['session']] = []
            assert set(extract.keys()) == known_keys, f"Keys are not as expected: {extract.keys()}"
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
        #
        knowledge = []
        for idx, date in ids_dates:
            if idx not in self.sessions:
                print(f"Warning: Session {idx} not found in sessions.")
                continue
            knowledge.append((date, self.sessions[idx]))
        ################ Done processing..  Now let's use the knowledge.
        # OK, let's just get the grounded sentences.
        groundeds = []
        for date, session in knowledge:
            facts = []
            for extract in session:
                assert 'grounded' in extract, f"Extract does not have 'grounded' key: {extract}"
                facts.append(extract['grounded'])
            groundeds.append((date, facts))
        # Now we have a list, sorted by date with each item like this.
        # ('2023/05/20 (Sat) 02:21',
        #  ['The farmer needs to transport a fox, a chicken, and some grain across a river using a boat.',
        #   'The fox cannot be left alone with the chicken.',
        #   ...
        #  ])
        #
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

def test():
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

################################################################
import os, inspect, hashlib
from utils import Evaluator, predict_with_early_stopping, evaluate_qa

def main():
    paul_thing = PaulThing()
    process_func = paul_thing.process_question
    filename = 'longmemeval_s.json' # longmemeval_m.json, longmemeval_oracle.json
    assert filename == FILENAME, f"Filename {filename} does not match {FILENAME}."
    # check if file exists
    if not os.path.exists(f'{DATA_DIR}/{filename}'):
        print(f'File {filename} not found in {DATA_DIR}.  Download these from the LongMemEval source.')
    haystacks = json.load(open(f'{DATA_DIR}/{filename}'))
    evaluator = Evaluator(haystacks)
    #
    # Early stopping parameters
    # confidence = 0.99 # Set to 1 to disable early stopping
    confidence = 1.
    b_successes = 0
    b_nobs = 0
    tolerance = 0.05 # Not used if b_nobs > 0
    #
    # Print stuff about the process function using inspect.
    print(f'process_func name: {process_func.__name__}')
    print(f'process_func docstring: {inspect.getdoc(process_func)}')
    print(f'process_func source hash: {hashlib.sha256(inspect.getsource(process_func).encode()).hexdigest()}')
    print(f'process_func module: {process_func.__module__}')
    print(f'process_func file: {inspect.getfile(process_func)}')
    #
    hypotheses, num_success, nobs = predict_with_early_stopping(haystacks, process_func, evaluator, confidence, b_successes, b_nobs, tolerance, verbose=False)
    print(f'Evaluated {nobs} hypotheses with {num_success} successes.  Accuracy: {num_success / nobs:.4f}')
    evaluate_qa(hypotheses, evaluator)

if __name__ == '__main__': main()
