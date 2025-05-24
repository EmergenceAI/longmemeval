import json
from collections import Counter

DATA_DIR = './data/'

def main():
    filename = 'longmemeval_s.json'
    haystacks = json.load(open(f'{DATA_DIR}/{filename}'))

    total_turns = 0
    total_convs = 0
    max_turns = 0
    max_conv = 0
    empties = 0
    dupturns = 0
    ddates = []
    for i, haystack in enumerate(haystacks):
        sessions = haystack['haystack_sessions']
        if len(sessions) > max_conv: max_conv = len(sessions)
        total_convs += len(sessions)
        hashable_sessions = [tuple((turn['role'], turn['content']) for turn in session) for session in sessions]
        newturns = sum(len(x) for x in hashable_sessions)
        if newturns > max_turns: max_turns = newturns
        # Get the latest date
        haystack_dates = sorted(haystack['haystack_dates'])
        last_date = haystack_dates[-1]
        question_date = haystack['question_date']
        # Check if the last date is before the question date
        ddates.append((question_date, last_date))
        total_turns += newturns
        counts = Counter(hashable_sessions)
        empties += counts[()]
        for hashable_session, count in counts.items():
            if count > 1 and len(hashable_session) > 0:
                dupturns += len(hashable_session)
                print(f"Duplicate session found in haystack {i}: {len(hashable_session)} (count: {count})")
    print(f'Total turns: {total_turns}, max turns: {max_turns}, total convs: {total_convs}, max convs: {max_conv}')
    print(dupturns)


    dcount = 0
    for i, (question_date, last_date) in enumerate(ddates):
        if last_date > question_date:
            dcount += 1
            print(f"Haystack {i} has a date after the question date: {last_date} > {question_date}")
            assert last_date[:10] == question_date[:10], 'huh?'
    print(f"Total haystacks with date after question date: {dcount}")

def main2():
    filename_s = 'longmemeval_s.json'
    filename_m = 'longmemeval_m.json'

    haystacks_s = json.load(open(f'{DATA_DIR}/{filename_s}'))
    haystacks_m = json.load(open(f'{DATA_DIR}/{filename_m}'))

    all_questions_s = [haystack['question'] for haystack in haystacks_s]
    all_questions_m = [haystack['question'] for haystack in haystacks_m]
    assert len(all_questions_s) == len(set(all_questions_s)), "Duplicate questions in longmemeval_s.json"
    assert set(all_questions_s) == set(all_questions_m), "Questions in longmemeval_s.json and longmemeval_m.json do not match"

    # Sort haystacks by question
    haystacks_s = sorted(haystacks_s, key=lambda x: x['question'])
    haystacks_m = sorted(haystacks_m, key=lambda x: x['question'])

    temp_counta = Counter()
    temp_countb = Counter()
    conv_counta = Counter()
    conv_countb = Counter()
    for i, (haystack_s, haystack_m) in enumerate(zip(haystacks_s, haystacks_m)):
        sessions_s = haystack_s['haystack_sessions']
        sessions_m = haystack_m['haystack_sessions']
        assert haystack_s['question'] == haystack_m['question'], f"Haystack {i} has different questions: {haystack_s['question']} != {haystack_m['question']}"
        # assert haystack_s['question_date'] == haystack_m['question_date'], f"Haystack {i} has different question dates: {haystack_s['question_date']} != {haystack_m['question_date']}"
        # print(len(sessions_m), len(sessions_s))
        dates_s = haystack_s['haystack_dates']
        dates_m = haystack_m['haystack_dates']
        temp_counta.update(dates_s)
        temp_countb.update(dates_m)
        # This triggers for both s and m
        # assert len(set(dates_s)) == len(dates_s), f"Haystack {i} has duplicate dates in longmemeval_s.json"
        # assert len(set(dates_m)) == len(dates_m), f"Haystack {i} has duplicate dates in longmemeval_m.json"
        conv_counta.update(str(conv) for conv in sessions_s)
        conv_countb.update(str(conv) for conv in sessions_m)
        #
        zipped_s = list(zip(dates_s, sessions_s))
        zipped_m = list(zip(dates_m, sessions_m))
        #
        # See if zipped_s is a subset of zipped_m
        zipped_s_set = {str(x) for x in zipped_s}
        zipped_m_set = {str(x) for x in zipped_m}
        assert zipped_s_set <= zipped_m_set, f"Haystack {i} has different sessions"


def main3():

    #
    allsessions = {} # map id to session
    for haystack in haystacks:
        haystack_sessions = haystack['haystack_sessions']
        haystack_session_ids = haystack['haystack_session_ids']
        assert len(haystack_sessions) == len(haystack_session_ids), f"Haystack session ids and dates do not match: {len(haystack_sessions)} != {len(haystack_session_ids)}"
        for idx, session in zip(haystack_session_ids, haystack_sessions):
            if idx not in allsessions: allsessions[idx] = session
            else:
                assert allsessions[idx] == session, f"Session {idx} does not match: {allsessions[idx]} != {session}"

    missing = open('../missing_sessions.txt', 'r').readlines()
    missing = [x.strip() for x in missing]

    for missed in missing:
        assert missed in allsessions, f"Missing session {missing} not in allsessions."
        session = allsessions[missed]
        if len(session) > 0:
            print(f"Turns is {len(session)}, {missed}")
    

if __name__ == '__main__': main()
