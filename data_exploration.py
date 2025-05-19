import json
from collections import Counter


def main():
    DATA_DIR = './data/'
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


if __name__ == '__main__': main()
