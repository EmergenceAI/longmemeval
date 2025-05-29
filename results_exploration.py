import json, os, inspect, hashlib
from utils import Evaluator, predict_with_early_stopping, evaluate_qa

DATA_DIR = './data/'

def bigmess():
    # Run all the things!
    # A big, big mess, he was all mixed up in a big mess.
    from example_function import hello_world_process_question as process_func_dont_know
    from example_guessbot import hello_world_process_question as process_func_guessbot
    from example_jeremy_mem import hello_world_process_question as process_func_jeremy_mem
    from example_fullcontext import process_question as process_func_fullcontext
    from example_memcheat import process_question as process_func_memcheat
    from paul_thing import process_question as process_func_paul_thing

    all_process_funcs = {
        'dont_know': process_func_dont_know,
        'guessbot': process_func_guessbot,
        'jeremy_mem': process_func_jeremy_mem,
        'fullcontext': process_func_fullcontext,
        'memcheat': process_func_memcheat,
        'paul_thing': process_func_paul_thing,
    }

    import numpy as np
    filename = 'longmemeval_s.json' # longmemeval_m.json, longmemeval_oracle.json
    haystacks = json.load(open(f'{DATA_DIR}/{filename}'))
    evaluator = Evaluator(haystacks)

    all_hypotheses = {}
    confidence = 1.
    b_successes = 0
    b_nobs = 0
    tolerance = 0.05 # Not used if b_nobs > 0
    for func_name, pfunc in all_process_funcs.items():
        hypotheses, num_success, nobs = predict_with_early_stopping(haystacks, pfunc, evaluator, confidence, b_successes, b_nobs, tolerance, verbose=False)
        for hypo in hypotheses:
            qid = hypo['question_id']
            if qid not in all_hypotheses: all_hypotheses[qid] = {}
            assert func_name not in all_hypotheses[qid]
            assert hypo['label'] in [True, False]
            all_hypotheses[qid][func_name] = hypo['label']

    qid_to_question = {haystack['question_id']: haystack['question'] for haystack in haystacks}

    # Fill the matrix with 1s and 0s.  Use -1 for missing values.
    all_qids = list(all_hypotheses.keys())
    all_labels = list(all_hypotheses[all_qids[0]].keys())
    all_matrix = np.zeros((len(all_qids), len(all_labels)), dtype=int)
    all_matrix.fill(-1)
    for i, qid in enumerate(all_qids):
        for j, label in enumerate(all_labels):
            if label in all_hypotheses[qid]:
                all_matrix[i][j] = all_hypotheses[qid][label]
    # Count -1 values
    num_missing = np.sum(all_matrix == -1)
    print(f'Number of missing values: {num_missing}')

    # Bin the questions according to results.
    bins = {}
    for question, q_results in all_hypotheses.items():
        # Put the results in a canonical, hashable form.
        q_hashable = tuple(sorted(q_results.items()))
        if q_hashable not in bins: bins[q_hashable] = []
        bins[q_hashable].append(question)
    # Print the bins as a table.
    columns = list(all_labels) + ['Count']
    print(' | '.join(columns))
    print('-' * 80)
    for q_results, questions in bins.items():
        # Print the results as a row.
        row = [('X' if x[1] else '.') for x in q_results] + [str(len(questions))]
        print(' | '.join(row))


    # Let's look at the results for paul_thing vs fullcontext.
    # Questions they both got right.
    m1, m2 = 'paul_thing', 'fullcontext'
    both_right = []
    both_wrong = []
    m1_worse = []
    m1_better = []
    for question, q_results in all_hypotheses.items():
        assert m1 in q_results and m2 in q_results
        if q_results[m1] == True and q_results[m2] == True: both_right.append(question)
        elif q_results[m1] == False and q_results[m2] == False: both_wrong.append(question)
        elif q_results[m1] == True and q_results[m2] == False: m1_better.append(question)
        elif q_results[m1] == False and q_results[m2] == True: m1_worse.append(question)
        else: assert False, f'Unexpected results: {q_results}'

    print(f'Both right: {len(both_right)}')
    print(f'Both wrong: {len(both_wrong)}')
    print(f'{m1} better: {len(m1_better)}')
    print(f'{m2} better: {len(m1_worse)}')

    print(f'Questions where {m1} does better than {m2}:')
    for question in m1_better:
        print(f'{qid_to_question[question]}')

    print(f'Questions where {m2} does better than {m1}:')
    for question in m1_worse:
        print(f'{qid_to_question[question]}')

    # both wrong
    print('Questions where both are wrong:')
    for question in both_wrong:
        print(f'{qid_to_question[question]}')
