# from example_function import hello_world_process_question as process_func
# from example_guessbot import hello_world_process_question as process_func
# from example_rest_api import call_rest_endpoint as process_func
# from example_jeremy_mem import hello_world_process_question as process_func
# from example_fullcontext import process_question as process_func
# from example_memcheat import process_question as process_func
# from paul_thing import process_question as process_func
# from example_prakhar_bot import process_question as process_func
# from example_prakharbot_v2 import process_question as process_func
# from example_prakhar_bot_v3 import process_question as process_func, extract_facts_only
# from example_prakhar_bot_v5 import process_question as process_func


# # Edit ^^^ to point to your function.
# from pathlib import Path
# import json, os, inspect, hashlib
# from utils import Evaluator, predict_with_early_stopping, evaluate_qa
#
# DATA_DIR = '/home/pdx/Desktop/longmemeval/data/'
#
# def main():
#
#     filename = 'longmemeval_s.json' # longmemeval_m.json, longmemeval_oracle.json
#     # check if file exists
#     if not os.path.exists(f'{DATA_DIR}/{filename}'):
#         print(f'File {filename} not found in {DATA_DIR}.  Download these from the LongMemEval source.')
#     print(f"Loading file: {filename}")
#     haystacks = json.load(open(f'{DATA_DIR}/{filename}'))
#     evaluator = Evaluator(haystacks)
#     # Early stopping parameters
#     # confidence = 0.99 # Set to 1 to disable early stopping
#     confidence = 1.
#     b_successes = 0
#     b_nobs = 0
#     tolerance = 0.05 # Not used if b_nobs > 0
#     #
#     # Print stuff about the process function using inspect.
#     print(f'process_func name: {process_func.__name__}')
#     print(f'process_func docstring: {inspect.getdoc(process_func)}')
#     print(f'process_func source hash: {hashlib.sha256(inspect.getsource(process_func).encode()).hexdigest()}')
#     print(f'process_func module: {process_func.__module__}')
#     print(f'process_func file: {inspect.getfile(process_func)}')
#     #
#     hypotheses, num_success, nobs = predict_with_early_stopping(haystacks, process_func, evaluator, confidence, b_successes, b_nobs, tolerance, verbose=False)
#     print(f'Evaluated {nobs} hypotheses with {num_success} successes.  Accuracy: {num_success / nobs:.4f}')
#     evaluate_qa(hypotheses, evaluator)
#
# if __name__ == '__main__': main()

'''------------------main for one haystack code for PrakharLTM-----------------------------------'''
# main.py

# from PrakharLTM_onehaystack import process_question as process_func
from PrakharLTM_v2 import process_question as process_func
from pathlib import Path
import json, os, inspect, hashlib
from utils import Evaluator, predict_with_early_stopping, evaluate_qa

DATA_DIR = os.path.expanduser("~/Desktop/longmemeval/data/")

def main():
    filename = 'longmemeval_s.json'  # change to longmemeval_m.json or _oracle.json if needed

    # Check if file exists
    full_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(full_path):
        print(f'File {filename} not found in {DATA_DIR}.  Download these from the LongMemEval source.')
        return

    print(f"Loading file: {filename}")
    with open(full_path, 'r') as f:
        haystacks = json.load(f)

    # Setup evaluation logic
    evaluator = Evaluator(haystacks)

    # Early stopping config
    confidence = 0.999        # Set to 1 to disable early stopping
    b_successes = 0
    b_nobs = 0
    tolerance = 0.05         # Not used if b_nobs > 0

    # Display process_func metadata
    print(f'process_func name: {process_func.__name__}')
    print(f'process_func docstring: {inspect.getdoc(process_func)}')
    print(f'process_func source hash: {hashlib.sha256(inspect.getsource(process_func).encode()).hexdigest()}')
    print(f'process_func module: {process_func.__module__}')
    print(f'process_func file: {inspect.getfile(process_func)}')

    # Evaluate predictions
    hypotheses, num_success, nobs = predict_with_early_stopping(
        haystacks,
        process_func,
        evaluator,
        confidence,
        b_successes,
        b_nobs,
        tolerance,
        verbose=True
    )

    print(f'Evaluated {nobs} hypotheses with {num_success} successes. Accuracy: {num_success / nobs:.4f}')
    evaluate_qa(hypotheses, evaluator)

if __name__ == '__main__':
    main()
