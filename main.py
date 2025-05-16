from example_function import hello_world_process_question as process_func
# from example_guessbot import hello_world_process_question as process_func
# Edit ^^^ to point to your function.

import json, os, inspect, hashlib
from utils import Evaluator, predict_with_early_stopping, evaluate_qa

DATA_DIR = './data/'

def main():
    filename = 'longmemeval_s.json' # longmemeval_m.json, longmemeval_oracle.json
    # check if file exists
    if not os.path.exists(f'{DATA_DIR}/{filename}'):
        print(f'File {filename} not found in {DATA_DIR}.  Download these from the LongMemEval source.')
    haystacks = json.load(open(f'{DATA_DIR}/{filename}'))
    evaluator = Evaluator(haystacks)
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
    hypotheses, num_success, nobs = predict_with_early_stopping(haystacks, process_func, evaluator, confidence, b_successes, b_nobs, tolerance)
    print(f'Evaluated {nobs} hypotheses with {num_success} successes.  Accuracy: {num_success / nobs:.4f}')
    evaluate_qa(hypotheses, evaluator)

if __name__ == '__main__': main()
