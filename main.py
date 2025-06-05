# from algos.example_function import hello_world_process_question as process_question
# from algos.example_guessbot import hello_world_process_question as process_question
# from algos.example_prakhar_bot import process_question
# from algos.example_rest_api import call_rest_endpoint as process_question
# from algos.example_jeremy_mem import process_question
# from algos.example_fullcontext import process_question
# from algos.example_memcheat import process_question
# from algos.paul_thing import process_question
# from algos.paul_thing3 import process_haystack, process_question
# from algos.PrakharLTM_V3_two_step import process_question, process_haystack
# from algos.full_o1 import process_question, process_haystack
# from algos.paul_thing4 import process_question, process_haystack
from algos.full_o3 import process_question, process_haystack

### Edit ^^^ to point to your function.

import json, os
# from utils import Evaluator, predict_with_early_stopping, evaluate_qa, DumbLogger
from utils import Evaluator, predict_with_early_stopping_two_step, evaluate_qa, DumbLogger

DATA_DIR = './data/'
REPO_PATH = '.'
LOG_DIR = 'runs'

def run_expt():
    filename = 'longmemeval_s.json' # longmemeval_m.json, longmemeval_oracle.json
    # check if file exists
    if not os.path.exists(f'{DATA_DIR}/{filename}'):
        print(f'File {filename} not found in {DATA_DIR}.  Download these from the LongMemEval source.')
    haystacks = json.load(open(f'{DATA_DIR}/{filename}'))
    evaluator = Evaluator(haystacks)
    # Early stopping parameters
    confidence = .9999 # Set to 1 to disable early stopping
    b_successes = 350
    b_nobs = 500
    tolerance = 0.05 # Not used if b_nobs > 0
    #
    # hypotheses, num_success, nobs, process_time = predict_with_early_stopping(haystacks, process_question, evaluator, confidence, b_successes, b_nobs, tolerance, verbose=False)
    hypotheses, num_success, nobs, haystack_time, question_time = predict_with_early_stopping_two_step(haystacks, process_haystack, process_question, evaluator, confidence, b_successes, b_nobs, tolerance, verbose=True)
    print(f'Evaluated {nobs} hypotheses with {num_success} successes.  Accuracy: {num_success / nobs:.4f}')
    metrics = evaluate_qa(hypotheses, evaluator)
    return metrics + f'Haystack time: {haystack_time/nobs:.4f}s/it\nQuestion time: {question_time/nobs:.4f}s/it'

def main():
    # This must be run *before* run_expt() so that the logger is initialized with the proper checkpoint and timestamps.
    dumblogger = DumbLogger(REPO_PATH, LOG_DIR, process_question.__module__, process_question.__name__)
    metrics = run_expt()
    dumblogger.log_it_up(metrics)

if __name__ == '__main__': main()
