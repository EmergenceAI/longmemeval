import openai, numpy as np, json, diskcache, os, random, tqdm, time
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from scipy.stats import fisher_exact

DATA_DIR = './data/'
CACHE_DIR = './cache/'

# Remove the left indent and first and last carriage returns.
def deindent(s: str) -> str:
    lines = s.split('\n')
    assert len(lines) > 1
    assert lines[0].strip() == ''
    assert lines[-1].strip() == ''
    if len(lines) == 2: return ''
    lines = lines[1:]
    # Assume the first line the the leftmost indent.
    indent = len(lines[0]) - len(lines[0].lstrip())
    # Remove up to indent spaces from each line, if they are there.
    lines = [line[min(len(line) - len(line.lstrip()), indent):] for line in lines]
    return '\n'.join(lines)

################ GPT with disk caching.
def immutify_messages(messages):
    return tuple(json.dumps(message, sort_keys=True) for message in messages)
def remutify_messages(immutable_messages):
    return [json.loads(immutable_message) for immutable_message in immutable_messages]

cache_openai = diskcache.Cache(os.path.join(CACHE_DIR, 'openai'), eviction_policy='none')
cache_openai.reset('cull_limit', 0)
@cache_openai.memoize()
def _callgpt_helper(immutable_messages, model:str, max_tokens):
    if _callgpt_helper.client is None: _callgpt_helper.client = openai.OpenAI()
    messages = remutify_messages(immutable_messages)
    response = _callgpt_helper.client.chat.completions.create(model=model, messages=messages, temperature=0., max_tokens=max_tokens)
    message = response.choices[0].message
    return message.content
_callgpt_helper.client = None

def callgpt(messages, model:str, max_tokens:int):
    immutable_messages = immutify_messages(messages)
    return _callgpt_helper(immutable_messages, model, max_tokens)

################ The evaluator
def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else: raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt

class Evaluator:
    model_zoo = {'gpt-4o-mini': 'gpt-4o-mini-2024-07-18', 'gpt-4o': 'gpt-4o-2024-08-06'}
    def __init__(self, haystacks:list[dict]):
        references = [{k: haystack[k] for k in ['answer', 'question_id', 'question', 'question_type']} for haystack in haystacks]
        self.metric_model = self.model_zoo['gpt-4o']
        self.qid2qdata = {entry['question_id']: entry for entry in references}
        self.qid2qtype = {entry['question_id']: entry['question_type'] for entry in references}
        self.qtypes = set(list(self.qid2qtype.values()))
    def evaluate(self, entry: dict) -> bool:
        # entry = {'question_id': str, 'hypothesis': str}
        assert entry['question_id'] in self.qid2qtype, 'question_id not in reference data'
        qtype = self.qid2qtype[entry['question_id']]
        q = self.qid2qdata[entry['question_id']]['question']
        ans = self.qid2qdata[entry['question_id']]['answer']
        hyp = entry['hypothesis']
        prompt = get_anscheck_prompt(qtype, q, ans, hyp, abstention='_abs' in entry['question_id'])
        eval_response = callgpt(messages=[{"role": "user", "content": prompt}], model=self.metric_model, max_tokens=10)
        label = 'yes' in eval_response.lower()
        return label

################ Evaluate with early stopping.
def run_haystack(haystack, process_func):
    question_id = haystack['question_id']
    question = haystack['question']
    question_date = haystack['question_date']
    haystack_dates = haystack['haystack_dates']
    haystack_sessions = haystack['haystack_sessions']
    hypothesis = process_func(haystack_sessions, question, question_date, haystack_dates)
    return {'question_id': question_id, 'hypothesis': hypothesis}

def fixed_shuffle(pairs, seed=42):
    rng = random.Random(seed)  # new random generator instance
    shuffled = pairs[:]        # make a copy to avoid in-place changes
    rng.shuffle(shuffled)
    return shuffled

def prob_different(hA, obsA, hB, obsB, N=100_000):
    samples_A = np.random.beta(hA + 1, obsA - hA + 1, N)
    samples_B = np.random.beta(hB + 1, obsB - hB + 1, N)
    ppAgreater = np.mean(samples_A > samples_B) # Probability P(H|A) > P(H|B)
    return max(ppAgreater, 1 - ppAgreater)  # Return the maximum of the two-tailed probabilities

def stop_early(num_success, nobs, confidence=0.95, b_successes=0, b_nobs=0, tolerance=0.05):
    if nobs < 10 or confidence > .999999999: return False  # Wilson is OK even for small nobs
    if b_nobs > 0:
        pp_value = prob_different(num_success, nobs, b_successes, b_nobs)
        return pp_value > confidence
    '''
        # Compare to B.  Return True if we're confident that A is better than B or vice versa.
        if (nobs == 0 or b_nobs == 0 or (num_success == 0 and b_successes == 0) or (num_success == nobs and b_successes == b_nobs)):
            # Guard: if both methods have same outcome (e.g., 0/100 and 0/100), test is meaningless
            return False  # Not enough information to conclude difference
        if nobs < 30 or b_nobs < 30 or min(num_success, b_successes) == 0 or num_success == nobs or b_successes == b_nobs:
            # Use Fisher's exact test for small sample sizes or edge cases
            table = [[num_success, nobs - num_success], [b_successes, b_nobs - b_successes]]
            _test_stat, p_value = fisher_exact(table, alternative='two-sided')
        else:
            # Falls back to two-proportion z-test otherwise.
            _test_stat, p_value = proportions_ztest([num_success, b_successes], [nobs, b_nobs], alternative='two-sided')
        return p_value < (1 - confidence)
    '''
    # Returns whether to stop early when the Wilson score confidence interval
    # for accuracy is within +/- tolerance at the given confidence level.
    lower, upper = proportion_confint(count=num_success, nobs=nobs, alpha=1-confidence, method='wilson')
    interval_width = upper - lower
    return interval_width <= 2 * tolerance

class Stopwatch:
    def __init__(self): self._start = None
    def start(self): self._start = time.perf_counter()
    def stop(self): return time.perf_counter() - self._start

def predict_with_early_stopping(haystacks, process_func, evaluator, confidence=0.99, b_successes=0, b_nobs=0, tolerance=0.05, verbose=False):
    """
    Processes Xs one by one, stopping early when some certainty is reached.
    1. If b_nobs > 0, compare against baseline.
    2. If b_nobs == 0, use tolerance.

    process_func: function to process the haystack, see example of "process_question" above.

    Parameters:
    - confidence: confidence level (default 0.95)

    # If comparing agains baseline, use these:
    - b_successes: number of successful outcomes from baseline
    - b_nobs: total trials from baseline
    - direction: "better", "worse", or "either" (default)

    # If no baseline, use tolerance:
    - tolerance: maximum allowed half-width of the confidence interval
    """
    num_success, nobs = 0, 0
    hypotheses = []
    stopwatch = Stopwatch()
    process_time = 0.
    for haystack in tqdm.tqdm(fixed_shuffle(haystacks)):
        stopwatch.start()
        hypothesis = run_haystack(haystack, process_func)
        process_time += stopwatch.stop()
        result = evaluator.evaluate(hypothesis)
        hypothesis['label'] = result
        hypotheses.append(hypothesis)
        num_success += result
        nobs += 1
        if verbose:
            tqdm.tqdm.write(f'\nQuestion: {haystack["question"]}')
            tqdm.tqdm.write(f'Hypothesis: {hypothesis["hypothesis"]}')
            tqdm.tqdm.write(f'Ground truth: {haystack["answer"]}')
            tqdm.tqdm.write(f'Processed {nobs} trials with {num_success} successes.  The last result was {result}.')
        if stop_early(num_success, nobs, confidence, b_successes, b_nobs, tolerance):
            tqdm.tqdm.write(f'Stopping early at {nobs} trials with {num_success} successes.')
            if b_nobs > 0: tqdm.tqdm.write(f'Current model is {"BETTER" if num_success / nobs > b_successes / b_nobs else "WORSE"} THAN baseline.')
            break
    return hypotheses, num_success, nobs, process_time

################ Hacka hacka

def predict_with_early_stopping_two_step(haystacks, process_haystack, process_question, evaluator, confidence=0.99, b_successes=0, b_nobs=0, tolerance=0.05, verbose=False):
    """
    Processes Xs one by one, stopping early when some certainty is reached.
    1. If b_nobs > 0, compare against baseline.
    2. If b_nobs == 0, use tolerance.

    process_func: function to process the haystack, see example of "process_question" above.

    Parameters:
    - confidence: confidence level (default 0.95)

    # If comparing agains baseline, use these:
    - b_successes: number of successful outcomes from baseline
    - b_nobs: total trials from baseline
    - direction: "better", "worse", or "either" (default)

    # If no baseline, use tolerance:
    - tolerance: maximum allowed half-width of the confidence interval
    """
    num_success, nobs = 0, 0
    hypotheses = []
    stopwatch = Stopwatch()
    haystack_time, question_time = 0., 0.
    for haystack in tqdm.tqdm(fixed_shuffle(haystacks)):
        question_id = haystack['question_id']
        question = haystack['question']
        question_date = haystack['question_date']
        haystack_dates = haystack['haystack_dates']
        haystack_sessions = haystack['haystack_sessions']
        stopwatch.start()
        memstruct = process_haystack(haystack_sessions, haystack_dates)
        haystack_time += stopwatch.stop()
        stopwatch.start()
        guess = process_question(memstruct, question, question_date)
        question_time += stopwatch.stop()
        hypothesis = {'question_id': question_id, 'hypothesis': guess}
        result = evaluator.evaluate(hypothesis)
        hypothesis['label'] = result
        hypotheses.append(hypothesis)
        num_success += result
        nobs += 1
        if verbose:
            tqdm.tqdm.write(f'\nQuestion: {haystack["question"]}')
            tqdm.tqdm.write(f'Hypothesis: {hypothesis["hypothesis"]}')
            tqdm.tqdm.write(f'Ground truth: {haystack["answer"]}')
            tqdm.tqdm.write(f'Processed {nobs} trials with {num_success} successes.  The last result was {result}.')
        if stop_early(num_success, nobs, confidence, b_successes, b_nobs, tolerance):
            tqdm.tqdm.write(f'Stopping early at {nobs} trials with {num_success} successes.')
            if b_nobs > 0: tqdm.tqdm.write(f'Current model is {"BETTER" if num_success / nobs > b_successes / b_nobs else "WORSE"} THAN baseline.')
            break
    return hypotheses, num_success, nobs, haystack_time, question_time


################ Print some stats
# This basically just prints out the model stats.  It can do the evaluation, but it shouldn't.
def evaluate_qa(hypotheses:list[dict], evaluator:Evaluator):
    qtype2acc = {t: [] for t in evaluator.qtypes}
    logs = []
    for entry in hypotheses:
        assert 'label' in entry, 'label not in entry...'
        if 'label' not in entry:
            print(f'Wha lao eh?  Label not in entry! entry: {entry}\nWARNING: Evaluating label here.')
            entry['label'] = evaluator.evaluate(entry)
        label = entry['label']
        logs.append(label)
        qtype2acc[evaluator.qid2qtype[entry['question_id']]].append(1 if label else 0)
    metrics = ''
    print('Accuracy:', round(np.mean(logs).item(), 4))
    metrics += f'Accuracy: {round(np.mean(logs).item(), 4)}\n'
    for k,v in sorted(qtype2acc.items()):
        metrics += f'{k:<27}: {round(np.mean(v), 4):>6.2%} ({len(v)} obs)\n'
        print(f'\t{k:<27}: {round(np.mean(v), 4):>6.2%} ({len(v)} obs)')
    return metrics

################ Logging stuff
import uuid
from datetime import datetime
from git import Repo, InvalidGitRepositoryError

class DumbLogger:
    def __init__(self, repo_path, log_dir, module_name, func_name):
        # Ensure the git repository is up to date.
        repo = Repo(repo_path, search_parent_directories=True)
        os.makedirs(log_dir, exist_ok=True)
        if repo.is_dirty():
            raise RuntimeError("Git working directory is dirty. Please commit before running.")
        try:
            repo = Repo(repo_path, search_parent_directories=True)
            commit = repo.head.commit.hexsha
            dirty = repo.is_dirty()
            branch = repo.active_branch.name
        except InvalidGitRepositoryError: raise RuntimeError("Not a git repository.")
        except Exception as e: raise RuntimeError(f"Failed to get git info: {e}")
        self.git_info = {"commit": commit, "dirty": dirty, "branch": branch}
        self.start_time = datetime.now()
        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.log_dir = log_dir
        self.description = f"{module_name}.{func_name}"
    def log_it_up(self, metrics):
        end_time = datetime.now()
        git_info, run_id, log_dir = self.git_info, self.run_id, self.log_dir
        data = {"id": run_id, "timestamp": datetime.now().isoformat(), "description": self.description, "git": git_info, "metrics": metrics, "total_time": (end_time - self.start_time).total_seconds()}
        out_path = os.path.join(log_dir, f"{run_id}.json")
        with open(out_path, "w") as f: json.dump(data, f, indent=2)
        print(f"[+] Logged run {run_id} to {out_path}")
