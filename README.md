# longmemeval

## To run, 
1. Download the data from [LongMemEval](https://github.com/xiaowu0162/LongMemEval?tab=readme-ov-file).
2. Install requirements from requirements.txt

```
/longmemeval(M) python3 ./main.py 
process_func name: hello_world_process_question
process_func docstring: None
process_func source hash: eefb6ae7ceca1c0206caab9d2ae662a19b62eda0ac7d5ab32ebc0cbf5e8dd251
process_func module: example_function
process_func file: /home/mpickett/longmemeval/example_function.py
Stopping early at 179 trials with 12 successes.                                                                                                                                        
 36%|██████████████████████████████████████████████████▌                                                                                           | 178/500 [00:00<00:00, 8218.60it/s]
Evaluated 179 hypotheses with 12 successes.  Accuracy: 0.0670
Accuracy: 0.067
        multi-session              : 14.00% (50 obs)
        single-session-assistant   :  0.00% (12 obs)
        temporal-reasoning         :  2.13% (47 obs)
        knowledge-update           :  3.33% (30 obs)
        single-session-preference  :  0.00% (15 obs)
        single-session-user        : 12.00% (25 obs)
```
