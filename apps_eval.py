import os 
os.environ['HF_DATASETS_OFFLINE'] = '1'

import itertools
import numpy as np
import datasets 
from collections import Counter 
from argparse import ArgumentParser
from datasets import (disable_caching, 
                      disable_progress_bar,
                      load_dataset)
from utils import estimate_pass_at_k


datasets.utils.logging.set_verbosity_error()
# disable_caching()
disable_progress_bar()


DATASET = "codeparrot/apps"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--result-path", type=str, required=True,
                        help="Directory or file path storing execution results")
    parser.add_argument("--k-list", type=int, nargs="+", default=[1, 5, 100],
                        help="A space separated list of values")
    parser.add_argument("--level-list", type=str, nargs="+",
                        default=["introductory", "interview", "competition", "all"])
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"])
    args = parser.parse_args()

    if os.path.isdir(args.result_path):
        data_files = [os.path.join(args.result_path, f) for f in os.listdir(args.result_path)
                      if os.path.isfile(os.path.join(args.result_path, f))]
    elif os.path.isfile(args.result_path):
        data_files = [args.result_path]
        
    dataset = load_dataset("json", data_files=data_files, split="train")
        
    for level in args.level_list:
        eval_dataset = load_dataset(DATASET, split=args.split, difficulties=[level])
        problem_ids = set(eval_dataset['problem_id'])
    
        samples = dataset.filter(lambda x: x["task_id"] in problem_ids)
        correct = samples.filter(lambda x: x["passed"])
        
        num_samples = Counter(samples["task_id"])
        num_correct = Counter(correct["task_id"])
        
        task_ids = num_samples.keys()
        num_samples = np.array([num_samples[k] for k in task_ids])
        num_correct = np.array([num_correct[k] for k in task_ids])
        
        for k in args.k_list:
            if np.all(num_samples >= k):
                passk = np.mean(estimate_pass_at_k(num_samples, num_correct, k))
                print(f"{level} Pass@{k}: {passk:.6f}")
