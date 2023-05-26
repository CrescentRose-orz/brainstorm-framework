import os 
import itertools
import json
import multiprocessing
import numpy as np
from tqdm import tqdm 
from typing import Dict, List
from collections import defaultdict
from datasets import load_dataset,load_from_disk
from utils import run_test, write_jsonl,write_single_dict_2_jsonl
# from utils import run_test, write_jsonl

os.environ['HF_DATASETS_OFFLINE'] = '1'
from argparse import ArgumentParser
from datasets import (disable_caching, 
                      disable_progress_bar,
                      load_dataset)

disable_caching()


DATASET = "/home/lixy/xuejt/think-before-you-write/data/code_contests_apps_ver/"
# DATASET = "codeparrot/apps"
TIMEOUT = 10
def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
            
    return result[0]



STR_2_ID_MAPPING = {
    'Accepted':1,
    'Wrong Answer':0,
    'Compilation Error':-2
}
def evaluate_example(example: dict, code_contests_dataset,path,debug: bool=False):
    task_id = example['task_id']
    choices=['train', 'test','valid']
    found = False
    for choice in choices:
        for candidate in code_contests_dataset[choice]:
            if candidate['problem_id'] == task_id:
                sample = candidate
                found = True
                break
        if found:
            break
    # sample = code_contests_dataset[task_id]
    
    curr_res = [-2]
    try:
        curr_res = check_correctness(sample, example['solution'], timeout=TIMEOUT, debug=debug)
        if debug:
            print(f"\nSuccessful compilation of task {example['task_id']}!")
        fixed = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
        if not np.all(curr_res):
            if debug:
                print(f"Results were not True for all test cases")
    except Exception as e:
        if debug:
            print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
        assert False
    finally:
        assert isinstance(curr_res, list)
        example['result'] = json.dumps((list(map(int, curr_res))))
        example['passed'] = bool(np.all(np.array(curr_res) == 1))
    # print('writing jsonl')
    write_single_dict_2_jsonl(path,example,True)
    return example




flag = False
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test','valid'])
    parser.add_argument('--generation-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
   
    args = parser.parse_args()
    dataset = load_dataset("json", data_files=[args.generation_path], split="train")
    eval_dataset = load_from_disk(DATASET,keep_in_memory=False)
    if os.path.exists(args.result_path):
        pending_dataset = load_dataset("json", data_files=[args.result_path], split="train")
        if len(pending_dataset) == len(dataset):
            print(f'{args.generation_path} already exectuted.') 
            exit(0)
        else:
            print(f'{args.generation_path} incompletely exectuted.') 
            flag = True
            append_flag = False
            for sample in (dataset):
                for sample_done in pending_dataset:
                    if sample_done['task_id'] == sample['task_id']:
                        new_sample = sample.copy()
                        new_sample['result'] = sample_done['result']
                        new_sample['passed'] = sample_done['passed']
                        # write_jsonl(args.result_path,pending_dataset,False)
                        write_single_dict_2_jsonl(args.result_path,new_sample,append_flag)
                        append_flag = True
                        break
            
    if args.result_path is None:
        args.result_path = f'{args.generation_path}-result.jsonl'
    
    dirname = os.path.dirname(args.result_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
    

    """
    test with apps
    """
    # eval_dataset = load_dataset(DATASET, split=args.split)
    # eval_dict = {example['problem_id']: example for example in eval_dataset}
    
    
    # eval_dict = {example['problem_id']: example for choice in choices for example in eval_dataset[choice] }
    for sample in tqdm(dataset):
        skip_flag = False
        if flag:
            for sample_done in pending_dataset:
                if sample_done['task_id'] == sample['task_id']:
                    skip_flag = True
                    break
                
        if skip_flag:
            continue
        evaluate_example(sample,eval_dataset,args.result_path)
    # tmp = dataset.map(lambda x: evaluate_example(x,eval_dataset,args.result_path))
    # dataset.to_json(args.result_path)
     #print(len(eval_dict.keys()))



"""
python code_contest_execute_app_ver.py --generation-path results/code_contests_baseline/results/test_0.jsonl  --result-path ./test.jsonl --split test

"""
