import sys 
import os 
import json

current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_dir)

from typing import List
from datasets import load_dataset 


def build_format_string(example):
    if "input_output" in example and len(example["input_output"]) > 0:
        input_output_dict = json.loads(example["input_output"])
    else:
        input_output_dict = {}
    
    if "fn_name" in input_output_dict:
        format_str = ("Complete the given template code." + 
                        "```python\n" +  example["starter_code"] + "\n```")
    else:
        format_str = "Read input from standard input, compute then output to standard output."
    return format_str
    

Q1 = """
Suppose you are a programming teacher and you will given high-level thoughts after reading the problem description from codeforces. 
1. Thoughts should be written in natural language and not include any form of code or pseudo-code.
2. Thoughts should not include any reference or to external resources.
3. We prefer the simple solution if the problem has multiple solutions.
4. Priorities from high to low: brute-force, greedy, dynamic programming ...


"""


def build_messages_stage1(args):
    dataset = load_dataset("json", data_files=[args.input_path], split="train")

    def map_message(example):
        task_id = example["problem_id"]
        question = example["question"]

        prompt = (Q1 + "\n" + 
                    "Let's think step by step and come up with a clever and efficient solution. ")

        example["task_id"] = task_id 
        example["query"] = [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": question + "\n" + prompt}]
        return example

    dataset = dataset.map(map_message, remove_columns=dataset.column_names)
    
    return dataset




def build_messages_stage2(args):
    dataset = load_dataset("json", data_files=[args.input_path], split="train")
    problems = load_dataset("json", data_files=[args.problem_path], split="train")
    problems_dict = {example["problem_id"]: example
                     for example in problems}
    
    def map_message(example):
        task_id = example["task_id"]
        query = example["query"]
        response = example["response"]
        format_str = build_format_string(problems_dict[task_id])
        
        # assert isinstance(query, List[dict]) and isinstance(response, dict)
        
        query.append(response)
        query += [{"role": "user", 
                   "content": "Write the solution in python3 and no explanation is required. Code should be written in a markdown codeblock. " + format_str}]
        
        example["query"] = query 
        return example
    
    dataset = dataset.map(map_message)
    return dataset