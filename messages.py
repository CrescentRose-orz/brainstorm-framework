import json 

from datasets import (load_dataset, 
                      load_from_disk, 
                      disable_caching,
                      disable_progress_bar)
from torch.utils.data import Dataset
from typing import Dict, List, Union, Optional
from tqdm import tqdm 
from collections import defaultdict


from utils import write_jsonl, stream_jsonl


disable_caching()
disable_progress_bar()


class MessageDataset(Dataset):
    
    def __init__(self,
                 task_ids,
                 messages):
        super().__init__()
        self.messages = messages
        self.task_ids = task_ids
        

    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, index):
        return self.task_ids[index],  self.messages[index]


Q1 = """
Suppose you are a programming teacher and you will given useful and informative hints after reading the problem description from codeforces. 
1. Hints should be written in natural language and not include any form of code or pseudo-code.
2. Hints should not include any reference or to external resources.
3. We prefer the simple solution if the problem has multiple solutions.
4. Priorities from high to low: brute-force, greedy, dynamic programming ...
"""

A1 = """
Here are some guidelines to follow:
1. For brute-force problems:
- Start by understanding the problem requirements thoroughly.
- Identify the smallest input size for the problem.
- Consider how you would solve the problem if you had infinite computing power.
- Try to simplify the problem by reducing the input size or breaking it down into smaller sub-problems.
- Iterate over all possible solutions until you find the correct one.
2. For greedy problems:
- Identify the objective function that needs to be optimized.
- Consider which decisions can be made locally without affecting the global solution.
- Try to prove that the greedy solution is correct, or come up with a counter-example that shows it's incorrect.
- Implement the greedy algorithm and check whether it produces the expected result.
3. For dynamic programming problems:
- Break down the problem into smaller sub-problems.
- Identify the base case(s) that can be solved trivially.
- Determine how to combine the solutions of the sub-problems to solve the overall problem.
- Consider memoization to avoid recalculating the same sub-problems multiple times.
4. General tips:
- Read the problem statement carefully, and make sure you understand what the problem is asking.
- Consider edge cases and corner cases when designing your solution.
- Test your solution thoroughly before submitting it.
"""

    
def chat_message_hints(split, shard):
    if shard is None:
        dataset = load_from_disk(f"data/apps/{split}")
    else:
        dataset = load_from_disk(f"data/apps/{split}_{shard}")
        
    def build_messages(example):
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": Q1},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": example['question']}
        ]
        example['message'] = message
        return example
        
    dataset = dataset.map(build_messages)
    
    task_ids = dataset['problem_id']
    messages = dataset['message']
    
    return MessageDataset(task_ids, messages)


def chat_message_code(split, shard, hints):
    if shard is None:
        dataset = load_from_disk(f"data/apps/{split}")
    else:
        dataset = load_from_disk(f"data/apps/{split}_{shard}")
    
    def build_messages(example, hints):
        task_id = example['problem_id'][0]
        question = example['question'][0]
        input_output_dict = json.loads(example['input_output'][0])
        if 'fn_name' in input_output_dict:
            format_str = ("Complete the given template code." + 
                          "```python\n" + example['starter_code'][0] + "\n```")
        else:
            format_str = "Read input from standard input, compute then output to standard output."
        
        task_hints = hints[task_id]
        prompt = "Write the solution in python3 and no explanation is required. Code should be written in a markdown codeblock. " + format_str
        
        messages = []
        for h in task_hints:
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": Q1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": question},
                {"role": "assistant", "content": h},
                {"role": "user", "content": prompt}
            ]
            messages.append(message)
            
        task_ids = [task_id] * len(messages)
        return {"task_id": task_ids, "message": messages}
        
    
    dataset = dataset.map(lambda example: build_messages(example, hints),
                          batched=True, batch_size=1, remove_columns=dataset.column_names,
                          keep_in_memory=True)
    
    task_ids = dataset['task_id']
    messages = dataset['message']
    
    return MessageDataset(task_ids, messages)


def chat_message_baseline(split, shard):
    if shard is None:
        dataset = load_from_disk(f"data/apps/{split}")
    else:
        dataset = load_from_disk(f"data/apps/{split}_{shard}")
    
    def build_messages(example):
        task_id = example['problem_id']
        question = example['question']
        input_output_dict = json.loads(example['input_output'])
        if 'fn_name' in input_output_dict:
            format_str = ("Complete the given template code." + 
                          "```python\n" + example['starter_code'] + "\n```")
        else:
            format_str = "Read input from standard input, compute then output to standard output."
        
        prompt = "Write the solution in python3 and no explanation is required. Code should be written in a markdown codeblock. " + format_str
        example['task_id'] = task_id
        example['message'] = [{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": question + "\n" + prompt}]         
        return example
        
    
    dataset = dataset.map(build_messages, keep_in_memory=True)
    
    task_ids = dataset['task_id']
    messages = dataset['message']
    
    return MessageDataset(task_ids, messages)


def chat_message_cot(split, shard):
    if shard is None:
        dataset = load_from_disk(f"data/apps/{split}")
    else:
        dataset = load_from_disk(f"data/apps/{split}_{shard}")
    
    def build_messages(example):
        task_id = example['problem_id']
        question = example['question']
        input_output_dict = json.loads(example['input_output'])
        if 'fn_name' in input_output_dict:
            format_str = ("Complete the given template code." + 
                          "```python\n" + example['starter_code'] + "\n```")
        else:
            format_str = "Read input from standard input, compute then output to standard output."
        
        prompt = "Think step by step to come up with a simple and clever solution. Write the solution in python3. Code should be written in a markdown codeblock." + format_str
        example['task_id'] = task_id
        example['message'] = [{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": question + "\n" + prompt}]         
        return example
        
    
    dataset = dataset.map(build_messages, keep_in_memory=True)
    
    task_ids = dataset['task_id']
    messages = dataset['message']
    
    return MessageDataset(task_ids, messages)