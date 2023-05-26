import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_dir)

import json
from datasets import load_dataset 


def build_messages(args):
    dataset = load_dataset("json", data_files=[args.input_path], split='train')
    
    def map_message(example):
        task_id = example['name']
        question = example['description']
        
        prompt = "Write the solution in python3 and no explanation is required. Code should be written in a markdown codeblock. "
        example['task_id'] = task_id 
        example['message'] = [{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": question + "\n" + prompt}]
        return example
    
    dataset = dataset.map(map_message, remove_columns=dataset.column_names)
    return dataset 