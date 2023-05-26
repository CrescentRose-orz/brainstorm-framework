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
        task_id = example['problem_id']
        question = example['question']

        if 'input_output' in example and len(example['input_output']) > 0:
            input_output_dict = json.loads(example['input_output'])
        else:
            input_output_dict = {}
        
        if 'fn_name' in input_output_dict:
            format_str = ("Complete the given template code." + 
                          "```python\n" + example['starter_code'] + "\n```")
        else:
            format_str = "Read input from standard input, compute then output to standard output."
        
        prompt = "Write the solution in python3 and no explanation is required. Code should be written in a markdown codeblock. " + format_str
        example['task_id'] = task_id
        example['query'] = [{"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": question + "\n" + prompt}]         
        return example
    
    dataset = dataset.map(map_message, remove_columns=dataset.column_names)
    
    return dataset