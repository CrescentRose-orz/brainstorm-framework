import re 
import os
from argparse import ArgumentParser

from datasets import load_dataset, disable_caching, disable_progress_bar
from utils import write_jsonl, stream_jsonl

disable_caching()
disable_progress_bar()


def completion2code(completion: str):
    pattern = r"```([\s\S]*?)\n([\s\S]*?)```"
    oneside_pattern = r"```([\s\S]*?)$"
    if completion.count("```") == 0:
        return completion
    elif completion.count("```") == 1:
        matched = re.findall(oneside_pattern, completion)
        return matched[0][1]
    elif completion.count("```") == 2:
        try:
            matched = re.findall(pattern, completion)
            codeblock = matched[0][1]
        except:
            codeblock = ''
        return codeblock
    else:
        # If there exists multiple codeblocks,
        # extract content of the first block.
        # Code in other blocks may be testcases ...
        try: 
            matched = re.findall(pattern, completion)
            codeblock = matched[0][1]
        except:
            codeblock = ''
        return codeblock


def postprocess(input_path: str, output_path: str):
    
    def process_example(example):
        try:
            code_block = completion2code(example['response']['content'])
        except Exception as e:
            print(e)
            print(example['response']['content'])
            assert False 
        example['solution'] = code_block
        return example
    
    dataset = load_dataset('json', data_files=input_path, split='train')
    dataset = dataset.map(process_example, remove_columns=["response"])
    flag = False
    #for col_name_list in dataset.column_names.items():
    #    for col in col_name_list:
    for col in dataset.column_names:
        if col =='query':
            flag = True
            break
    if flag:
        dataset.remove_columns(['query'])       
    dataset.to_json(output_path)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    dirname = os.path.dirname(args.output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
    if not os.path.exists(args.output):
        postprocess(args.input, args.output)
