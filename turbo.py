import os 
import time 
import openai 

import numpy as np 

from tqdm import tqdm 
from argparse import ArgumentParser
from collections import defaultdict, Counter

from datasets import load_dataset 
from utils import (APIKeyHandler, get_sha1_hash, write_jsonl, num_tokens_from_messages)
from distutils.util import strtobool

from openai.error import RateLimitError, OpenAIError

MODEL_NAME = "gpt-3.5-turbo-0301"
MODEL_MAX_TOKENS = 4096


def query_one_generation(example, api_key_handler, args):
    retry_limits = 4
    retry_counts = 0
    exponential = 1
    
    # time.sleep(5)
    task_id, query = example['task_id'], example['query']
    if 'completion_id' in example:
        completion_id = example['completion_id']
    else:
        completion_id = get_sha1_hash()
    
    query_tokens = num_tokens_from_messages(query)
    max_tokens = min(MODEL_MAX_TOKENS - query_tokens, args.max_tokens)
    
    while retry_counts < retry_limits:
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=query,
                max_tokens=max_tokens,
                top_p=args.p,
                temperature=args.t,
                stop=args.stop,
            )
            finish_reason = response["choices"][0]["finish_reason"]
            if args.ignore_fail:
                break
            else:
                if finish_reason == "stop":
                    break 
                else:
                    print(f"Retry due to finish reason {finish_reason}")
        except RateLimitError as e:
            print(f"RateLimitError occured: {e}")
            if "billing details" in str(e):
                print(f"API key has expired: {openai.api_key}")
                api_key_handler.expire_api_key(openai.api_key)
                openai.api_key = api_key_handler.get_api_key()
            else:
                exponential *= 2
                print(f"Sleep for {20} seconds")
                time.sleep(20)
        
        except OpenAIError as e:
            print(f"OpenAIError occured: {e}")
            if "RemoteDisconnected" in str(e):
                print("sleep for 20s")
                time.sleep(20)
            
        except Exception as e:
            print(e)

        retry_counts += 1
        print(f"Retried times {retry_counts}")
        time.sleep(10)
        
    if retry_counts < retry_limits:
        # completion_text = response["choices"][0]["message"]["content"]
        # completion_role = response["choices"][0]["message"]["role"]
        contents = [{"task_id": task_id,
                     "completion_id": completion_id, 
                     "query": query,
                     "response": response["choices"][0]["message"],
                     "finish_reason": response["choices"][0]["finish_reason"],
                     "prompt_tokens": response["usage"]["prompt_tokens"],
                     "completion_tokens": response["usage"]["completion_tokens"]}]
        write_jsonl(args.output_path, contents, append=True)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--api-key-path", type=str, required=True)

    parser.add_argument("--dataset", type=str, default="apps",
                        choices=["apps", "code_contests"])
    parser.add_argument("--config", type=str, default="baseline",
                        choices=["baseline", 
                                 "cot",
                                 "simple_stage1",
                                 "perfect1_stage1",
                                 "perfect2_stage1",
                                 "perfect3_stage1",
                                 "simple_stage2",
                                 "perfect1_stage2",
                                 "perfect2_stage2",
                                 "perfect3_stage2"])

    parser.add_argument("--problem-path", type=str, default=None,
                        help="Provide necessary information about problem for stage 2 generation")
    
    parser.add_argument("--n", "--num-generations", type=int, default=1)
    parser.add_argument("--p", "--top-p", type=float, default=0.95)
    parser.add_argument("--t", "--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--stop", type=str, nargs='+')
    parser.add_argument("--ignore-fail", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--ignore-expire", type=lambda x: bool(strtobool(x)), default=False)
    
    args = parser.parse_args()

    api_key_handler = APIKeyHandler(args.api_key_path, args.ignore_expire)
    openai.api_key = api_key_handler.get_api_key()
    
    if args.dataset == "apps":
        if args.config == "baseline":
            from configs.apps.baseline import build_messages
        elif args.config == "cot":
            from configs.apps.cot import build_messages
        elif args.config == "simple_stage1":
            from configs.apps.simple import build_messages_stage1 as build_messages
        elif args.config == "perfect1_stage1":
            from configs.apps.perfect1 import build_messages_stage1 as build_messages
        elif args.config == "simple_stage2":
            from configs.apps.simple import build_messages_stage2 as build_messages
        elif args.config == "perfect1_stage2":
            from configs.apps.perfect1 import build_messages_stage2 as build_messages
        elif args.config == "perfect2_stage1":
            from configs.apps.perfect2 import build_messages_stage1 as build_messages
        elif args.config == "perfect2_stage2":
            from configs.apps.perfect2 import build_messages_stage2 as build_messages
        elif args.config == "perfect3_stage1":
            from configs.apps.perfect3 import build_messages_stage1 as build_messages
        elif args.config == "perfect3_stage2":
            from configs.apps.perfect3 import build_messages_stage2 as build_messages
        
    elif args.dataset == "code_contests":
        if args.config == "baseline":
            from configs.code_contests.baseline import build_messages
        elif args.config == "perfect1_stage1":
            from configs.code_contests.perfect1 import build_messages_stage1 as build_messages
        elif args.config == "perfect1_stage2":
            from configs.code_contests.perfect1 import build_messages_stage2 as build_messages
        elif args.config == "perfect2_stage1":
            from configs.code_contests.perfect2 import build_messages_stage1 as build_messages
        elif args.config == "perfect2_stage2":
            from configs.code_contests.perfect2 import build_messages_stage2 as build_messages
        elif args.config == "perfect3_stage1":
            from configs.code_contests.perfect3 import build_messages_stage1 as build_messages
        elif args.config == "perfect3_stage2":
            from configs.code_contests.perfect3 import build_messages_stage2 as build_messages
        elif args.config == "simple_stage1":
            from configs.code_contests.simple import build_messages_stage1 as build_messages
        elif args.config == "simple_stage2":
            from configs.code_contests.simple import build_messages_stage2 as build_messages 
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # hints = get_hints_dict(args.hints_path)
    # dataset = chat_message_code(args.split, args.shard, hints)
    # dataset = chat_message_baseline(args.split, args.shard)
    # dataset = chat_message_cot(args.split, args.shard)
    
    dataset = build_messages(args)
    
    num_generations = args.n

    if os.path.exists(args.output_path):
        completed = load_dataset("json", data_files=[args.output_path], split='train')
        tasks_counter = Counter(completed['task_id'])
        tasks_counter = defaultdict(int, tasks_counter)
    else:
        try:
            base_dir = os.path.dirname(args.output_path)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
        except FileExistsError:
            pass 
        
        tasks_counter= defaultdict(int)
    
    for example in tqdm(dataset):
        for _ in range(max(0, num_generations - tasks_counter[example['task_id']])):
            query_one_generation(example, api_key_handler, args)

