from agentless.util.api_requests import num_tokens_from_messages
from agentless.util.model import make_model

from abc import ABC, abstractmethod

from agentless.repair.repair import construct_topn_file_context
from agentless.util.compress_file import get_skeleton
from agentless.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from agentless.util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    line_wrap_content,
    show_project_structure,
)

import argparse
import concurrent.futures
import json
import os

from datasets import load_dataset
from tqdm import tqdm

from agentless.fl.FL import LLMFL
from agentless.util.preprocess_data import (
    filter_none_python,
    filter_out_test_files,
    get_full_file_paths_and_classes_and_functions,
    show_project_structure,
)
from agentless.util.utils import (
    load_existing_instance_ids,
    load_json,
    load_jsonl,
    setup_logger,
)
from get_repo_structure.get_repo_structure import (
    clone_repo,
    get_project_structure_from_scratch,
)

import re

MAX_CONTEXT_LENGTH = 128000


def message_too_long(message):
    return (
        num_tokens_from_messages(message, "gpt-4o-2024-05-13") >= MAX_CONTEXT_LENGTH
    )
def split_prompt(original_prompt):
    # 提取GitHub问题描述
    github_issue = re.search(r'### GitHub Problem Description ###(.*?)### Skeleton of Relevant Files ###', original_prompt, re.DOTALL).group(1).strip()
    
    # 提取所有文件内容
    files = re.findall(r'### File: (.*?) ###\n```python(.*?)```', original_prompt, re.DOTALL)
    
    # 创建多个小prompt
    prompts = []
    for file_path, file_content in files:
        prompt_template = f"""Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{github_issue}

### Skeleton of Relevant Files ###

### File: {file_path} ###
```python{file_content}```

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
\n### Examples:\n```\nfull_path1/file1.py\nfunction: my_function_1\nclass: MyClass1\nfunction: MyClass2.my_method\n\nfull_path2/file2.py\nvariable: my_var\nfunction: MyClass3.my_method\n\nfull_path3/file3.py\nfunction: my_function_2\nfunction: my_function_3\nfunction: MyClass4.my_method_1\nclass: MyClass5\n```\n\nReturn just the locations.\n
"""
        prompts.append(prompt_template)
    
    return prompts

def process_prompt(prompt, model):
    traj = model.codegen(prompt, num_samples=1)[0]
    traj["prompt"] = prompt
    raw_output = traj["response"]
    return raw_output

def process_jsonl_object(obj):
    instance_id = obj["instance_id"]
    file_names = obj["found_files"]
    original_prompt = obj["related_loc_traj"]["prompt"]
    
    log_file = os.path.join("results_function_name_rethink", "localization_logs", f"{instance_id}.log")
    logger = setup_logger(log_file)
    model = make_model(
        model="gpt-4o-2024-05-13",
        backend="deepseek",
        logger=logger,
        max_tokens=300,
        temperature=0,
        batch_size=1,
    )

    split_prompts = split_prompt(original_prompt)
    all_raw_outputs = []
    all_model_found_locs = []

    for prompt in split_prompts:
        raw_output = process_prompt(prompt, model)
        all_raw_outputs.append(raw_output)
        model_found_locs = extract_code_blocks(raw_output)
        all_model_found_locs.extend(model_found_locs)
        print(f"Instance ID: {instance_id}")
        print(raw_output)

    model_found_locs_separated = extract_locs_for_files(all_model_found_locs, file_names)
    print(f"Instance ID: {instance_id}")
    print(all_raw_outputs)
    print(model_found_locs_separated)

    return {
        "instance_id": instance_id,
        "raw_outputs": all_raw_outputs,
        "prompts": split_prompts,
        "model_found_locs_separated": model_found_locs_separated
    }

def main():
    input_file = '/home/wsl/AgentlessOri/Agentless/results_0822_SWE-Bench_Verified/location/loc_outputs.jsonl'
    output_file = '/home/wsl/AgentlessOri/Agentless/results_0822_SWE-Bench_Verified/location/function_name_pairwise.jsonl'

    with open(input_file, 'r') as f_in, open(output_file, 'a') as f_out:
        # # 跳过前9行
        for _ in range(468):
            next(f_in, None)
        
        # 从第10行开始处理
        for line in f_in:
            obj = json.loads(line)
            result = process_jsonl_object(obj)
            json.dump(result, f_out)
            f_out.write('\n')
            f_out.flush()  # 确保立即写入文件

if __name__ == "__main__":
    main()