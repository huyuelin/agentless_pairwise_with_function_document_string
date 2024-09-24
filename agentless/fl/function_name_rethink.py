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

MAX_CONTEXT_LENGTH = 128000


def message_too_long(message):
    return (
        num_tokens_from_messages(message, "gpt-4o-2024-05-13") >= MAX_CONTEXT_LENGTH
    )

function_name_rethink_prompt = """
Please review the following previously generated Found Functions for your reference.

### Found Functions ###
{found_related_locs}

###

{function_name_localization_original_prompt}



"""

def process_jsonl_object(obj):
    instance_id = obj["instance_id"]
    file_names = obj["found_files"]
    function_name_localization_original_prompt = obj["related_loc_traj"]["prompt"]
    found_related_locs = obj["found_related_locs"]

    template = function_name_rethink_prompt

    flattened_found_related_locs = [item for sublist in found_related_locs for item in sublist if item]
    message = template.format(
        function_name_localization_original_prompt=function_name_localization_original_prompt,
        found_related_locs="\n".join(flattened_found_related_locs),)


    log_file = os.path.join(
        "results_function_name_rethink", "localization_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)
    model = make_model(
        model="gpt-4o-2024-05-13",
        backend="deepseek",
        logger=logger,
        max_tokens=300,
        temperature=0,
        batch_size=1,
    )

    traj = model.codegen(message, num_samples=1)[0]
    traj["prompt"] = message
    raw_output = traj["response"]
    print(f"Instance ID: {instance_id}")
    print(raw_output)

    model_found_locs = extract_code_blocks(raw_output)
    model_found_locs_separated = extract_locs_for_files(
        model_found_locs, file_names
    )

    return {
        "instance_id": instance_id,
        "raw_output": raw_output,
        "message_rethink": message,
        "model_found_locs_separated": model_found_locs_separated
    }

def main():
    input_file = '/home/wsl/AgentlessOri/Agentless/results_0822_SWE-Bench_Verified/location/loc_outputs.jsonl'
    output_file = '/home/wsl/AgentlessOri/Agentless/results_0822_SWE-Bench_Verified/location/function_name_rethink.jsonl'

    with open(input_file, 'r') as f_in, open(output_file, 'a') as f_out:
        
        for _ in range(350):
            next(f_in, None)
            
        for line in f_in:
            obj = json.loads(line)
            result = process_jsonl_object(obj)
            json.dump(result, f_out)
            f_out.write('\n')
            f_out.flush()  # 确保立即写入文件

if __name__ == "__main__":
    main()