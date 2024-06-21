import time
from baseline.prompt_tuning import PromptTuning
from baseline.autoprompt import AutoPrompt
from baseline.gcg import GreedyCoordinateGradient
from baseline.fluent_prompt import FluentPrompt
import os
from dotenv import load_dotenv 
load_dotenv()
import torch
import json

def evaluate(method, data_filename, model_name, device, num_virtual_tokens, max_length=50):
    # instanciate the method
    method_instance = method(model_name, device, num_virtual_tokens)

    with open(data_filename, 'r') as file:
        data = json.load(file)
    score = 0
    for target in data:
        print("target:", target)
        method_instance.fit(target)
        output = method_instance.generate(max_length)
        score += int(target in output)
    score = score / len(data)

    # delete the method
    del method_instance

    return score

def run_methods(data_filename, methods, model_name, device, num_virtual_tokens):
    results = {}
    for method in methods:
        start_time = time.time()
        result = evaluate(method, data_filename, model_name, device, num_virtual_tokens)
        end_time = time.time()
        results[method.__class__.__name__] = {
            'result': result,
            'time': end_time - start_time
        }
    return results

def compare_results(results):
    for method_name, result_info in results.items():
        print(f"Method: {method_name}")
        print(f"Result: {result_info['result']}")
        print(f"Time: {result_info['time']}s")
        print("-" * 30)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_results(results):
    for method_name, result_info in results.items():
        logger.info(f"Method: {method_name}")
        logger.info(f"Result: {result_info['result']}")
        logger.info(f"Time: {result_info['time']}s")
        logger.info("-" * 30)

if __name__ == "__main__":
    model_name = os.getenv("LLAMA2_PATH") # os.getenv("MISTRAL_PATH")
    print("model_name:", model_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_filename = "targets.json"
    num_virtual_tokens = 10

    methods = [PromptTuning, FluentPrompt, AutoPrompt, GreedyCoordinateGradient]

    results = run_methods(data_filename, methods, model_name, device, num_virtual_tokens)
    compare_results(results)
    log_results(results)
