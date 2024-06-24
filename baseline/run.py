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
import datetime

def evaluate(method, data_filename, config_filename, model_name, device):
    # instanciate the method
    with open(config_filename, 'r') as f:
        config = json.load(f)

    method_instance = method(model_name, device, config["num_virtual_tokens"])

    with open(data_filename, 'r') as f:
        data = json.load(f)
    
    prompts = {}

    score = 0
    for target in data:
        print("target:", target)
        params = config[method.__name__]
        method_instance.fit(target, **params)
        prompts[target] = method_instance.prompt
        output = method_instance.generate(config["max_length"])
        print("output:", output)
        score += int(target in output)
    score = score / len(data)

    # delete the method
    del method_instance

    return prompts, score

def run_methods(data_filename, config_filename, methods, model_name, device):
    results = {}
    for method in methods:
        print("----- method:", method.__name__, "-----")
        start_time = time.time()
        prompts, score = evaluate(method, data_filename, config_filename, model_name, device)
        end_time = time.time()
        results[method.__name__] = {
            'prompts': prompts,
            'success_rate': score,
            'runtime': end_time - start_time
        }
    return results

def compare_results(results):
    for method_name, result_info in results.items():
        print(f"Method: {method_name}")
        print(f"Prompts: {result_info['prompts']}")
        print(f"Success rate: {result_info['success_rate']}")
        print(f"Runtime: {result_info['runtime']}s")
        print("-" * 30)

def save_results(results, data_filename, model_name, config_filename):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%m-%d_%H-%M")
    filename = f"results_{formatted_datetime}.json"
    
    with open(config_filename, 'r') as config_file:
        config_data = json.load(config_file)

    log_data = {
        "data": data_filename,
        "model_name": model_name,
        "config": config_data,
        "results": []
    }

    for method_name, result_info in results.items():
        result_entry = {
            "Method": method_name,
            "Prompts": result_info["prompts"],
            "Success rate": result_info["success_rate"],
            "Time": f"{result_info['runtime']}s"
        }
        log_data["results"].append(result_entry)

    with open(filename, 'w') as file:
        json.dump(log_data, file, indent=4)

if __name__ == "__main__":
    model_name = os.getenv("LLAMA2_PATH") # os.getenv("MISTRAL_PATH")
    print("model_name:", model_name)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    data_filename = "datasets/targets.json"
    config_filename = "config.json"

    methods = [PromptTuning, FluentPrompt, AutoPrompt, GreedyCoordinateGradient]

    results = run_methods(data_filename, config_filename, methods, model_name, device)
    compare_results(results)
    save_results(results, data_filename, model_name, config_filename)
