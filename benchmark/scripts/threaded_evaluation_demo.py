import os
import sys
import time

import click
import pandas as pd
import requests
import json
import queue
import threading
import concurrent.futures
import logging 
import csv

sys.path.append("backend")

from pipelines.gptq_pipeline import GPTQInference  # noqa
from pipelines.models import MODELS_INFO  # noqa


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_SCORING_SERVER_URLS = ["http://213.173.102.136:10400", "http://213.173.102.136:10401", "http://213.173.102.136:10402"]
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "benchmark_demo")


# globals 
scoring_servers = queue.Queue()


def write_to_csv(csv_filename, data):
    logger.info(f"DEBUG: {data}")
    fieldnames = [
        'rewards',
        'reward_details'
    ]
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(data)

def score_answers(session, data, result, url):
    logger.info("Scoring answer...")
    req_data = {
        'verify_token': 'SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n',
        'prompt': data['prompt'],
        'responses': result
    }
    try: 
        response = session.post(url, json=req_data)
        response.raise_for_status()
        response_json = response.json()
        logger.info(f"Scores received: {response_json['rewards']}")
        logger.info(f"Reward details: {response_json['reward_details']}")
        return response_json
    except Exception as e:
        logger.error(f"Error scoring response: {e}")
        return {'rewards': None, 'reward_details': {}}


def request_llm_answer(prompt, model):
    start_time = time.time()
    full_response = model.generate(prompt)
    # extract assistant only response
    start_index = full_response.find("### Assistant:") + len("### Assistant:")
    response = full_response[start_index:]
    end_time = time.time()
    elapsed_time = end_time - start_time
    return response

def generate_and_score(executor, session, data, writer, scoring_url, prompt_number, model_name, model, file_name):
    logger.info("Starting generate and score function...")

    generated_score = []
    all_scores = {}
    result = None

    def generate_and_score_model():
        nonlocal generated_score, result, all_scores
        task = executor.submit(request_llm_answer, data["prompt"], model)
        result = task.result()
        
        logger.info("Scoring generated answer...")
        response_data = score_answers(session, data, result, scoring_url)
        generated_score = response_data['rewards']
        all_scores[f"{model_name}"] = response_data["reward_details"]
        write_to_csv(file_name, response_data)

    try:
        model_thread = threading.Thread(target=generate_and_score_model)
        model_thread.start()
        model_thread.join(9.4)
    finally: 
        scoring_servers.put(scoring_url)
        logger.info("Finished generate and score function...")


def evaluate(
    prompts_file:str,
    model_name: str,
    scoring_server_urls:list=DEFAULT_SCORING_SERVER_URLS,
    output_dir:str=DEFAULT_OUTPUT_DIR,

):
    with open(prompts_file, "r") as f:
        prompts = json.load(f)

    for url in scoring_server_urls:
        scoring_servers.put(url)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_name}_evaluation_scores.csv")

    # load model
    model_dir = MODELS_INFO[model_name]["model_dir"]
    file_name = MODELS_INFO[model_name]["model_file"]
    group_size = MODELS_INFO[model_name]["group_size"]

    model = GPTQInference(model_dir, file_name, group_size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor, requests.Session() as session:
        threads = []
        for idx, prompt in enumerate(prompts[0:5]):
            data = {"prompt": prompt}
            logger.info(f"Generating answer for prompt {idx+1}")
            try:
                scoring_url = scoring_servers.get(block=True, timeout=10)
                t = threading.Thread(target=generate_and_score, args=(executor, session, data, write_to_csv, scoring_url, idx+1, model_name, model, save_path))
                t.start()
                threads.append(t)
            except queue.Empty:
                logger.error("No scoring server available after waiting 10 seconds")
                continue 

    # unload model
    model.unload_model()
    del model




if __name__ == "__main__":
    evaluate("/home/betogaona7/text_generation/benchmark/assets/dataset/only_prompts.json", "airoboros-13B-GPTQ")
    

