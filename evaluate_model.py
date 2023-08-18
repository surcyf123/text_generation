import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv
from vllm_api_client import post_http_request, get_response
load_dotenv() 

only_prompts = pd.read_json("/dataset/only_prompts.json")[0]
N_PROMPTS_TO_PROCESS = 5

BASE_URL = "http://213.173.102.136:10400"

VERIFY_TOKEN = os.get_environ("VERIFY_TOKEN")

def get_evaluation(json_payload):
    response = requests.post(BASE_URL, json=json_payload)
    return response.json()

def get_model_response(prompt):
    start_time = time.time()
    host = "localhost"
    port = 8000
    api_url = f"http://{host}:{port}/generate"
    n = 4
    stream = False
    response = post_http_request(prompt, api_url, n, stream)
    output = get_response(response)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time, output

def generate_evaluation_json_list(prompts_df):
    evaluations_json_list = []
    for prompt in prompts_df[:N_PROMPTS_TO_PROCESS]:
        elapsed_time, text_response = get_model_response(prompt)
        json_payload = {
            "verify_token": VERIFY_TOKEN,
            "prompt": prompt,
            "responses": [text_response,]
        }
        evaluation_json = get_evaluation(json_payload)
        evaluation_json["time_elapsed_in_seconds"] = elapsed_time
        evaluations_json_list.append(evaluation_json)
    return evaluations_json_list

def generate_evaluation_df_from_json_list(evaluations_json_list):
    evaluation_data_list = []
    for evaluation_json in evaluations_json_list:
        rewards_details = evaluation_json["reward_details"]
        reciprocate_reward_model=rewards_details["reciprocate_reward_model"][0]
        relevance_filter=rewards_details["relevance_filter"][0]
        rlhf_reward_model=rewards_details["rlhf_reward_model"][0]
        evaluation_data_list.append(
            [
                reciprocate_reward_model,
                relevance_filter,
                rlhf_reward_model,
                evaluation_json["rewards"][0],
                evaluation_json["time_elapsed_in_seconds"]
            ]
        )
    evaluation_df= pd.DataFrame(evaluation_data_list, columns=["reciprocate_reward_model", "relevance_filter", "rlhf_reward_model", "rewards", "time_elapsed_in_seconds"])
    return evaluation_df


if __name__ == "main":
    prompts = pd.read_json("/dataset/only_prompts.json")
    evaluations_json_list = generate_evaluation_json_list(prompts)
    evaluation_df = generate_evaluation_df_from_json_list(evaluations_json_list)
    evaluation_df.to_csv("/dataset/evaluation.csv")

