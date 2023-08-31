import logging
import os
import sys
import time
import json

import click
import pandas as pd
import requests

sys.path.append("backend")

from pipelines.gptq_pipeline import GPTQInference  # noqa
from pipelines.models import MODELS_INFO  # noqa

AVAILABLE_MODELS = list(MODELS_INFO.keys())
DEFAULT_EVALUATION_URL = "http://213.173.102.136:10400"
DEFAULT_OUTPUT_DIR = "/home/betogaona7/text_generation/benchmark/benchmark_demo"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_evaluation(evaluation_url, json_payload):
    response = requests.post(evaluation_url, json=json_payload)
    return response.json()

def get_model_response(prompts, batch_size, model):
    start_time = time.time()
    batch_responses = model.generate_batch(prompts, batch_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time/len(prompts), batch_responses

def convert_to_pd(evaluations_json_list):
    evaluation_data_list = []
    for evaluation_json in evaluations_json_list:
        rewards_details = evaluation_json["reward_details"]
        reciprocate_reward_model = rewards_details["reciprocate_reward_model"][0]
        relevance_filter = rewards_details["relevance_filter"][0]
        rlhf_reward_model = rewards_details["rlhf_reward_model"][0]
        evaluation_data_list.append(
            [
                reciprocate_reward_model,
                relevance_filter,
                rlhf_reward_model,
                evaluation_json["rewards"][0],
                evaluation_json["avg_time_elapsed_in_seconds"],
            ]
        )
    evaluation_df = pd.DataFrame(
        evaluation_data_list,
        columns=[
            "reciprocate_reward_model",
            "relevance_filter",
            "rlhf_reward_model",
            "rewards",
            "avg_time_elapsed_in_seconds",
        ],
    )
    return evaluation_df


def generate_evaluation_json_list(
    prompts_path, model_name, evaluation_url, output_dir=DEFAULT_OUTPUT_DIR, n_prompts=-1
):
    prompts_df = pd.read_json(prompts_path)[0]
    logger.info(f"Getting {prompts_df.shape[0]} prompts from {prompts_path}")

    evaluations_json_list = []
    # load model
    model_dir = MODELS_INFO[model_name]["model_dir"]
    file_name = MODELS_INFO[model_name]["model_file"]
    group_size = MODELS_INFO[model_name]["group_size"]

    model = GPTQInference(model_dir, file_name, group_size)

    if n_prompts == -1: n_prompts = prompts_df.shape[0]
    prompts = prompts_df[:n_prompts]

    batch_size = 2
    answers={}
    idx = 2

    start_time = time.time()
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        logger.info(
            f"{model_name}: Sending prompts {batch_start+1} to {batch_start + len(batch_prompts)} of {len(prompts)}" 
        )

        avg_elapsed_time, batch_responses = get_model_response(batch_prompts, batch_size, model)

        for prompt, text_response in zip(batch_prompts, batch_responses):
            logger.info(f"{model_name} Evaluating batch")
            start_index = text_response.find("ASSISTANT:") + len("ASSISTANT:")
            response = text_response[start_index:]

            json_payload = {
                "verify_token": "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n",
                "prompt": prompt,
                "responses": [
                    response,
                ],
            }
            evaluation_json = get_evaluation(evaluation_url, json_payload)
            evaluation_json["avg_time_elapsed_in_seconds"] = avg_elapsed_time
            answers[f"{idx}"] = response
            evaluations_json_list.append(evaluation_json)
            idx += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"TOTAL ELAPSED TIME: {elapsed_time}")

    # unload model
    model.unload_model()
    del model

    result_df = convert_to_pd(evaluations_json_list)
    save_path = os.path.join(output_dir, f"{model_name}_evaluation_results.csv")
    logging.info(f"Saving evaluation results in {save_path}")
    result_df.to_csv(save_path)

    save_path = os.path.join(output_dir, f"{model_name}_llm_answers.json")
    logging.info(f"Saving answers results in {save_path}")
    with open(save_path, mode="w") as file:
        json.dump(answers, file, indent=4)


@click.command()
@click.option("--prompts", "prompts_path", required=True)
@click.option(
    "--model",
    "model_name",
    required=False,
    default=None,
    help=f"Available models: {AVAILABLE_MODELS}",
)
@click.option("--output-dir", "output_dir", required=False, default=DEFAULT_OUTPUT_DIR)
def main(
    prompts_path: str,
    model_name: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:

    if not model_name:
        for model_name in AVAILABLE_MODELS:
            logger.info(f"Evaluating {model_name}")
            if not os.path.exists(os.path.join(output_dir, f"{model_name}_evaluation_results.csv")):
                generate_evaluation_json_list(
                    prompts_path, model_name, DEFAULT_EVALUATION_URL, output_dir
                )
            else:
                logger.info(f"{model_name} already evaluated...")
    else:
        logger.info(f"Evaluating {model_name}")
        generate_evaluation_json_list(
            prompts_path, model_name, DEFAULT_EVALUATION_URL, output_dir
        )



if __name__ == "__main__":
    main()
