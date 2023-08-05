import concurrent.futures
import requests
import openai
import json
import logging
import threading
import csv
import time
import queue
from collections import defaultdict, deque
import os
import random
from flask import Flask, request, jsonify


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

min_question_length = 500
num_responses = 3
start_prompt = 333
temperatures = [0.33]
openai.api_key = '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'
scoring_server_urls = ["http://64.247.206.99:43183", "http://64.247.206.99:43184", "http://64.247.206.99:43185", "http://64.247.206.99:43186", "http://64.247.206.99:43187"]
system_prompts = [
    '''
    Demonstrate a potential experiment while utilizing and enumerating the scientific method clearly and explain every step for a potential theory of the following context.
    ''',
    '''
    Provide enumerated steps to provide clear instructions on how to perform a task based on the following context.
    ''',
]
strategies = ['start', 'middle']  # define the strategies you want to use

with open('/home/fsuser/mining/wandb/cleaned_ranked_answers_per_question.json', 'r') as f:
    dataset = json.load(f)

with open('top_30.json', 'r') as f:
    predefined_answers = json.load(f)

# Create a dictionary mapping prompts to their corresponding answers in the dataset
dataset_answers = {}
for entry in dataset:
    prompt = entry['prompt']
    answers = entry['sorted_answers']
    dataset_answers[prompt] = answers

n = 50

# Create a queue for scoring servers
scoring_servers = queue.Queue()
for url in scoring_server_urls:
    scoring_servers.put(url)

fieldnames = ['Question Number', 'System Prompt Index', 'Temperature', 'Strategy', 'Final Score', 'RLHF Reward Model Score', 'Reciprocate Reward Model Score', 'Diversity Reward Model Score', 'Relevance Filter Score', 'Rank', 'Best Generated Answer', 'Source Identifier', 'OpenAI Parameters', 'Was Random', 'Short Question']

# Create a lock for the CSV file
csv_filename = 'output.csv'
csv_lock = threading.Lock()
def write_to_csv(data):
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(data)


def select_prompt_portion(prompt, strategy, portion_size=50):  # default portion size
    if strategy == "start":
        return prompt[:portion_size]
    elif strategy == "middle":
        start = max(0, len(prompt) // 2 - portion_size // 2)
        return prompt[start:start+portion_size]
    elif strategy == "end":
        return prompt[-portion_size:]
    elif strategy == "full":
        return prompt
    else:
        raise ValueError("Unknown strategy")

def request_to_openai(model_name, messages, temperature, max_tokens, top_p, system_prompt_index, strategy, frequency_penalty=1, presence_penalty=0.1):
    logging.info(f"Calling OpenAI API with system prompt index {system_prompt_index}, temperature {temperature}, and strategy {strategy}")
    start_time = time.time()
    portion = select_prompt_portion(data['prompt'], strategy)

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=1,
            )
            duration = time.time() - start_time
            logging.info(f"OpenAI API request took {duration} seconds")
            answers = [choice["message"]["content"].strip() for choice in response["choices"]]
            return (answers, system_prompt_index, temperature, strategy)
        except Exception as e:
            logging.error("Error making request to OpenAI API. Retrying in 5 seconds.", e)
            time.sleep(5)

def score_answers(session, data, all_answers, url):
    logging.info("Scoring answers")
    req_data = {
        'verify_token': 'SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n',
        'prompt': data['prompt'],
        'responses': all_answers
    }

    try:
        response = session.post(url, json=req_data)
        response.raise_for_status()  # This will raise an error if the request fails
        response_json = response.json()
        logging.info(f"Scores received: {response_json['rewards']}")
        logging.info(f"Reward details: {response_json['reward_details']}")
        return response_json
    except Exception as e:
        logging.error(f"Error scoring answers: {e}")
        return {'rewards': None, 'reward_details': {}}

def generate_and_score(executor, session, data, temperatures, system_prompt_indices, strategies, writer, scoring_url, question_number, min_question_length):
    
    logging.info("Starting generate_and_score function.")
    predefined_scores = []
    generated_scores = []
    all_scores = {}
    delay_time = 9.4
    
    predefined_finished_event = threading.Event()
    openai_finished_event = threading.Event()

    def score_predefined():
        nonlocal predefined_scores, all_scores
        try:
            logging.info("Scoring predefined answers...")
            predefined_response_data = score_answers(session, data, predefined_answers, scoring_url)
            predefined_scores = predefined_response_data['rewards']
            all_scores["predefined"] = predefined_response_data['reward_details']
        finally:
            predefined_finished_event.set()

    results = []
    def generate_and_score_openai():
        nonlocal generated_scores, results, all_scores

        tasks = []
        for temperature in temperatures:
            for system_prompt_index in system_prompt_indices:
                for strategy in strategies:
                    logging.info(f"Generating answers for temp: {temperature}, prompt index: {system_prompt_index}, strategy: {strategy}...")
                    messages = [
                        {"role": "system", "content": system_prompts[system_prompt_index]},
                        {"role": "user", "content": data['prompt']}
                    ]
                    task = executor.submit(request_to_openai, 'gpt-3.5-turbo', messages, temperature, 50, 0.8, system_prompt_index, strategy)
                    tasks.append(task)

        results = [future.result() for future in tasks]
        all_answers = [result[0][0] for result in results]

        logging.info("Scoring generated answers...")
        response_data = score_answers(session, data, all_answers, scoring_url)
        generated_scores = response_data['rewards']
        all_scores["openai"] = response_data['reward_details']

    is_short_question = len(data['prompt']) < min_question_length
    if is_short_question:
        # If short, only get one generated answer without scoring
        logging.info("Question is short. Getting one OpenAI answer without scoring...")
        messages = [
            {"role": "system", "content": system_prompts[0]},
            {"role": "user", "content": data['prompt']}
        ]
        result = request_to_openai('gpt-3.5-turbo', messages, temperatures[0], 50, 0.8, system_prompt_indices[0], strategies[0])
        return result[0][0] # return the generated response immediately
    else:
        try:
            predefined_thread = threading.Thread(target=score_predefined)
            openai_thread = threading.Thread(target=generate_and_score_openai)
            predefined_thread.start()
            openai_thread.start()

            predefined_thread.join(delay_time)
            openai_thread.join(delay_time)

            # Determine the details of the best answer
            best_data = {}
            if predefined_finished_event.is_set() or openai_finished_event.is_set():
                combined_scores = []
                if predefined_finished_event.is_set():
                    combined_scores += [(score, "predefined", idx) for idx, score in enumerate(predefined_scores)]
                if openai_finished_event.is_set():
                    combined_scores += [(score, "openai", idx) for idx, score in enumerate(generated_scores)]
                
                best_score_tuple = max(combined_scores, key=lambda x: x[0])
                best_score, source, index = best_score_tuple
                is_short_question = len(data['prompt']) < min_question_length

                best_data = {
                    'Question Number': question_number,
                    'Short Question': is_short_question,
                    'Final Score': best_score,
                    'Was Random': False,
                    'OpenAI Parameters': f"{results[index][2]}, {results[index][3]}, {results[index][4]}" if source == "openai" else "",
                    # 'Best Generated Answer': predefined_answers[index] if source == "predefined" else results[index][0][0],
                    'Reciprocate Reward Model Score': all_scores[source]['reciprocate_reward_model'][index],
                    'Diversity Reward Model Score': all_scores[source]['diversity_reward_model'][index],
                    'Relevance Filter Score': all_scores[source]['relevance_filter'][index],
                    'Rank': index+1,
                    'Source Identifier': f"openai-{results[index][2]}-{results[index][3]}-{results[index][4]}" if source == "openai" else f"predefined-{index}"
                }
            else:
                # If none were scored within the time limit, return a random predefined answer
                best_data = {
                    'Question Number': question_number,
                    'Was Random': True,
                    'OpenAI Parameters': "",
                    # 'Best Generated Answer': random.choice(predefined_answers),
                    'Source Identifier': f"predefined-random"
                }

            #write to csv
            csv_lock.acquire()
            try:
                write_to_csv(best_data)
            except Exception as e:
                print(f"Exception while writing to CSV: {e}")
            finally:
                csv_lock.release()
            logging.info("Finished generate_and_score function.")

        finally:
            # This will ensure the scoring server URL is put back to the queue even if an exception occurs
            scoring_servers.put(scoring_url)
            logging.info("Finished generate_and_score function.")

with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor, requests.Session() as session:
    threads = []
    for i, data in enumerate(dataset[start_prompt-1:start_prompt-1+n]):
        logging.info(f"Generating Answers...")
        for temperature in temperatures:
            # Block until a scoring server is available
            try:
                scoring_url = scoring_servers.get(block=True, timeout=10)  # wait 10 seconds
                t = threading.Thread(target=generate_and_score, args=(executor, session, data, [temperature], [0, 1], strategies, write_to_csv, scoring_url, i + 1, min_question_length))
                t.start()
                threads.append(t)
            except queue.Empty:
                logging.error("No scoring server available after waiting for 10 seconds.")
                continue

    # Wait for all of the threads to finish
    for t in threads:
        t.join()

