import asyncio
import aiohttp
import csv
import logging
import json
import requests
import time

DEFAULT_SCORE_SERVER = "http://213.173.102.136:10400"
DEFAULT_LLM_SERVER = "http://35.204.237.75:8000/generate" 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_evaluation(evaluation_url, json_payload):
    response = requests.post(evaluation_url, json=json_payload)
    return response.json()

async def process_requests_async(prompts, model_name, concurrency_limit=10):
    logger.info("processing prompts")
    async def process_prompt(prompt):
        logger.info(f"Processing prompt:  {prompt[:100]}")
        async with aiohttp.ClientSession() as session:
            async with session.post(DEFAULT_LLM_SERVER, headers=headers, json={"query":prompt}, timeout=21600) as response:
                response_json = await response.json()
                logger.info(f"RESPONSE {response_json}")
                return response_json
            
    headers = {
        'accept': 'application/json',
        'model-name': model_name,
        'Content-Type': 'application/json'
    }

    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = [process_prompt(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

async def main():
    with open("/home/betogaona7/text_generation/benchmark/assets/dataset/only_prompts.json", "r") as f:
        prompts = json.load(f)
    model_name = "airoboros-13B-GPTQ"
    
    loop = asyncio.get_event_loop()

    start_time = time.time()
    results = await process_requests_async(prompts[:10], model_name, concurrency_limit=10)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"TOTAL ELAPSED TIME: {elapsed_time}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
