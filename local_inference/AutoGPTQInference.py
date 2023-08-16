
model_name_or_path = "seonglae/opt-125m-4bit-gptq"
model_basename = "gptq_model-4bit-128g"

from flask import Flask, request, jsonify

import accelerate
import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from typing import List
from auto_gptq import AutoGPTQForCausalLM
use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          use_fast=False,
                                          trust_remote_code=True)

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def convert_stopwords_to_ids(stopwords : List[str]):
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stopwords]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria


model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                                           model_basename=model_basename,
                                           use_safetensors=False,
                                           trust_remote_code=True,
                                           device="cuda:0",
                                           use_triton=use_triton)

def generate_output(system_prompt,text,max_new_tokens,temperature,top_p,top_k,repetition_penalty,stop_tokens):
    input_ids = tokenizer(system_prompt+text, return_tensors="pt").input_ids.to("cuda")
    tokens = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p = top_p,
        top_k = top_k,
        repetition_penalty = repetition_penalty,
        stopping_criteria=convert_stopwords_to_ids(stop_tokens),
        do_sample=True)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

app = Flask(__name__)
import re

@app.route('/generate', methods=['POST'])
def generate_text():
    data=request.json
    # Get the hyperparameters and prompt from request
    text = generate_output(
        data['system_prompt'],
        data['prompt'],
        data['max_new_tokens'],
        data['temperature'],
        data['top_p'],
        data['top_k'],
        data['repetition_penalty'],
        data.get('stopwords', []))
    
    # Remove the prompts from the output
    pruned_system = re.sub('<\|.*?\|>', '', data['system_prompt'])
    pruned_user = re.sub('<\|.*?\|>', '', data['prompt'])
    text = text.replace(pruned_system, "").replace(pruned_user, "")
    
    # Remove StopToken from the Generation
    for stop in data.get('stopwords', []):
        text = text.replace(stop, "")
    
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=True, port=6666)
