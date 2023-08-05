import argparse

import torch
import torch.nn as nn
import quant

from utils import find_layers
import transformers
from transformers import AutoTokenizer


def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model





    
import os
import argparse
from typing import Dict, List
from flask import Flask, request, jsonify
import json
import random
import traceback
import sys
import time
import requests
import re
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define the Flask app
app = Flask(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')

    # parser.add_argument('--text', type=str, help='input text')

    # parser.add_argument('--min_length', type=int, default=10, help='The minimum length of the sequence to be generated.')

    # parser.add_argument('--max_length', type=int, default=50, help='The maximum length of the sequence to be generated.')

    # parser.add_argument('--top_p',
    #                     type=float,
    #                     default=0.95,
    #                     help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.')

    # parser.add_argument('--temperature', type=float, default=0.8, help='The value used to module the next token probabilities.')

    # parser.add_argument('--device', type=int, default=-1, help='The device used to load the model when using safetensors. Default device is "cpu" or specify, 0,1,2,3,... for GPU device.')

    # fused mlp is sometimes not working with safetensors, no_fused_mlp is used to set fused_mlp to False, default is true
    parser.add_argument('--fused_mlp', action='store_true')
    parser.add_argument('--no_fused_mlp', dest='fused_mlp', action='store_false')
    parser.add_argument("--auth_token", default="", help="Authentication token")
    parser.add_argument("--port", default=8008, type=int, help="Authentication token")
    parser.set_defaults(fused_mlp=True)

    args = parser.parse_args()
    return args


@app.route("/", methods=["POST"])
def chat():
    # Check authentication token
    request_data = request.get_json()
    auth_token = request_data.get("verify_token")
    if auth_token != args.auth_token:
        return jsonify({"error": "Invalid authentication token"}), 401

    # Get messages from the request
    
    messages = request_data.get("messages", [])
    n = request_data.get('n', 4)

    # Call the forward function and get the response
    try:
        response = miner.forward(messages, num_replies = n)
    except:
        traceback.print_exc(file=sys.stderr)
        response = ["That is an excellent question..."]

    # Return the response
    return jsonify({"response": response})



class ModelMiner():

    def __init__( self, args, device="cuda", max_length=200, temperature=0.7, do_sample=True ):
        super( ModelMiner, self ).__init__()
        
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.do_sample = do_sample
        
        self.system_prompt=""

        if type(args.load) is not str:
            args.load = args.load.as_posix()

        if args.load:
            self.model = load_quant(args.model, args.load, args.wbits, args.groupsize, fused_mlp=args.fused_mlp)
        else:
            self.model = get_llama(args.model)
            self.model.eval()

        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        print("model loaded")


    def _process_history(self, history: List[str]) -> str:
        processed_history = ''

        if self.system_prompt:
            processed_history += self.system_prompt

        for message in history:
            if message['role'] == 'system':
                processed_history += '' + message['content'].strip() + ' '

            if message['role'] == 'assistant':
                processed_history += 'ASSISTANT:' + message['content'].strip() + '</s>'
            if message['role'] == 'user':
                processed_history += 'USER: ' + message['content'].strip() + ' '
        return processed_history

    def forward(self, messages, num_replies=4):

        history = self._process_history(messages)
        prompt = history + "ASSISTANT:"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # output = self.model.generate(
        #     input_ids,
        #     max_length=input_ids.shape[1] + self.max_length,
        #     temperature=self.temperature,
        #     do_sample=self.do_sample,
        #     pad_token_id=self.tokenizer.eos_token_id,
        # )

        # generation = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + self.max_length,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=num_replies,  # Set the number of desired replies
                # penalty_alpha=0.6, top_k=4,
            )
        
        generations = []
        for sequence in output:
            generation = self.tokenizer.decode(sequence[input_ids.shape[1]:], skip_special_tokens=True)
            generations.append(generation)

        # Logging input and generation if debugging is active
        print("Message: " + str(messages),flush=True)
        print("Generation: " + str(generation),flush=True)

        return generations


if __name__ == "__main__":
    args = parse_arguments()
    miner = ModelMiner(args)
    app.run(host="0.0.0.0", port=args.port, threaded=False)