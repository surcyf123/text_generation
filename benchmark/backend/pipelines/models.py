import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets")

MODELS_INFO = {
    "airoboros-13B-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/airoboros-13B-GPTQ"),
        "group_size": 128
    },
    "bluemoonrp-13b": {
       "model_file": "bluemoonrp-13b-4k-epoch6-4bit-128g",
       "model_dir": os.path.join(DATA_DIR, "quantized_models/bluemoonrp-13b"),
       "group_size": 128
    },
    "gpt4-x-vicuna-13B-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/gpt4-x-vicuna-13B-GPTQ"),
        "group_size": 128
    },
    "GPT4All-13B-snoozy-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/GPT4All-13B-snoozy-GPTQ"),
    },
    "koala-13B-GPTQ-4bit-128g": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/koala-13B-GPTQ-4bit-128g"),
        "group_size": 128
    },
    "Llama-2-13B-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/Llama-2-13B-GPTQ"),
        "group_size": 128
    },
    "Manticore-13B-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/Manticore-13B-GPTQ"),
        "group_size": 128
    },
    "Metharme-13b-4bit-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/Metharme-13b-4bit-GPTQ"),
        "group_size": 128
    },
    "Nous-Hermes-13B-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/Nous-Hermes-13B-GPTQ"),
        "group_size": 128
    },
    "stable-vicuna-13B-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/stable-vicuna-13B-GPTQ"),
        "group_size": 128
    },
    "vicuna-7B-GPTQ-4bit-128g": {
       "model_file": "model",
       "model_dir": os.path.join(DATA_DIR, "quantized_models/vicuna-7B-GPTQ-4bit-128g"),
       "group_size": 128
    },
    "open_llama_3b_4bit_128g": {
       "model_file": "model",
       "model_dir": os.path.join(DATA_DIR, "quantized_models/open_llama_3b_4bit_128g"),
       "group_size": 128
    },
    "guanaco-33B-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/guanaco-33B-GPTQ"),
        "group_size": -1
    },
    "h2ogpt-oasst1-512-30B-GPTQ": {
       "model_file": "model",
       "model_dir": os.path.join(DATA_DIR, "quantized_models/h2ogpt-oasst1-512-30B-GPTQ/"),
       "group_size": -1
    },
    "tulu-30B-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/tulu-30B-GPTQ"),
        "group_size": -1
    },
    "WizardLM-30B-Uncensored-GPTQ": {
        "model_file": "model",
        "model_dir": os.path.join(DATA_DIR, "quantized_models/WizardLM-30B-Uncensored-GPTQ"),
        "group_size": -1
    }
}



