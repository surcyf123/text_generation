# Benchmark

Evaluate GPTQ LLM's models.

## Installation

Create and enter to a conda env:

```bash
conda create -n benchmark python=3.11 -y
conda activate benchmark
```

Install requirements

```bash
pip install -r requirements.txt
```

Install AutoGPTQ

Follow the instructions to install [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) package.

## Usage

Run the command

```bash
uvicorn backend.main:app
```

## Scripts

Evaluate a model:

```bash
python scripts/evaluation_demo.py --prompts="PATH/TO/PROMPTS/JSON/FILE" --output-dir="PATH/TO/SAVE/EVALUATION/FILE" --model-name="<MODEL>"
```

To see the list of available models add the flag `--help`
