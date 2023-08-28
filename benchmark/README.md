# Backend

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
