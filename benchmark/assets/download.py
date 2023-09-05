import os

import click
from huggingface_hub import snapshot_download


@click.command()
@click.option(
    "-r",
    "--repository-id",
    "repository_id",
    required=False,
    help="HuggingFace path to the model (user/repo).",
    default=None
)
@click.option(
    "-d",
    "--output_dir",
    "output_dir",
    required=True,
    help="Directory path to save downloaded models.",
)
def main(repository_id: str, output_dir: str):
    if not repository_id:
        models = [
            #"TheBloke/airoboros-13B-GPTQ",
            #"reeducator/bluemoonrp-13b",
            #"TehVenom/Metharme-13b-4bit-GPTQ",
            #"TheBloke/gpt4-x-vicuna-13B-GPTQ",
            #"TheBloke/GPT4All-13B-snoozy-GPTQ",
            #"TheBloke/guanaco-33B-GPTQ",
            #"TheBloke/h2ogpt-oasst1-512-30B-GPTQ",
            #"TheBloke/Llama-2-13B-GPTQ",
            #"TheBloke/Manticore-13B-GPTQ",
            #"TheBloke/Nous-Hermes-13B-GPTQ",
            #"TheBloke/tulu-30B-GPTQ",
            #"TheBloke/WizardLM-30B-Uncensored-GPTQ",
            "upstage/llama-30b-instruct-2048",
            "TheBloke/scarlett-33B-GPTQ",
            "TheBloke/GPlatty-30B-GPTQ",
            "TheBloke/SuperPlatty-30B-GPTQ",
            "TheBloke/vicuna-33B-GPTQ",
            "TheBloke/airochronos-33B-GPTQ",
            "TheBloke/Firefly-Llama2-13B-v1.2-GPTQ",
            "TheBloke/Vigogne-2-13B-Instruct-GPTQ",
            "TheBloke/Platypus-30B-SuperHOT-8K-GPTQ"
        ]
        for model in models: 
            save_path = os.path.join(output_dir, os.path.basename(model))
            snapshot_download(repo_id=model, local_dir=save_path)

    else:
        save_path = os.path.join(output_dir, os.path.basename(repository_id))
        snapshot_download(repo_id=repository_id, local_dir=save_path)


if __name__ == "__main__":
    main()
    