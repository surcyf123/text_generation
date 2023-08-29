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
            "iambestfeed/open_llama_3b_4bit_128g",
            "TheBloke/airoboros-13B-GPTQ",
            "reeducator/bluemoonrp-13b",
            "TehVenom/Metharme-13b-4bit-GPTQ",
            "TheBloke/gpt4-x-vicuna-13B-GPTQ",
            "TheBloke/GPT4All-13B-snoozy-GPTQ",
            "TheBloke/guanaco-33B-GPTQ",
            "TheBloke/h2ogpt-oasst1-512-30B-GPTQ",
            "TheBloke/koala-13B-GPTQ-4bit-128g",
            "TheBloke/Llama-2-13B-GPTQ",
            "TheBloke/Manticore-13B-GPTQ",
            "TheBloke/Nous-Hermes-13B-GPTQ",
            "TheBloke/tulu-30B-GPTQ",
            "TheBloke/stable-vicuna-13B-GPTQ",
            "TheBloke/WizardLM-30B-Uncensored-GPTQ",
            "TheBloke/vicuna-7B-GPTQ-4bit-128g"
        ]
        for model in models: 
            save_path = os.path.join(output_dir, os.path.basename(model))
            snapshot_download(repo_id=model, local_dir=save_path)

    else:
        save_path = os.path.join(output_dir, os.path.basename(repository_id))
        snapshot_download(repo_id=repository_id, local_dir=save_path)


if __name__ == "__main__":
    main()
    