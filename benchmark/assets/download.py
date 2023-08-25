import os

import click
from huggingface_hub import snapshot_download


@click.command()
@click.option(
    "-r",
    "--repository-id",
    "repository_id",
    required=True,
    help="HuggingFace path to the model (user/repo).",
)
@click.option(
    "-d",
    "--output_dir",
    "output_dir",
    required=True,
    help="Directory path to save downloaded models.",
)
def main(repository_id: str, output_dir: str):
    save_path = os.path.join(output_dir, os.path.basename(repository_id))
    snapshot_download(repo_id=repository_id, local_dir=save_path)


if __name__ == "__main__":
    main()
