from typing import Callable

from backend.pipelines.gptq_pipeline import GPTQInference
from backend.pipelines.models import MODELS_INFO


def llm_dependency(model_name: str) -> Callable[[], GPTQInference]:
    model_dir = MODELS_INFO[model_name]["model_dir"]
    model_name = MODELS_INFO[model_name]["model_file"]
    return GPTQInference(model_dir=model_dir, model_name=model_name)
