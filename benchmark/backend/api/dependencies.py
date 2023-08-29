from typing import Callable

from backend.pipelines.gptq_pipeline import GPTQInference
from backend.pipelines.models import MODELS_INFO


def llm_dependency(model_name: str) -> Callable[[], GPTQInference]:
    model_dir = MODELS_INFO[model_name]["model_dir"]
    model_name = MODELS_INFO[model_name]["model_file"]
    group_size = MODELS_INFO[model_name]["group_size"]
    return GPTQInference(model_dir=model_dir, model_name=model_name, group_size=group_size)
