import logging

from backend.api.dependencies import GPTQInference, llm_dependency
from backend.api.schemas import Request, Response
from backend.pipelines.models import MODELS_INFO
from fastapi import Depends, Header, HTTPException
from fastapi.routing import APIRouter

AVAILABLE_MODELS = list(MODELS_INFO.keys())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


default_model_name = "airoboros-13B-GPTQ"
default_llm_dependency = llm_dependency(default_model_name)

async def custom_llm_dependency(model_name: str = Header(None)):
    global default_model_name, default_llm_dependency
    if model_name is None:
        return default_llm_dependency

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model not available. Available models: {AVAILABLE_MODELS}",
        )
    
    if model_name != default_model_name:
        default_model_name = model_name
        default_llm_dependency.unload_model()
        default_llm_dependency = llm_dependency(model_name)

    return default_llm_dependency


@router.post("/generate")
async def ask_llm(
    request: Request,
    model_name: str = Header(None),
    llm_dependency: GPTQInference = Depends(custom_llm_dependency),
) -> Response:
    """
    Generate an answer using a language model.
    
    This route generates an answer using a language model based on the provided query.
    
    Parameters:
      - **model_name**: The name of the language model to use. (Header)

    Avilable models: [
        airoboros-13B-GPTQ,
        bluemoonrp-13b,
        Metharme-13b-4bit-GPTQ,
        gpt4-x-vicuna-13B-GPTQ,
        GPT4All-13B-snoozy-GPTQ,
        koala-13B-GPTQ-4bit-128g,
        Llama-2-13B-GPTQ,
        Manticore-13B-GPTQ,
        Nous-Hermes-13B-GPTQ,
        stable-vicuna-13B-GPTQ,
        guanaco-33B-GPTQ,
        h2ogpt-oasst1-512-30B-GPTQ,
        tulu-30B-GPTQ,
        WizardLM-30B-Uncensored-GPTQ
    ]
    """
    global default_model_name
    query = request.query
    logger.info(f"{default_model_name} User query: {query}")
    answer = llm_dependency.generate(query)
    start_index = answer.find("ASSISTANT:") + len("ASSISTANT:")
    response = answer[start_index:]

    return {"answer": response}
