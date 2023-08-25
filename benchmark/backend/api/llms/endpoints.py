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


async def custom_llm_dependency(model_name: str = Header(None)):
    if model_name is None:
        raise HTTPException(
            status_code=400, detail="No model_name provided in the request header"
        )

    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model not available. Available models: {AVAILABLE_MODELS}",
        )

    return llm_dependency(model_name)


@router.post("/generate")
async def ask_llm(
    request: Request,
    model_name: str = Header(None),
    llm_dependency: GPTQInference = Depends(custom_llm_dependency),
) -> Response:
    """Endpoint to ask an llm"""
    query = request.query
    logger.info(f"{model_name} User query: {query}")
    answer = llm_dependency.generate(query)
    llm_dependency.unload_model()

    # answer = "works"
    return {"answer": answer}
