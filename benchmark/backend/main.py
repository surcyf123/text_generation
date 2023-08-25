from backend.api.llms import api_router
from fastapi import FastAPI

app = FastAPI(title="LLM Benchmark")
app.include_router(api_router, prefix="/app")
