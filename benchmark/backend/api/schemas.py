from pydantic import BaseModel


class Request(BaseModel):
    query: str


class Response(BaseModel):
    answer: str
