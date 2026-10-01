from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_serializer

# TODO: load from the YAML config once it exists.
MODEL_NAME = "qwen-mock"


# --- Request models ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: Optional[int] = Field(default=None, ge=1)


# --- Response models ---


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "llm-simulator"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class Delta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    index: int
    delta: Delta
    finish_reason: Optional[str] = None

    @field_serializer("delta")
    def _serialize_delta(self, delta: Delta):
        # OpenAI omits unset delta fields, so the final chunk's delta is just {}.
        return delta.model_dump(exclude_none=True)


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChunkChoice]


# --- Error models ---


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
