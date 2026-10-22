"""Request and response bodies of the OpenAI-compatible HTTP API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_serializer

# --- Request models ---


class ChatMessage(BaseModel):
    role: str
    content: str


class StreamOptions(BaseModel):
    # OpenAI documents stream_options as "only set this when you set stream: true";
    # it is ignored for non-streaming requests.
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    # Fields OpenAI accepts that the simulator does not use (temperature, top_p,
    # tools, ...) are accepted and ignored, so existing clients keep working.
    model: str
    messages: List[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    # Deprecated by OpenAI in favour of max_completion_tokens, which wins if both are set.
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    seed: Optional[int] = None


# --- Response models ---


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
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
    # Only serialized when stream_options.include_usage is set: null on every
    # chunk except the last, which carries the totals.
    usage: Optional[Usage] = None


# --- Error models ---


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
