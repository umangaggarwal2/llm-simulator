from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from models import ErrorDetail, ErrorResponse


class OpenAIError(Exception):
    """Raised from route code to return a specific OpenAI-style error."""

    def __init__(
        self,
        status: int,
        message: str,
        type: str = "invalid_request_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.param = param
        self.code = code


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(OpenAIError, openai_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    # Starlette's HTTPException also covers routing errors (404, 405) and
    # FastAPI's HTTPException, which subclasses it.
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


async def openai_exception_handler(_: Request, exc: OpenAIError):
    return openai_error(exc.status, exc.message, exc.type, exc.param, exc.code)


async def validation_exception_handler(_: Request, exc: RequestValidationError):
    # OpenAI reports one problem at a time, so surface only the first error.
    error = exc.errors()[0]
    param = _format_param(error["loc"])

    if error["type"] == "json_invalid":
        message = (
            "We could not parse the JSON body of your request. "
            "Please make sure the body is valid JSON."
        )
        param = None
    elif error["type"] == "missing" and param is None:
        message = "Missing request body."
    elif error["type"] == "missing":
        message = f"Missing required parameter: '{param}'."
    else:
        message = f"Invalid value for '{param}': {error['msg']}."

    return openai_error(400, message, param=param)


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        message = f"Invalid URL ({request.method} {request.url.path})"
    elif exc.status_code == 405:
        message = f"Method not allowed ({request.method} {request.url.path})"
    else:
        message = str(exc.detail)

    response = openai_error(exc.status_code, message)
    # Keep headers such as Allow on 405 responses.
    if exc.headers:
        response.headers.update(exc.headers)
    return response


async def unhandled_exception_handler(_: Request, exc: Exception):
    # Starlette re-raises the exception after this returns, so the server
    # still logs the full traceback; the client only sees a generic message.
    return openai_error(
        500,
        "The server had an error while processing your request. Sorry about that!",
        type="server_error",
    )


def openai_error(
    status: int,
    message: str,
    type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(message=message, type=type, param=param, code=code)
    )
    return JSONResponse(status_code=status, content=body.model_dump())


def _format_param(loc) -> Optional[str]:
    # ("body", "messages", 0, "content") -> "messages.[0].content", matching OpenAI.
    parts = [f"[{p}]" if isinstance(p, int) else str(p) for p in loc[1:]]
    return ".".join(parts) or None
