"""Domain errors raised below the API layer, which maps them to HTTP responses."""

from typing import Optional


class ContextLengthExceeded(Exception):
    """The prompt, plus any requested completion, does not fit the model's context."""

    def __init__(
        self, prompt_tokens: int, max_tokens: Optional[int], context_length: int
    ):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.context_length = context_length

        # Same wording as OpenAI's context_length_exceeded errors.
        if max_tokens is None:
            message = (
                f"This model's maximum context length is {context_length} tokens. "
                f"However, your messages resulted in {prompt_tokens} tokens. "
                "Please reduce the length of the messages."
            )
        else:
            message = (
                f"This model's maximum context length is {context_length} tokens. "
                f"However, you requested {prompt_tokens + max_tokens} tokens "
                f"({prompt_tokens} in the messages, {max_tokens} in the completion). "
                "Please reduce the length of the messages or completion."
            )
        super().__init__(message)
