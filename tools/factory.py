# factories.py
from typing import cast
from litellm_client.response.message import Message, MessageOutput


def user(content: str) -> Message:
    return {"role": "user", "content": content}


def system(content: str) -> Message:
    return {"role": "system", "content": content}


def assistant(content: str) -> MessageOutput:
    return cast(MessageOutput, {"role": "assistant", "content": content})
