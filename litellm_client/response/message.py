from typing import Literal, TypedDict, Required, Any


class Message(TypedDict):
    content: Required[str]
    role: Required[Literal["user", "system", "assistant"]]


class MessageOutput(TypedDict):
    content: Required[str]
    role: Required[Literal["assistant"]]


class ToolCall(TypedDict):
    name: Required[str]
    """The name of the function to call"""

    arguments: Required[dict[str, Any]]
    """A JSON string of the arguments to pass to the function"""


class ToolCallOutput(TypedDict):
    output: Required[dict[str, Any]]
    """A JSON string of the output of the function tool call"""


class Reasoning(TypedDict):
    content: Required[str]
    """The reasoning content"""


class ImageInput(TypedDict):
    image_url: Required[str]
    """The url of the uploaded image"""


ResponseInput = Message | ToolCallOutput | Reasoning | ImageInput
ResponseOutput = MessageOutput | ToolCall | Reasoning
