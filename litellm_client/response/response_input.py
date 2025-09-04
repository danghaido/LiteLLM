from typing import Literal

Role = Literal["system", "user", "assistant"]


class ResponseInput:
    def __init__(self, message: str, role="user") -> None:
        self.message = message
        self.role = role

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.message}
