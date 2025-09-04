import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import litellm
from loguru import logger
from phoenix.evals.models.base import BaseModel, ExtraInfo, Usage
from phoenix.evals.templates import MultimodalPrompt, PromptPartContentType
from typing_extensions import override

from litellm_client.common import CONFIG

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponse


@dataclass
class OpenAIModel(BaseModel):
    """OpenAI model for text generation and completion."""

    model: str = CONFIG.eval_model.model or "gpt-3.5-turbo"
    api_key: Optional[str] = CONFIG.eval_model.api_key or None
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1
    num_retries: int = 0
    request_timeout: int = 60
    phoenix_endpoint: str = "http://localhost:6006"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._init_environment()

    @property
    def _model_name(self) -> str:
        return self.model

    def _init_environment(self) -> None:
        try:
            self._litellm = litellm

            """Setup environment variables and debugging."""
            os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = self.phoenix_endpoint
            os.environ[CONFIG.eval_model.env] = self.api_key
        except ImportError:
            logger.error("Please install litellm to use OpenAIModel.")
            return None

    async def _async_generate_with_extra(
        self, prompt: Union[str, MultimodalPrompt], **kwargs: Dict[str, Any]
    ) -> Tuple[str, ExtraInfo]:
        if isinstance(prompt, str):
            prompt = MultimodalPrompt.from_string(prompt)

        return self._generate_with_extra(prompt, **kwargs)

    def _extract_text(self, response: "ModelResponse") -> str:
        from litellm.types.utils import Choices

        if (
            response.choices
            and (choice := response.choices[0])
            and isinstance(choice, Choices)
            and choice.message.content
        ):
            return str(choice.message.content)
        return ""

    def _extract_usage(self, response: "ModelResponse") -> Optional[Usage]:
        from litellm.types.utils import Usage as ResponseUsage

        if isinstance(response_usage := response.get("usage"), ResponseUsage):  # type: ignore[no-untyped-call]
            return Usage(
                prompt_tokens=response_usage.prompt_tokens,
                completion_tokens=response_usage.completion_tokens,
                total_tokens=response_usage.total_tokens,
            )
        return None

    def _parse_output(self, response: "ModelResponse") -> Tuple[str, ExtraInfo]:
        text = self._extract_text(response)
        usage = self._extract_usage(response)
        return text, ExtraInfo(usage=usage)

    @override
    def _generate_with_extra(
        self, prompt: Union[str, MultimodalPrompt], **kwargs: Dict[str, Any]
    ) -> Tuple[str, ExtraInfo]:
        if isinstance(prompt, str):
            prompt = MultimodalPrompt.from_string(prompt)

        messages = self._get_messages_from_prompt(prompt)
        response = self._litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            num_retries=self.num_retries,
            request_timeout=self.request_timeout,
            **self.model_kwargs,
        )
        return self._parse_output(response)

    def _get_messages_from_prompt(
        self, prompt: MultimodalPrompt
    ) -> List[Dict[str, str]]:
        # LiteLLM requires prompts in the format of messages
        messages = []
        for part in prompt.parts:
            if part.content_type == PromptPartContentType.TEXT:
                messages.append({"content": part.content, "role": "user"})
            else:
                logger.warning(
                    f"Unsupported content type: {part.content_type}. Only TEXT is supported."
                )
                return None
        return messages
