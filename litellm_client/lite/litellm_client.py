import os
from typing import List

from phoenix_tools.trace.tracing import tracer

from litellm import completion
from opentelemetry.trace import Status, StatusCode

from litellm_client.common import CONFIG
from litellm_client.response import ResponseInput, ResponseOutput

from tools.factory import assistant

os.environ[CONFIG.env_key] = CONFIG.api_key


class LiteLLMClient:
    def __init__(self, model_name: str = None, temperature: float = 0.7):
        self.model_name = model_name or CONFIG.model
        self.temperature = temperature or CONFIG.temperature

    def complete(self, message: List[ResponseInput], **kwargs) -> ResponseOutput:  # type: ignore
        query = ""
        if message and isinstance(message[0], dict):
            query = message[0].get("content", "")

        with tracer.start_as_current_span("query") as span:
            span.set_attribute("openinference.span.kind", "llm")
            span.set_attribute("input.value", query)

            try:
                response = completion(
                    model=self.model_name,
                    messages=message,
                    temperature=self.temperature,
                    **kwargs,
                )
                output = response.choices[0].message.content
                span.set_attribute("output.value", output)
                span.set_status(Status(StatusCode.OK))
                return assistant(output)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return assistant(f"[ERROR] {e}")

    def batch_complete(
        self, queries: List[ResponseInput], batch_size: int = 3, **kwargs
    ) -> List[ResponseOutput]:
        results: List[ResponseOutput] = []

        for i in range(0, len(queries), batch_size):
            group = queries[i : i + batch_size]
            batched_messages = [[q] for q in group]  # <- bỏ .to_dict()
            with tracer.start_as_current_span("batch") as span:
                span.set_attribute("openinference.span.kind", "batch")
                inputs = []
                for m in batched_messages:
                    if m and isinstance(m[0], dict):
                        inputs.append(m[0].get("content", ""))
                span.set_attribute("input.value", "\n".join(inputs))

                outs_preview: List[str] = []
                try:
                    for message in batched_messages:
                        response = completion(
                            model=self.model_name,
                            messages=message,
                            temperature=self.temperature,
                            **kwargs,
                        )
                        out_text = response.choices[0].message.content or ""
                        results.append(assistant(out_text))
                        outs_preview.append(out_text)

                    span.set_attribute(
                        "output.value", "\n---\n".join(o[:400] for o in outs_preview)
                    )
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    break

        return results
