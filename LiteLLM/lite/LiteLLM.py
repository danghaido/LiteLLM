import os
from typing import List

from Phoenix.trace.tracing import tracer

from litellm import completion
from opentelemetry.trace import Status, StatusCode

from LiteLLM.common import CONFIG
from LiteLLM.Response import ResponseInput, ResponseOutput

os.environ[CONFIG.env_key] = CONFIG.api_key


class LiteLLMClient:
    def __init__(self, model_name: str = None, temperature: float = 0.7):
        self.model_name = model_name or CONFIG.model
        self.temperature = temperature or CONFIG.temperature

    def complete(self, query: List[ResponseInput], **kwargs) -> ResponseOutput:  # type: ignore
        message = [m.to_dict() for m in query]
        query = message[0].get("content")

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
                out = ResponseOutput(response).transform()
                span.set_attribute("output.value", out)
                span.set_status(Status(StatusCode.OK))
                return ResponseOutput(response)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

        return ResponseOutput(response)

    def batch_complete(
        self, queries: List[ResponseInput], batch_size: int = 3, **kwargs
    ) -> List[ResponseOutput]:
        results: List[ResponseOutput] = []

        for i in range(0, len(queries), batch_size):
            group = queries[i : i + batch_size]
            batched_messages = [[q.to_dict()] for q in group]
            with tracer.start_as_current_span("batch") as span:
                span.set_attribute("openinference.span.kind", "batch")
                inputs = []
                for m in batched_messages:
                    if m and isinstance(m[0], dict):
                        inputs.append(m[0].get("content", ""))
                span.set_attribute("input.value", "\n".join(inputs))

                outs_preview = []
                try:
                    for m in batched_messages:
                        resp = completion(
                            model=self.model_name,
                            messages=m,
                            temperature=self.temperature,
                            **kwargs,
                        )
                        out_obj = ResponseOutput(resp)
                        results.append(out_obj)
                        outs_preview.append(out_obj.transform())

                    span.set_attribute(
                        "output.value", "\n---\n".join(o[:400] for o in outs_preview)
                    )
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return results
