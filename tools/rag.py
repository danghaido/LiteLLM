# tools/rag.py
from opentelemetry.trace import Status, StatusCode

from phoenix_tools.trace.tracing import tracer
from tools.retriever import retrieve_chunks

PROMPT_TEMPLATE = """You are a careful assistant. Use the context below to support your answer.

Context (numbered chunks):
{context}

Question: {query}
"""


def build_prompt(query: str, top_k: int = 5) -> str:
    with tracer.start_as_current_span("build_prompt") as span:
        span.set_attribute("openinference.span.kind", "TOOL")
        span.set_attribute("tool.name", "build_prompt")
        span.set_attribute(
            "tool.description",
            "Construct RAG prompt from retrieved chunks",
        )
        span.set_attribute("tool.parameters", PROMPT_TEMPLATE)
        span.set_attribute("input.value", query)
        try:
            chunks = retrieve_chunks(query, top_k=top_k)
            context = (
                "\n".join([f"[{i}] {text}" for i, text in enumerate(chunks, start=1)])
                or "[1] (no context)"
            )

            # log ids + preview context
            span.set_attribute("prompt.context.ids", list(range(1, len(chunks) + 1)))
            span.set_attribute("prompt.context.preview", context)

            prompt = PROMPT_TEMPLATE.format(context=context, query=query)
            span.set_attribute("output.value", prompt)
            span.set_status(Status(StatusCode.OK))
            return prompt
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            return None
