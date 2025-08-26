from opentelemetry.trace import Status, StatusCode
from Phoenix.trace.tracing import tracer

from LiteLLM.lite import LiteLLMClient
from LiteLLM.Response import ResponseInput
from tools.rag import build_prompt

if __name__ == "__main__":
    client = LiteLLMClient()

    print("RAG REPL — nhập câu hỏi, gõ quit để thoát.")
    try:
        while True:
            query = input(">>> ").strip()
            if not query:
                continue
            if query in (":q", "exit", "quit"):
                break

            with tracer.start_as_current_span("Thought") as span:
                span.set_attribute("openinference.span.kind", "CHAIN")
                span.set_attribute("input.value", query)

                try:
                    prompt = build_prompt(query, top_k=5)

                    msg = ResponseInput(prompt)
                    response = client.complete([msg])

                    out = response.transform()
                    print(f"Answer from LLM: {out}")
                    print(response.usage())

                    span.set_attribute("output.value", out)
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    print(f"[ERROR] {e}")
    except KeyboardInterrupt:
        pass

    print("Bye.")
