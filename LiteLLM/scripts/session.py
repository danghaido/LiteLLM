# app.py
import uuid
import gradio as gr
from openinference.instrumentation import using_session
from opentelemetry.trace import Status, StatusCode
from Phoenix.trace.tracing import tracer

from LiteLLM.lite import LiteLLMClient
from LiteLLM.Response import ResponseInput
from tools.rag import build_prompt

client = LiteLLMClient()

def respond(user_msg, history, request: gr.Request):
    # 1) Lấy session id ổn định cho tab hiện tại
    #    Dùng session_hash của Gradio; fallback sang UUID nếu không có
    sid = getattr(request, "session_hash", None) or str(uuid.uuid4())

    with using_session(sid):
        with tracer.start_as_current_span("Thought") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("input.value", user_msg)
            try:
                prompt = build_prompt(user_msg, top_k=5)
                msg = ResponseInput(prompt)
                resp = client.complete([msg])
                out = resp.transform()

                span.set_attribute("output.value", out)
                span.set_status(Status(StatusCode.OK))
                return out
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return f"[ERROR] {e}"

demo = gr.ChatInterface(
    fn=respond,
    title="RAG Chat (Phoenix sessions)",
    description="Mỗi tab là một session; Phoenix sẽ group trace theo session.id.",
)

if __name__ == "__main__":
    demo.launch()
