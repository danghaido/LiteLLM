from opentelemetry.trace import Status, StatusCode
from Phoenix.trace.tracing import tracer 

from LiteLLM.lite import LiteLLMClient
from LiteLLM.Response import ResponseInput
import phoenix as px
import pandas as pd

from tools.rag import build_prompt

def prepare_data():
    client = px.Client()
    dataset = client.get_dataset(name="ai_studio_code")

    examples = dataset.examples

    rows = []
    for ex_id, ex in examples.items():
        row = {
            "id": ex_id,
            **ex.input,
            **ex.output,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    queries_text = df["question"].tolist()

    return queries_text


if __name__ == "__main__":
    client =  LiteLLMClient()
    queries_text = prepare_data()

    for idx, query in enumerate(queries_text, 1):
        with tracer.start_as_current_span("Thought") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("input.value", query)

            try:
                prompt = build_prompt(query, top_k=3)

                msg = ResponseInput(prompt)
                response = client.complete([msg])

                out = response.transform()
                print(f"\n=== Q{idx}: {query}")
                print("Answer:", out)
                print("Usage:", response.usage())

                span.set_attribute("output.value", out[:400])
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                print(f"[ERROR @ Q{idx}] {e}")

    print("Done.")