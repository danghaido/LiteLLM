import os

import google.generativeai as genai
from opentelemetry import trace
from phoenix.otel import register


tracer_provider = register(project_name="my-llm=app", auto_instrument=True)

tracer = tracer_provider.get_tracer(__name__)

os.environ["GEMINI_API_KEY"] = "AIzaSyCEzOyv8a71R6CRzoKuApw1Vc8d6kA6q9s"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# 3) Gọi Gemini trong 1 span
prompt = "The cat or the dog is bigger?"
with tracer.start_as_current_span("llm.gemini.generate") as span:
    span.set_attribute("llm.vendor", "google")
    span.set_attribute("llm.model", "gemini-2.5-flash")
    span.set_attribute("llm.prompt.preview", prompt)

    resp = model.generate_content(prompt)
    text = resp.text

    span.set_attribute("llm.output.preview", text)

print(text)
