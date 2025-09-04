from openinference.instrumentation.litellm import LiteLLMInstrumentor
from phoenix.otel import register

tracer_provider = register(project_name="hugging-face", auto_instrument=False)
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)