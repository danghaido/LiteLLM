import os, re

from phoenix.evals import (
    llm_classify,
    LiteLLMModel
)

import litellm
litellm._turn_on_debug()

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
os.environ["HUGGINGFACE_API_KEY"] = "API_KEY_HERE"  # Replace with your actual API key

import phoenix as px
import pandas as pd

from phoenix.trace.dsl import SpanQuery
from phoenix.trace import SpanEvaluations, using_project

pd.set_option("display.max_colwidth", None)

def normalize_newline(s: str) -> str:
    """Chuyển '\\n' literal thành newline thật."""
    return s.replace("\\n", "\n") if isinstance(s, str) else s

def split_ref_items(s: str):
    """
    Tách chuỗi ref dạng:
    [1] Intel focuses...
    [2] AMD Ryzen offers...
    [4] Laptop battery life...
    Thành list [(id, text), ...]
    """
    if not isinstance(s, str):
        return []
    s = normalize_newline(s)
    pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=(?:\n\[\d+\])|$)", flags=re.S)
    return pattern.findall(s)

def explode_refs(df: pd.DataFrame, ref_col: str = "ref") -> pd.DataFrame:
    """
    Nhận DataFrame có cột 'ref', trả về DataFrame chỉ gồm context.trace_id và ref_text.
    """
    tmp = df.copy()
    tmp["ref_items"] = tmp[ref_col].apply(split_ref_items)
    out = tmp.explode("ref_items", ignore_index=True)
    out[["ref_id", "ref_text"]] = pd.DataFrame(out["ref_items"].tolist(), index=out.index)
    out["ref_id"] = out["ref_id"].astype(int)
    return out[["context.trace_id", "ref_text"]]

QA_TEMPLATE = """You are given a question and an answer.
Decide if the answer correctly and fully answers the question.

[Question]
{input}

[Answer]
{output}

First, briefly explain your reasoning (1–3 sentences).
Then, on the **last line**, output ONLY ONE of the following labels (all lowercase):
correct
incorrect
"""

# Using the query DSL
query = SpanQuery().where("span_kind == 'CHAIN'", ).select("trace_id", input="input.value", output="output.value")
df = px.Client().query_spans(query, project_name="hugging-face")
print(len(df))

reference = SpanQuery().where("span_kind == 'TOOL'").select("trace_id", ref="prompt.context.preview")
spans_with_docs_df = px.Client().query_spans(reference, project_name="hugging-face")
print(len(spans_with_docs_df))

document_chunks_df = explode_refs(spans_with_docs_df)
print("Exploded format:\n", document_chunks_df)

df_merged = pd.merge(
    df,
    spans_with_docs_df,
    on="context.trace_id",
    how="inner",
    suffixes=("_chain", "_tool")
)
print(df_merged)


model = LiteLLMModel(
    model="huggingface/together/Qwen/Qwen2.5-7B-Instruct",
    temperature=0.0,
)

with using_project("hugging-face"):
    evals_df = llm_classify(
        data=df,
        template=QA_TEMPLATE,
        model=model,
        rails=["correct", "incorrect"],
        provide_explanation=True,
    )

print(evals_df.head())

# px.Client().log_evaluations(
#     SpanEvaluations(eval_name="valid", dataframe=evals_df),
# )