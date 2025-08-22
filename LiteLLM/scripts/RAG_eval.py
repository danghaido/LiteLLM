import os, re, json

from phoenix.evals import (
    llm_classify,
    LiteLLMModel,
    llm_generate,
)

import litellm
litellm._turn_on_debug()

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
os.environ["HUGGINGFACE_API_KEY"] = "API"

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

def output_parser(response: str, index: int):
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        return {"__error__": str(e)}

generate_questions_template = """\
Context information is below.

---------------------
{ref_text}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
3 questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."

Output the questions in JSON format with the keys question_1, question_2, question_3.
"""

# Using the query DSL
query = SpanQuery().where("span_kind == 'CHAIN'", ).select("trace_id", input="input.value", output="output.value")
df = px.Client().query_spans(query, project_name="hugging-face")

reference = SpanQuery().where("span_kind == 'TOOL'").select("trace_id", ref="prompt.context.preview")
spans_with_docs_df = px.Client().query_spans(reference, project_name="hugging-face")
print(len(spans_with_docs_df))

document_chunks_df = explode_refs(spans_with_docs_df)
# print("Exploded format:\n", document_chunks_df)

model = LiteLLMModel(
    model="huggingface/together/Qwen/Qwen2.5-7B-Instruct",
    temperature=0.0,
)

questions_df = llm_generate(
    dataframe=document_chunks_df,
    template=generate_questions_template,
    model=model,
    output_parser=output_parser,
    concurrency=20,
)

# Construct a dataframe of the questions and the document chunks
questions_with_document_chunk_df = pd.concat([questions_df, document_chunks_df], axis=1)
questions_with_document_chunk_df = questions_with_document_chunk_df.melt(
    id_vars=["context.trace_id", "ref_text"], value_name="question"
).drop("variable", axis=1)
# If the above step was interrupted, there might be questions missing. Let's run this to clean up the dataframe.
questions_with_document_chunk_df = questions_with_document_chunk_df[
    questions_with_document_chunk_df["question"].notnull()
]

print(questions_df.head(10))
print(questions_with_document_chunk_df.head(10))