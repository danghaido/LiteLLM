import os
import json

import litellm
import pandas as pd
import phoenix as px

from LiteLLM.common import CONFIG
from phoenix.trace.dsl import SpanQuery
from phoenix.client import Client

# Configuration constants
PHOENIX_COLLECTOR_ENDPOINT = "http://localhost:6006"
PROJECT_NAME = "hugging-face"

# Enable debug mode for LiteLLM
litellm._turn_on_debug()

# Set environment variables
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = PHOENIX_COLLECTOR_ENDPOINT
os.environ["HUGGINGFACE_API_KEY"] = CONFIG.api_key

# Initialize Phoenix clients
phoenix_client = px.Client()
span_client = Client()

# Define query for spans with CHAIN kind
def export_evaluation_to_csv():
    span_query = (
        SpanQuery()
        .where("span_kind == 'CHAIN'")
        .select(
            "context.span_id",
            "context.trace_id",
            input="input.value",
            output="output.value",
        )
    )

    # Query spans and prepare dataframe
    spans_df = phoenix_client.query_spans(span_query, project_name=PROJECT_NAME)
    spans_df = spans_df.reset_index().rename(columns={"index": "context.span_id"})

    # Get additional spans dataframe
    all_spans_df = span_client.spans.get_spans_dataframe(
        query=SpanQuery(),
        project_identifier=PROJECT_NAME
    )

    # Get span annotations dataframe
    annotations_df = span_client.spans.get_span_annotations_dataframe(
        spans_dataframe=all_spans_df,
        project_identifier=PROJECT_NAME
    )

    # Process annotations dataframe
    annotations_df = (
        annotations_df
        .reset_index()  # Convert index to column
        .rename(columns={
            "span_id": "context.span_id",
            "result.label": "eval",
            "result.explanation": "explanation",
            "annotation_name": "metrics",
        })
    )

    # Merge spans and annotations dataframes
    final_df = pd.merge(spans_df, annotations_df, on="context.span_id", how="left")

    # Define columns to keep
    selected_columns = [
        "context.span_id",
        "input", 
        "output",
        "metrics",
        "eval",
        "explanation"
    ]

    # Select only the required columns
    final_df = final_df[selected_columns]
    final_df.to_csv("eval_result.csv", index=False, encoding="utf-8-sig")