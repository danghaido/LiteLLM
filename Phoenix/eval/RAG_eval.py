import os
import re
import json
from typing import Optional, List, Dict, Any

import litellm
import phoenix as px
import pandas as pd
from loguru import logger

from phoenix.evals import LiteLLMModel, RelevanceEvaluator, run_evals
from phoenix.trace.dsl import SpanQuery
from phoenix.trace import SpanEvaluations, using_project, DocumentEvaluations

from LiteLLM.common import CONFIG


class RAGEvaluator:
    """
    A class for evaluating RAG (Retrieval-Augmented Generation) document relevance
    using Phoenix tracing and evaluation framework.
    """

    def __init__(
        self,
        project_name: str = "hugging-face",
        model_name: str = "huggingface/together/Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        phoenix_endpoint: str = "http://localhost:6006",
        concurrency: int = 20,
    ):
        """
        Initialize the RAG evaluation system.

        Args:
            project_name: Phoenix project name
            model_name: LLM model to use for evaluation
            temperature: Temperature for model inference
            phoenix_endpoint: Phoenix collector endpoint
            concurrency: Number of concurrent evaluation requests
        """
        self.project_name = CONFIG.project or project_name
        self.model_name = CONFIG.eval_model.model or model_name
        self.temperature = CONFIG.eval_model.temperature or temperature
        self.concurrency = CONFIG.eval_model.concurrency or concurrency

        # Setup environment
        self._setup_environment(phoenix_endpoint)

        # Initialize model
        self.model = LiteLLMModel(
            model=self.model_name,
            temperature=self.temperature,
        )

        # Initialize Phoenix client
        self.client = px.Client()

        # Configure pandas display
        pd.set_option("display.max_colwidth", None)

    def _setup_environment(self, phoenix_endpoint: str) -> None:
        """Setup environment variables and debugging."""
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint
        os.environ[CONFIG.eval_model.env] = CONFIG.eval_model.api_key
        litellm._turn_on_debug()

    @staticmethod
    def robust_output_parser(response: str, index: int) -> Dict[str, Any]:
        """
        Parse model output robustly, handling various response formats.

        Args:
            response: Raw model response
            index: Index for logging purposes

        Returns:
            Parsed response dictionary
        """
        s = (response or "").strip()

        # Log first few samples for debugging
        if index < 3:
            logger.info(f"\nRAW[{index}]:\n{repr(s)}\n")

        if not s:
            return {
                "__error__": "empty",
                "question_1": None,
                "question_2": None,
                "question_3": None,
            }

        # Remove code fences ```json ... ```
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
            s = re.sub(r"\s*```$", "", s, flags=re.S)

        # Extract first JSON block
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(s[start : end + 1])
                return {
                    "question_1": obj.get("question_1"),
                    "question_2": obj.get("question_2"),
                    "question_3": obj.get("question_3"),
                }
            except json.JSONDecodeError:
                pass

        # Fallback: treat as plain text
        return {"question_1": s, "question_2": None, "question_3": None}

    def get_retrieval_data(self) -> pd.DataFrame:
        """
        Fetch retrieval spans that need relevance evaluation from Phoenix.

        Returns:
            DataFrame with retrieval spans to evaluate
        """
        query = (
            SpanQuery()
            .where("span_kind == 'RETRIEVER' and evals['relevance'].label is None")
            .select(
                "context.span_id",
                "context.trace_id",
                "input.value",
                "retrieval.documents",
            )
        )

        df = self.client.query_spans(query, project_name=self.project_name)
        df = df.reset_index().rename(columns={"index": "context.span_id"})
        logger.info(f"Found {len(df)} retrieval spans to evaluate")
        return df

    def prepare_documents_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare documents DataFrame for relevance evaluation.

        Args:
            df: Raw retrieval spans DataFrame

        Returns:
            Prepared documents DataFrame
        """
        # Explode documents
        df_exploded = df.explode("retrieval.documents", ignore_index=True)

        # Normalize nested document dictionaries
        docs = pd.json_normalize(df_exploded["retrieval.documents"])

        # Combine with trace information
        retrieved_documents_df = pd.concat(
            [df_exploded.drop(columns=["retrieval.documents"]), docs], axis=1
        )

        # Rename columns for evaluation
        retrieved_documents_df = retrieved_documents_df.rename(
            columns={"document.content": "reference", "input.value": "input"}
        )
        return retrieved_documents_df

    def run_relevance_evaluation(self, documents_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run relevance evaluation on the documents.

        Args:
            documents_df: DataFrame with documents to evaluate

        Returns:
            DataFrame with relevance evaluation results
        """
        relevance_evaluator = RelevanceEvaluator(self.model)

        logger.info("Running relevance evaluation...")
        retrieved_documents_relevance_df = run_evals(
            evaluators=[relevance_evaluator],
            dataframe=documents_df,
            provide_explanation=True,
            concurrency=self.concurrency,
        )[0]

        return retrieved_documents_relevance_df

    def prepare_final_dataframe(
        self, documents_df: pd.DataFrame, relevance_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare the final DataFrame with proper indexing for Phoenix logging.

        Args:
            documents_df: Original documents DataFrame
            relevance_df: Relevance evaluation results

        Returns:
            Final DataFrame ready for logging
        """
        # Combine documents with relevance evaluations
        documents_with_relevance_df = pd.concat(
            [documents_df, relevance_df.add_prefix("eval_")], axis=1
        )

        final_df = documents_with_relevance_df.copy()

        # Convert score to numeric, handle errors as NaN
        final_df["__score__"] = pd.to_numeric(
            final_df["document.score"], errors="coerce"
        )

        # Sort by span_id and score (descending); NaN values go to end of group
        final_df = final_df.sort_values(
            ["context.span_id", "__score__"], ascending=[True, False]
        )

        # Add document position within each span
        final_df["document_position"] = final_df.groupby("context.span_id").cumcount()

        # Create MultiIndex as required by Phoenix
        final_df = final_df.set_index(["context.span_id", "document_position"]).drop(
            columns="__score__"
        )
        final_df = final_df.rename(
            columns={
                "eval_label": "label",
                "eval_score": "score",
                "eval_explanation": "explanation",
            }
        )
        return final_df

    def log_evaluations(
        self, final_df: pd.DataFrame, eval_name: str = "relevance"
    ) -> None:
        self.client.log_evaluations(
            DocumentEvaluations(eval_name=eval_name, dataframe=final_df),
        )
        logger.info(f"Logged {len(final_df)} document evaluations as '{eval_name}'")

    def run_full_rag_evaluation(self) -> pd.DataFrame:
        """
        Run the complete RAG evaluation pipeline.

        Returns:
            DataFrame with evaluation results
        """

        # Get retrieval data
        retrieval_df = self.get_retrieval_data()

        if retrieval_df.empty:
            logger.info("No retrieval data to evaluate")
            return pd.DataFrame()

        # Prepare documents
        documents_df = self.prepare_documents_dataframe(retrieval_df)

        # Run relevance evaluation
        relevance_df = self.run_relevance_evaluation(documents_df)

        # Prepare final DataFrame
        final_df = self.prepare_final_dataframe(documents_df, relevance_df)

        # Log to Phoenix
        self.log_evaluations(final_df)

        return final_df


def main():
    """Main function to run RAG evaluation."""
    rag_evaluator = RAGEvaluator()
    results = rag_evaluator.run_full_rag_evaluation()
    return results


if __name__ == "__main__":
    main()
