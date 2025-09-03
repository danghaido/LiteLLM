import os
import re
from typing import Optional, List, Dict, Any

import litellm
import phoenix as px
import pandas as pd

from phoenix.evals import llm_classify, LiteLLMModel
from phoenix.trace.dsl import SpanQuery
from phoenix.trace import SpanEvaluations, using_project

from LiteLLM.common import CONFIG
from Phoenix.eval.Evaluation_Metrics import (
    EvaluationMetrics,
    EvaluationTemplates,
    EvaluationRails,
    parse_evaluation_metric,
)


class AnswerEval:
    """
    A class for evaluating LLM responses using Phoenix tracing and evaluation framework.
    """

    def __init__(
        self,
        evaluation_metric: str = "Q&A",
        custom_template: Optional[str] = None,
        custom_rails: Optional[List[str]] = None,
        project_name: str = "hugging-face",
        model_name: str = "huggingface/together/Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        phoenix_endpoint: str = "http://localhost:6006",
        dataset: str = "ai_studio_code",
    ):
        self.evaluation_metric = parse_evaluation_metric(evaluation_metric)

        # Set template and rails
        if self.evaluation_metric == EvaluationMetrics.CUSTOM:
            if not custom_template:
                raise ValueError(
                    "custom_template is required when using 'custom' evaluation metric"
                )
            self.template = custom_template
            self.rails = custom_rails or EvaluationRails.get_rails(
                EvaluationMetrics.CUSTOM
            )
        else:
            self.template = EvaluationTemplates.get_template(self.evaluation_metric)
            self.rails = custom_rails or EvaluationRails.get_rails(
                self.evaluation_metric
            )

        self.project_name = CONFIG.project or project_name
        self.model_name = CONFIG.eval_model.model or model_name
        self.temperature = CONFIG.eval_model.temperature or temperature
        self.dataset = CONFIG.dataset or dataset

        # Setup environment
        self._setup_environment(phoenix_endpoint)

        # Initialize model
        self.model = LiteLLMModel(
            model=self.model_name,
            temperature=self.temperature,
        )

        # Initialize Phoenix client
        self.client = px.Client()

    def _setup_environment(self, phoenix_endpoint: str) -> None:
        """Setup environment variables and debugging."""
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint
        os.environ[CONFIG.eval_model.env] = CONFIG.eval_model.api_key
        litellm._turn_on_debug()

    @staticmethod
    def normalize_newline(s: str) -> str:
        """Convert '\\n' literal to actual newline."""
        return s.replace("\\n", "\n") if isinstance(s, str) else s

    @staticmethod
    def split_ref_items(s: str) -> List[tuple]:
        """
        Split reference string in format:
        [1] Intel focuses...
        [2] AMD Ryzen offers...
        [4] Laptop battery life...
        Into list [(id, text), ...]
        """
        if not isinstance(s, str):
            return []
        s = AnswerEval.normalize_newline(s)
        pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=(?:\n\[\d+\])|$)", flags=re.S)
        return pattern.findall(s)

    def explode_refs(self, df: pd.DataFrame, ref_col: str = "ref") -> pd.DataFrame:
        tmp = df[["context.trace_id", ref_col]].copy()

        tmp[ref_col] = tmp[ref_col].astype("string").fillna("")

        tmp["ref_items"] = tmp[ref_col].apply(self.split_ref_items)

        tmp = tmp.loc[
            tmp["ref_items"].map(lambda x: isinstance(x, list) and len(x) > 0)
        ].copy()
        if tmp.empty:
            return pd.DataFrame(columns=["context.trace_id", "ref_id", "reference"])

        out = tmp.explode("ref_items", ignore_index=True)

        pairs = out["ref_items"].apply(pd.Series)
        pairs.columns = ["ref_id", "reference"]

        out = pd.concat(
            [
                out[["context.trace_id"]].reset_index(drop=True),
                pairs.reset_index(drop=True),
            ],
            axis=1,
        )

        out["ref_id"] = pd.to_numeric(out["ref_id"], errors="coerce").astype("Int64")

        return out[["context.trace_id", "reference"]]

    def get_evaluation_data(self) -> pd.DataFrame:
        """
        Fetch spans that need evaluation from Phoenix.

        Returns:
            DataFrame with spans to evaluate
        """
        query = (
            SpanQuery()
            .where("span_kind == 'CHAIN'")
            .select(
                "context.span_id",
                "context.trace_id",
                input="input.value",
                output="output.value",
            )
        )

        df = self.client.query_spans(query, project_name=self.project_name)
        return df

    def get_reference_data(self) -> pd.DataFrame:
        """
        Fetch reference/tool spans from Phoenix.

        Returns:
            DataFrame with reference data
        """
        reference_query = (
            SpanQuery()
            .where("span_kind == 'TOOL'")
            .select("context.span_id", "context.trace_id", ref="prompt.context.preview")
        )

        spans_with_docs_df = self.client.query_spans(
            reference_query, project_name=self.project_name
        )
        return spans_with_docs_df

    def get_expected_answer_data(self) -> pd.DataFrame:
        """
        Fetch expected answers from Phoenix.

        Returns:
            DataFrame with expected answers
        """
        dataset = self.client.get_dataset(name=self.dataset)
        examples = dataset.examples
        rows = []
        for ex_id, ex in examples.items():
            row = {
                "id": ex_id,
                **ex.input,
                **ex.output,
            }
            rows.append(row)

        expected_answers_df = pd.DataFrame(rows).rename(columns={"question": "input"})
        return expected_answers_df

    def prepare_evaluation_dataset(self) -> pd.DataFrame:
        """
        Prepare the complete dataset for evaluation.

        Returns:
            DataFrame with merged evaluation and reference data
        """
        # Get evaluation data
        eval_df = self.get_evaluation_data()

        # Get reference data
        reference_df = self.get_reference_data()

        # Get expected answer data
        expected_answers_df = self.get_expected_answer_data()

        # Merge evaluation and reference data
        merged_df = pd.merge(
            eval_df,
            reference_df.rename(columns={"ref": "reference"})[
                ["context.trace_id", "reference"]
            ],
            on="context.trace_id",
            how="left",
            suffixes=("_chain", "_tool"),
        )
        merged_df.index = eval_df.index
        merged_df.index.name = eval_df.index.name

        final_df = pd.merge(
            merged_df,
            expected_answers_df,
            on="input",
            how="left",
            suffixes=("_left", "_right"),
        )
        final_df.index = merged_df.index
        final_df.index.name = merged_df.index.name
        return final_df

    def run_evaluation(
        self,
        data: pd.DataFrame,
        provide_explanation: bool = True,
    ) -> pd.DataFrame:
        """
        Run LLM-based evaluation on the provided data.

        Args:
            data: DataFrame with input/output columns
            template: Evaluation template (uses default if None)
            rails: List of allowed labels (uses default if None)
            provide_explanation: Whether to include explanations

        Returns:
            DataFrame with evaluation results
        """

        with using_project(self.project_name):
            evals_df = llm_classify(
                data=data,
                template=self.template,
                model=self.model,
                rails=self.rails,
                provide_explanation=provide_explanation,
            )
        return evals_df

    def log_evaluations(
        self, evals_df: pd.DataFrame, eval_name: Optional[str] = None
    ) -> None:
        """
        Log evaluation results to Phoenix.

        Args:
            evals_df: DataFrame with evaluation results
            eval_name: Name for the evaluation
        """
        eval_name = eval_name or self.evaluation_metric.value
        evals_df["metadata"] = self.model_name
        self.client.log_evaluations(
            SpanEvaluations(eval_name=eval_name, dataframe=evals_df),
        )

    def run_full_evaluation(self) -> pd.DataFrame:
        """
        Run the complete evaluation pipeline.

        Returns:
            DataFrame with evaluation results
        """

        # Prepare dataset
        merged_df = self.prepare_evaluation_dataset()

        # Run evaluation
        evals_df = self.run_evaluation(merged_df)
        # Log to Phoenix
        self.log_evaluations(evals_df)

        return evals_df


def main():
    """Main function to run evaluation."""
    evaluator = AnswerEval("human_evaluation")
    results = evaluator.run_full_evaluation()
    return results


if __name__ == "__main__":
    main()
