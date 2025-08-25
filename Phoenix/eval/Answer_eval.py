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

class AnswerEval:
    """
    A class for evaluating LLM responses using Phoenix tracing and evaluation framework.
    """
    
    QA_TEMPLATE = """You are given a question and an answer.
        Decide if the answer correctly and fully answers the question.

        [Question]
        {input}

        [Answer]
        {output}

        First, briefly explain your reasoning (1–3 sentences).
        Then, on the **last line**, output must ONLY BE ONE of the following labels (all lowercase):
        correct
        incorrect
    """

    def __init__(
        self, 
        project_name: str = "hugging-face",
        model_name: str = "huggingface/together/Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        phoenix_endpoint: str = "http://localhost:6006"
    ):
        """
        Initialize the evaluation system.
        
        Args:
            project_name: Phoenix project name
            model_name: LLM model to use for evaluation
            temperature: Temperature for model inference
            phoenix_endpoint: Phoenix collector endpoint
        """
        self.project_name = CONFIG.project
        self.model_name = CONFIG.eval_model.model or model_name
        self.temperature = CONFIG.eval_model.temperature or temperature
        
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

        tmp = tmp.loc[tmp["ref_items"].map(lambda x: isinstance(x, list) and len(x) > 0)].copy()
        if tmp.empty:
            return pd.DataFrame(columns=["context.trace_id", "ref_id", "ref_text"])

        out = tmp.explode("ref_items", ignore_index=True)

        pairs = out["ref_items"].apply(pd.Series)
        pairs.columns = ["ref_id", "ref_text"]

        out = pd.concat(
            [out[["context.trace_id"]].reset_index(drop=True), pairs.reset_index(drop=True)],
            axis=1,
        )

        out["ref_id"] = pd.to_numeric(out["ref_id"], errors="coerce").astype("Int64")

        return out[["context.trace_id", "ref_text"]]

    def get_evaluation_data(self) -> pd.DataFrame:
        """
        Fetch spans that need evaluation from Phoenix.
        
        Returns:
            DataFrame with spans to evaluate
        """
        query = SpanQuery().where(
            "span_kind == 'CHAIN'"
        ).select("trace_id", input="input.value", output="output.value")
        
        df = self.client.query_spans(query, project_name=self.project_name)
        print(f"Found {len(df)} spans to evaluate")
        return df

    def get_reference_data(self) -> pd.DataFrame:
        """
        Fetch reference/tool spans from Phoenix.
        
        Returns:
            DataFrame with reference data
        """
        reference_query = SpanQuery().where(
            "span_kind == 'TOOL'"
        ).select("trace_id", ref="prompt.context.preview")
        
        spans_with_docs_df = self.client.query_spans(
            reference_query, 
            project_name=self.project_name
        )
        print(f"Found {len(spans_with_docs_df)} reference spans")
        return spans_with_docs_df

    def prepare_evaluation_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare the complete dataset for evaluation.
        
        Returns:
            Tuple of (evaluation_df, merged_df)
        """
        # Get evaluation data
        eval_df = self.get_evaluation_data()
        
        # Get reference data
        reference_df = self.get_reference_data()
        
        # Process reference data
        document_chunks_df = self.explode_refs(reference_df)
        print("Exploded reference format:\n", document_chunks_df.head())
        
        # Merge evaluation and reference data
        merged_df = pd.merge(
            eval_df,
            reference_df,
            on="context.trace_id",
            how="inner",
            suffixes=("_chain", "_tool")
        )
        print(f"Merged dataset shape: {merged_df.shape}")
        
        return eval_df, merged_df

    def run_evaluation(
        self, 
        data: pd.DataFrame, 
        template: Optional[str] = None,
        rails: Optional[List[str]] = None,
        provide_explanation: bool = True
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
        template = template or self.QA_TEMPLATE
        rails = rails or ["correct", "incorrect"]
        
        with using_project(self.project_name):
            evals_df = llm_classify(
                data=data,
                template=template,
                model=self.model,
                rails=rails,
                provide_explanation=provide_explanation,
            )
        
        return evals_df

    def log_evaluations(self, evals_df: pd.DataFrame, eval_name: str = "valid") -> None:
        """
        Log evaluation results to Phoenix.
        
        Args:
            evals_df: DataFrame with evaluation results
            eval_name: Name for the evaluation
        """
        self.client.log_evaluations(
            SpanEvaluations(eval_name=eval_name, dataframe=evals_df),
        )
        print(f"Logged {len(evals_df)} evaluations as '{eval_name}'")

    def run_full_evaluation(self) -> pd.DataFrame:
        """
        Run the complete evaluation pipeline.
        
        Returns:
            DataFrame with evaluation results
        """
        print("Starting evaluation pipeline...")
        
        # Prepare dataset
        eval_df, merged_df = self.prepare_evaluation_dataset()
        
        if eval_df.empty:
            print("No data to evaluate")
            return pd.DataFrame()
        
        # Run evaluation
        print("Running LLM evaluation...")
        evals_df = self.run_evaluation(eval_df)
        
        # Log to Phoenix
        self.log_evaluations(evals_df)
        
        return evals_df


def main():
    """Main function to run evaluation."""
    evaluator = AnswerEval()
    results = evaluator.run_full_evaluation()
    return results


if __name__ == "__main__":
    main()