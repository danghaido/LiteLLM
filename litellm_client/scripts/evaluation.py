import os

from loguru import logger

from phoenix_tools.eval.answer_eval import AnswerEval
from phoenix_tools.eval.rag_eval import RAGEvaluator
from tools.export_csv import export_evaluation_to_csv


def run_combined_evaluation():
    """
    Run both QA evaluation and RAG evaluation.
    """
    logger.info("=" * 60)
    logger.info("STARTING COMBINED EVALUATION")
    logger.info("=" * 60)

    if os.getenv("FROM_AUTO_QUERY") == "true":
        logger.info("Running Human Evaluation...")
        evaluator = AnswerEval("human_evaluation")
        results = evaluator.run_full_evaluation()

    # Run Evaluation
    logger.info("Running Hallucination Evaluation...")
    logger.info("-" * 40)
    evaluator = AnswerEval("hallucination")
    results = evaluator.run_full_evaluation()

    logger.info("Running Q&A Evaluation...")
    logger.info("-" * 40)
    evaluator = AnswerEval("Q&A")
    results = evaluator.run_full_evaluation()

    # Run RAG Evaluation
    logger.info("Running RAG Evaluation...")
    logger.info("-" * 40)
    rag_evaluator = RAGEvaluator()
    rag_results = rag_evaluator.run_full_rag_evaluation()

    logger.info("=" * 60)
    logger.success("BOTH EVALUATIONS COMPLETED")
    logger.info("=" * 60)

    # Export to CSV
    logger.info("Exporting results to CSV...")
    export_evaluation_to_csv()
    logger.success("Export completed.")


if __name__ == "__main__":
    results = run_combined_evaluation()
