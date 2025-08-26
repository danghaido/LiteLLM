import phoenix.evals
import os
from phoenix.evals import (
    HUMAN_VS_AI_PROMPT_TEMPLATE,
    HUMAN_VS_AI_PROMPT_RAILS_MAP,
    OpenAIModel,
    llm_classify,
)
from phoenix.experiments import run_experiment
from phoenix.experiments.types import EvaluationResult
import phoenix as px
import pandas as pd

from LiteLLM.Response import ResponseInput
from tools.rag import build_prompt

from LiteLLM.common import CONFIG


def prepare_data():
    client = px.Client()
    dataset = client.get_dataset(name="ai_studio_code")

    examples = dataset.examples  # Đây là dict: { id: Example(...) }

    rows = []
    for ex_id, ex in examples.items():
        row = {
            "id": ex_id,
            **ex.input,
            **ex.output,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


HUMAN_VS_AI_PROMPT_PROMPT_TEMPLATE = """
You are comparing a human ground truth answer from an expert to an answer from
an AI model. Your goal is to determine if the AI answer correctly matches, in
substance, the human answer.
    [BEGIN DATA]
    ************
    [Question]: {question}
    ************
    [Human Ground Truth Answer]: {expected_answer}
    ************
    [AI Answer]: {ai_generated_answer}
    ************
    [END DATA]

Compare the AI answer to the human ground truth answer. First, write out in a
step by step manner an EXPLANATION to show how to determine if the AI Answer is
'relevant' or 'irrelevant'. Avoid simply stating the correct answer at the
outset. You are then going to respond with a LABEL (a single word evaluation).
If the AI correctly answers the question as compared to the human answer, then
the AI answer LABEL is "correct". If the AI answer is longer but contains the
main idea of the Human answer please answer LABEL "correct". If the AI answer
diverges or does not contain the main idea of the human answer, please answer
LABEL "incorrect".

Example response:
************
EXPLANATION: An explanation of your reasoning for why the AI answer is "correct"
or "incorrect" LABEL: "correct" or "incorrect"
************

EXPLANATION:
"""


def human_eval(df):
    # Run the LLM classification
    eval_df = llm_classify(
        dataframe=df,
        template=HUMAN_VS_AI_PROMPT_PROMPT_TEMPLATE,
        model=OpenAIModel(model="gpt-4o-mini"),
        rails=["factual", "hallucinated"],
        provide_explanation=True,
    )

    # Map the eval df to EvaluationResult
    label = eval_df["label"][0]
    score = 1 if label == "correct" else 0
    explanation = eval_df["explanation"][0]

    # Return the evaluation result
    return EvaluationResult(label=label, score=score, explanation=explanation)


def my_task(df):
    prompt = build_prompt(query, top_k=3)
    msg = ResponseInput(prompt)
    response = client.complete([msg])

    out = response.transform()
    print(f"Answer from LLM: {out}")
    print(response.usage())


client = px.Client()
dataset = client.get_dataset(name="ai_studio_code")
df = prepare_data()

exp = run_experiment(dataset, human_eval)
