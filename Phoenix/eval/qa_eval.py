# eval_hf_min.py
from __future__ import annotations
import re
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Template theo Phoenix "Q&A on Retrieved Data"
_QA_TEMPLATE = """You are given a question, an answer and reference text. You must determine whether the
given answer correctly answers the question based on the reference text. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {question}
    ************
    [Reference]: {context}
    ************
    [Answer]: {answer}
    [END DATA]
Your response must be a single word, either "PASS" or "FAIL",
and should not contain any text or characters aside from that word.
"PASS" means that the question is correctly and fully answered by the answer.
"FAIL" means that the question is not correctly or only partially answered by the answer.
"""

_LABEL_RE = re.compile(r"\b(PASS|FAIL)\b", re.I)


class Eval:
    """
    Đánh giá: answer có trả lời đúng câu hỏi dựa trên context hay không.
    - __init__(model_id=...): khởi tạo HF model (transformers pipeline).
    - eval(question, answer, context) -> {"label": "correct"/"incorrect"/None, "raw": str}
    """

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 64,
        device: Optional[str] = None,
    ) -> None:
        

        self._tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self._mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device is None else device,
        )
        self._pipe = pipeline(
            task="text-generation",
            model=self._mdl,
            tokenizer=self._tok,
        )
        self._temperature = float(temperature)
        self._max_new = int(max_new_tokens)

    def eval(self, *, question: str, answer: str, context: str) -> dict:
        """
        Trả về:
          - label: 'correct' | 'incorrect' | None (nếu model không tuân thủ)
          - raw: toàn bộ text model sinh ra (để debug khi không phải 1 từ)
        """
        prompt = _QA_TEMPLATE.format(question=question, answer=answer, context=context)
        out = self._pipe(
            prompt,
            max_new_tokens=self._max_new,
            temperature=self._temperature,
            do_sample=self._temperature > 0,
            return_full_text=False,
        )[0]["generated_text"].strip()

        m = _LABEL_RE.search(out)
        label = m.group(1).lower() if m else None
        return {"label": label, "raw": out}
