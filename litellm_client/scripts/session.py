import re
import uuid
from pathlib import Path

import chromadb
import gradio as gr
import pandas as pd
from openinference.instrumentation import using_session
from opentelemetry.trace import Status, StatusCode

from embed_pdf import chunk_text_fixed, embed_chunks, extract_pdf_text, upsert_to_chroma
from litellm_client.common import CONFIG
from litellm_client.lite import LiteLLMClient
from litellm_client.response import ResponseInput
from phoenix_tools.trace.tracing import tracer
from tools.factory import user
from tools.rag import build_prompt

client = LiteLLMClient()


def respond(user_msg, history, request: gr.Request):
    # 1) Lấy session id ổn định cho tab hiện tại
    #    Dùng session_hash của Gradio; fallback sang UUID nếu không có
    sid = getattr(request, "session_hash", None) or str(uuid.uuid4())

    with using_session(sid):
        with tracer.start_as_current_span("Thought") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("input.value", user_msg)
            try:
                prompt = build_prompt(user_msg, top_k=5)
                msg: ResponseInput = user(prompt)
                response = client.complete([msg])
                out = response.get("content", "")

                span.set_attribute("output.value", out)
                span.set_status(Status(StatusCode.OK))
                return out
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                return f"[ERROR] {e}"


def _collection_count(persist_directory: str, collection_name: str) -> int:
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        col = client.get_collection(name=collection_name)
        return col.count()
    except Exception:
        return 0


def get_collection_overview() -> str:
    cfg_retrieve = CONFIG.retrieve
    collection_name: str = cfg_retrieve.collection_name
    persist_directory: str = cfg_retrieve.path
    embedding_model: str = cfg_retrieve.embedding_model
    count = _collection_count(persist_directory, collection_name)
    return (
        "### Collection Status\n"
        f"- Collection: **{collection_name}**\n"
        f"- Chroma path: **{persist_directory}**\n"
        f"- Embedding model: **{embedding_model}**\n"
        f"- Chunks hien tai: **{count}**"
    )


def embed_pdf_from_ui(pdf_file: str | None, chunk_size: int) -> tuple[str, str]:
    if not pdf_file:
        return "[ERROR] Chua chon file PDF.", get_collection_overview()

    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        return f"[ERROR] File khong ton tai: {pdf_path}", get_collection_overview()
    if pdf_path.suffix.lower() != ".pdf":
        return "[ERROR] Chi chap nhan file .pdf", get_collection_overview()

    cfg_retrieve = CONFIG.retrieve
    embedding_model: str = cfg_retrieve.embedding_model
    collection_name: str = cfg_retrieve.collection_name
    persist_directory: str = cfg_retrieve.path
    cfg_api_key: str = getattr(CONFIG, "api_key", "API_KEY")
    cfg_env_key: str = getattr(CONFIG, "env_key", "")

    before_count = _collection_count(persist_directory, collection_name)

    try:
        raw_text = extract_pdf_text(str(pdf_path))
        if not raw_text.strip():
            return "[WARN] Khong trich duoc text tu PDF.", get_collection_overview()

        chunks = chunk_text_fixed(raw_text, chunk_size=chunk_size)
        vectors = embed_chunks(
            chunks,
            embedding_model=embedding_model,
            api_key=cfg_api_key,
            env_key=cfg_env_key,
        )

        ids = [str(uuid.uuid4()) for _ in chunks]
        upsert_to_chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            ids=ids,
            documents=chunks,
            embeddings=vectors,
            source_file_name=pdf_path.name,
        )

        after_count = _collection_count(persist_directory, collection_name)
        dim = len(vectors[0]) if vectors else 0
        embed_result = (
            "[OK] Embed thanh cong\n"
            f"- File: {pdf_path.name}\n"
            f"- Chunks moi: {len(chunks)}\n"
            f"- Vector dim: {dim}\n"
            f"- Collection: {collection_name}\n"
            f"- Tong chunks: {before_count} -> {after_count}"
        )
        return embed_result, get_collection_overview()
    except Exception as e:
        return f"[ERROR] {e}", get_collection_overview()


def clear_chunks_from_ui() -> tuple[str, str]:
    cfg_retrieve = CONFIG.retrieve
    collection_name: str = cfg_retrieve.collection_name
    persist_directory: str = cfg_retrieve.path

    before_count = _collection_count(persist_directory, collection_name)

    try:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        try:
            chroma_client.delete_collection(name=collection_name)
        except Exception:
            # Neu collection chua ton tai thi van tao moi de app tiep tuc hoat dong.
            pass

        chroma_client.get_or_create_collection(name=collection_name)
        after_count = _collection_count(persist_directory, collection_name)

        clear_result = (
            "[OK] Da xoa toan bo chunks trong DB\n"
            f"- Collection: {collection_name}\n"
            f"- Tong chunks: {before_count} -> {after_count}"
        )
        return clear_result, get_collection_overview()
    except Exception as e:
        return f"[ERROR] {e}", get_collection_overview()


def _find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _parse_eval_output(raw_text: str) -> tuple[str, str]:
    text = (raw_text or "").strip()
    if not text:
        return "fail", "Empty evaluator output"

    label_match = re.search(r"LABEL\s*:\s*\"?(pass|fail)\"?", text, flags=re.I)
    if label_match:
        label = label_match.group(1).lower()
    elif re.search(r"\bpass\b", text, flags=re.I):
        label = "pass"
    elif re.search(r"\bfail\b", text, flags=re.I):
        label = "fail"
    else:
        label = "fail"

    explanation_match = re.search(
        r"EXPLANATION\s*:\s*(.*?)(?:\n\s*LABEL\s*:|$)",
        text,
        flags=re.I | re.S,
    )
    explanation = explanation_match.group(1).strip() if explanation_match else text[:1000]
    return label, explanation


def evaluate_csv_from_ui(csv_file: str | None) -> tuple[str, str | None]:
    if not csv_file:
        return "[ERROR] Chua chon file CSV.", None

    csv_path = Path(csv_file)
    if not csv_path.exists():
        return f"[ERROR] File khong ton tai: {csv_path}", None
    if csv_path.suffix.lower() != ".csv":
        return "[ERROR] Chi chap nhan file .csv", None

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return f"[ERROR] Khong doc duoc CSV: {e}", None

    input_col = _find_first_existing_column(df, ["input", "input_input", "question", "query"])
    output_col = _find_first_existing_column(df, ["output", "output_output", "answer", "response"])
    expected_col = _find_first_existing_column(
        df,
        ["expected", "ground_truth", "expected_answer", "reference", "gold"],
    )

    missing = []
    if not input_col:
        missing.append("input")
    if not output_col:
        missing.append("output")
    if not expected_col:
        missing.append("expected")

    if missing:
        return (
            "[ERROR] Thieu cot bat buoc: "
            + ", ".join(missing)
            + f"\n- Cac cot hien co: {', '.join(df.columns.astype(str).tolist())}",
            None,
        )

    eval_df = (
        df[[input_col, output_col, expected_col]]
        .rename(
            columns={
                input_col: "input",
                output_col: "output",
                expected_col: "ground_truth",
            }
        )
        .fillna("")
    )

    if eval_df.empty:
        return "[ERROR] CSV khong co dong du lieu nao de evaluate.", None

    cfg_eval = getattr(CONFIG, "eval_model", None)
    cfg_retrieve = getattr(CONFIG, "retrieve", None)
    cfg_cloud = getattr(cfg_retrieve, "cloud", None) if cfg_retrieve else None

    model_name = "gemini-3-flash-preview"
    api_key = "AIzaSyAEw_lPvc0g6KhuzQbPGadv490CmHBns98"
    base_url = getattr(cfg_cloud, "base_url", "") or getattr(CONFIG, "url_base", "")
    temperature = float(getattr(cfg_eval, "temperature", 0.0) or 0.0)

    if not model_name:
        return "[ERROR] Thieu model_name cho evaluator trong config.", None

    eval_client = LiteLLMClient(model_name=model_name, temperature=temperature)
    completion_kwargs = {}
    if api_key:
        completion_kwargs["api_key"] = api_key
    if base_url:
        completion_kwargs["base_url"] = base_url

    eval_rows: list[dict] = []
    try:
        for _, row in eval_df.iterrows():
            eval_prompt = (
                "You are an evaluator for QA responses. "
                "Compare model output with ground truth based on the input question.\n\n"
                f"[Input]\n{row['input']}\n\n"
                f"[Model Output]\n{row['output']}\n\n"
                f"[Ground Truth]\n{row['ground_truth']}\n\n"
                "Return exactly with this format:\n"
                "EXPLANATION: <short reason>\n"
                "LABEL: pass|fail\n"
                "Rules: pass if output is correct in substance and consistent with ground truth; "
                "otherwise fail."
            )
            eval_response = eval_client.complete([user(eval_prompt)], **completion_kwargs)
            raw_eval = str(eval_response.get("content", ""))
            label, explanation = _parse_eval_output(raw_eval)
            eval_rows.append(
                {
                    "label": label,
                    "explanation": explanation,
                    "score": 1.0 if label == "pass" else 0.0,
                    "raw_eval_output": raw_eval,
                }
            )
    except Exception as e:
        return f"[ERROR] Evaluate that bai: {e}", None

    eval_results_df = pd.DataFrame(eval_rows)
    combined_df = pd.concat(
        [eval_df.reset_index(drop=True), eval_results_df.reset_index(drop=True)],
        axis=1,
    )

    pass_rate = (combined_df["label"].astype(str).str.lower() == "pass").mean()

    output_path = csv_path.with_name(f"{csv_path.stem}_eval_result.csv")
    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    status = (
        "[OK] Evaluate CSV thanh cong\n"
        f"- So dong danh gia: {len(combined_df)}\n"
        f"- Cot su dung: input={input_col}, output={output_col}, expected={expected_col}\n"
        f"- Eval model: {model_name}\n"
        f"- Eval base_url: {base_url or '(mac dinh tu provider)'}\n"
        f"- Pass rate: {pass_rate * 100:.2f}%\n"
        f"- File ket qua: {output_path.name}"
    )
    return status, str(output_path)


custom_css = """
body, .gradio-container {
  font-family: "Segoe UI", "Noto Sans", sans-serif;
}
.app-hero {
  border: 1px solid #d1e3ff;
  border-radius: 14px;
  padding: 14px 18px;
  background: linear-gradient(135deg, #f5f9ff 0%, #ebfff5 100%);
  margin-bottom: 10px;
}
.app-hero h2 {
  margin: 0 0 6px 0;
  color: #0f2f62;
}
.app-hero p {
  margin: 0;
  color: #28456f;
}
#clear-db-btn {
    background: #b42318 !important;
    border-color: #8e1c13 !important;
    color: #ffffff !important;
    font-weight: 600;
}
#clear-db-btn:hover {
    background: #8e1c13 !important;
    border-color: #6f140f !important;
}
"""


with gr.Blocks(
    title="RAG Chat + PDF Embedding (Phoenix sessions)",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=custom_css,
) as demo:
    gr.HTML(
        """
        <div class='app-hero'>
            <h2>RAG Demo: Tracking + PDF Embedding</h2>
            <p>
                Embed file PDF vao Chroma de cap nhat kho tri thuc, sau do chat de demo RAG va theo doi trace tren Phoenix.
            </p>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            collection_overview = gr.Markdown("Dang tai thong tin collection...")
        with gr.Column(scale=1, min_width=160):
            refresh_button = gr.Button("Refresh DB Status", variant="secondary")

    with gr.Tabs():
        with gr.Tab("Embed PDF"):
            with gr.Group():
                pdf_file = gr.File(
                    label="Chon file PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                chunk_size = gr.Slider(
                    minimum=128,
                    maximum=2048,
                    step=64,
                    value=512,
                    label="Chunk size",
                    info="Khuyen nghi: 384-768 cho tai lieu hoc thuat",
                )
                embed_button = gr.Button("Embed vao DB", variant="primary")
                clear_button = gr.Button(
                    "Xoa toan bo chunks",
                    variant="stop",
                    elem_id="clear-db-btn",
                )
                embed_status = gr.Textbox(
                    label="Embedding status",
                    lines=8,
                    interactive=False,
                    placeholder="Ket qua embed se hien thi o day...",
                )

        with gr.Tab("Chat + Tracking"):
            gr.ChatInterface(
                fn=respond,
                title="RAG Chat (Phoenix sessions)",
                description="Moi tab la mot session; Phoenix se group trace theo session",
            )

        with gr.Tab("CSV Evaluation"):
            with gr.Group():
                csv_file = gr.File(
                    label="Keo tha file CSV",
                    file_types=[".csv"],
                    type="filepath",
                )
                csv_eval_button = gr.Button("Evaluate CSV", variant="primary")
                csv_eval_status = gr.Textbox(
                    label="CSV evaluation status",
                    lines=8,
                    interactive=False,
                    placeholder="Ket qua evaluation se hien thi o day...",
                )
                csv_eval_output_file = gr.File(
                    label="File ket qua evaluation",
                    interactive=False,
                )

    demo.load(fn=get_collection_overview, outputs=collection_overview)
    refresh_button.click(fn=get_collection_overview, outputs=collection_overview)
    embed_button.click(
        fn=embed_pdf_from_ui,
        inputs=[pdf_file, chunk_size],
        outputs=[embed_status, collection_overview],
    )
    clear_button.click(
        fn=clear_chunks_from_ui,
        outputs=[embed_status, collection_overview],
    )
    csv_eval_button.click(
        fn=evaluate_csv_from_ui,
        inputs=[csv_file],
        outputs=[csv_eval_status, csv_eval_output_file],
    )

if __name__ == "__main__":
    demo.launch()
