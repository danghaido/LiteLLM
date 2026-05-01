"""
embed_pdf.py
──────────────────────────────────────────────────────────────────────────────
Interactive REPL: nhận file PDF (paper NCKH), chunking + embedding,
lưu trực tiếp vào Chroma (collection_name + path từ configs/dev.yaml).

Song song với luồng chính — không sửa bất kỳ file nào trong luồng chính.

Usage:
    python embed_pdf.py [--chunk-size 512] [--overlap 64]

REPL commands:
    <duong_dan_pdf>   — embed file PDF
    status            — xem so chunks hien tai trong collection
    reset             — xoa collection hien tai
    quit / q / exit   — thoat

Vi du REPL:
    embed> paper1.pdf
    embed> paper2.pdf
    embed> status
    embed> reset
    embed> quit
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import inspect
import os
import sys
import uuid
from pathlib import Path

import chromadb
import pypdf
from chonkie import AutoEmbeddings, RecursiveChunker

# ── Local LiteLLM config loader ───────────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from litellm_client.common import CONFIG
except Exception as exc:  # pragma: no cover — fallback nếu chạy standalone
    sys.stderr.write(f"[WARN] Khong load duoc CONFIG tu LiteLLM: {exc}\n")
    # Fallback: đọc trực tiếp YAML
    import yaml

    _cfg_path = Path(__file__).parent / "configs" / "dev.yaml"
    with open(_cfg_path, encoding="utf-8") as f:
        _raw: dict = yaml.safe_load(f)
    _retrieve = _raw.get("retrieve", {})
    _cfg = type(
        "CONFIG",
        (),
        {
            "retrieve": type(
                "obj",
                (),
                {
                    "path": _retrieve.get("path", ".chroma_default"),
                    "embedding_model": _retrieve.get("embedding_model", "BAAI/bge-m3"),
                    "collection_name": _retrieve.get("collection_name", "journal_db"),
                },
            )()
        },
    )()

    CONFIG = _cfg  # type: ignore[assignment]


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: str) -> str:
    """Trích toàn bộ text từ file PDF."""
    reader = pypdf.PdfReader(pdf_path)
    pages_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text)
    return "\n\n".join(pages_text)


def _build_recursive_chunker(chunk_size: int, overlap: int) -> RecursiveChunker:
    """Build chunker, compatible across chonkie versions."""
    overlap = max(0, overlap)
    kwargs: dict[str, int] = {"chunk_size": chunk_size}

    try:
        init_params = set(inspect.signature(RecursiveChunker.__init__).parameters.keys())
    except (TypeError, ValueError):
        init_params = set()

    if "chunk_overlap" in init_params:
        kwargs["chunk_overlap"] = overlap
    elif "overlap" in init_params:
        kwargs["overlap"] = overlap

    return RecursiveChunker(**kwargs)


def chunk_text_fixed(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Chunking theo cấu trúc hierarchical của paper NCKH,
    dùng RecursiveChunker (chonkie).

    Split theo thứ tự ưu tiên:
        1. Double newline (paragraph)
        2. Single newline
        3. Single space
        4. Cuối cùng mới cắt theo token

    Args:
        text:       văn bản nguồn
        chunk_size: số token mỗi chunk
        overlap:    số token overlap giữa các chunk liên tiếp

    Returns:
        Danh sách các chunk text
    """
    chunker = _build_recursive_chunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk(text)
    return [c.text for c in chunks]


def embed_chunks(
    chunks: list[str],
    embedding_model: str,
    api_key: str | None,
    env_key: str | None,
) -> list[list[float]]:
    """
    Embed danh sách chunks bằng AutoEmbeddings (chonkie).
    Trả về list-of-list thuần Python để tương thích Chroma.
    """
    _key = api_key if (api_key and api_key != "API_KEY") else os.environ.get(env_key or "") or ""

    embeddings = AutoEmbeddings.get_embeddings(embedding_model)

    if _key:
        try:
            embeddings.token = _key  # type: ignore[attr-defined]
        except Exception:
            pass

    import numpy as np

    result = embeddings(chunks)
    return np.array(result).tolist() if hasattr(result, "tolist") else list(result)


def upsert_to_chroma(
    collection_name: str,
    persist_directory: str,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    source_file_name: str,
) -> None:
    """
    Tạo (hoặc lấy) Chroma collection rồi append documents.
    Mỗi document được gắn metadata: file_name, source_file và chunk_index.
    """
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)

    metadatas = [
        {
            "file_name": source_file_name,
            "source_file": source_file_name,
            "chunk_index": i,
        }
        for i in range(len(documents))
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        f"[OK] Da append {len(documents)} chunks vao collection '{collection_name}' (tong: {collection.count()} chunks)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def _show_banner(
    embedding_model: str,
    persist_directory: str,
    collection_name: str,
    chunk_size: int,
    overlap: int,
) -> None:
    print(
        f"""
╔══════════════════════════════════════════════════╗
║         LiteLLM — PDF Embed REPL                ║
╠══════════════════════════════════════════════════╣
║  Embedding model : {embedding_model:<28}║
║  Chroma path     : {persist_directory:<28}║
║  Collection      : {collection_name:<28}║
║  Chunk size      : {chunk_size:<28}║
║  Overlap         : {overlap:<28}║
╠══════════════════════════════════════════════════╣
║  Nhap duong dan PDF de embed.                     ║
║  goi 'status'  — xem so chunks hien tai           ║
║  goi 'reset'   — xoa collection hien tai          ║
║  goi 'quit'    — thoat                            ║
╚══════════════════════════════════════════════════╝
"""
    )


def _process_pdf(
    pdf_path: Path,
    chunk_size: int,
    overlap: int,
    cfg_retrieve,  # type: ignore[type-arg]
    cfg_api_key: str,
    cfg_env_key: str,
) -> None:
    """Embed 1 file PDF: extract -> chunk -> embed -> upsert."""
    embedding_model: str = cfg_retrieve.embedding_model
    collection_name: str = cfg_retrieve.collection_name
    persist_directory: str = cfg_retrieve.path

    print(f"[PDF] {pdf_path.name}")

    print("  [1/4] Trich text ...", end=" ", flush=True)
    raw_text = extract_pdf_text(str(pdf_path))
    print(f"{len(raw_text):,} ky tu")

    if not raw_text.strip():
        print("  [WARN] Khong trich duoc text. Bo qua.")
        return

    print("  [2/4] Chunking ...", end=" ", flush=True)
    chunks = chunk_text_fixed(raw_text, chunk_size=chunk_size, overlap=overlap)
    print(f"{len(chunks)} chunks (overlap={overlap})")

    print("  [3/4] Embedding ...", end=" ", flush=True)
    vectors = embed_chunks(
        chunks,
        embedding_model=embedding_model,
        api_key=cfg_api_key,
        env_key=cfg_env_key,
    )
    dim = len(vectors[0]) if vectors else 0
    print(f"{len(vectors)} vectors (dim={dim})")

    print("  [4/4] Luu vao Chroma ...", end=" ", flush=True)
    ids = [str(uuid.uuid4()) for _ in chunks]
    upsert_to_chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        ids=ids,
        documents=chunks,
        embeddings=vectors,
        source_file_name=pdf_path.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive REPL: embed PDF papers into Chroma (config from dev.yaml)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="So token moi chunk (mac dinh: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="So token overlap giua 2 chunk lien tiep (mac dinh: 64)",
    )
    args = parser.parse_args()

    cfg_retrieve = CONFIG.retrieve
    embedding_model: str = cfg_retrieve.embedding_model
    collection_name: str = cfg_retrieve.collection_name
    persist_directory: str = cfg_retrieve.path
    cfg_api_key: str = getattr(CONFIG, "api_key", "API_KEY")
    cfg_env_key: str = getattr(CONFIG, "env_key", "")

    overlap = max(0, args.overlap)

    _show_banner(
        embedding_model=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
        chunk_size=args.chunk_size,
        overlap=overlap,
    )

    while True:
        try:
            user_input = input("embed> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "q", "exit"):
            print("Bye!")
            break

        if cmd == "status":
            try:
                client = chromadb.PersistentClient(path=persist_directory)
                col = client.get_collection(name=collection_name)
                print(f"  Collection '{collection_name}': {col.count()} chunks")
            except Exception:
                print("  Collection chua ton tai (0 chunks)")
            continue

        if cmd == "reset":
            confirm = input(f"  Xoa collection '{collection_name}'? (y/N): ").strip().lower()
            if confirm == "y":
                try:
                    client = chromadb.PersistentClient(path=persist_directory)
                    client.delete_collection(name=collection_name)
                    print(f"  Da xoa collection '{collection_name}'")
                except Exception as e:
                    print(f"  Loi: {e}")
            else:
                print("  Huy.")
            continue

        pdf_path = Path(user_input)
        if not pdf_path.exists():
            print(f"  [ERROR] File khong ton tai: {pdf_path}")
            continue
        if pdf_path.suffix.lower() != ".pdf":
            print("  [ERROR] Chi chap nhan file .pdf")
            continue

        try:
            _process_pdf(
                pdf_path=pdf_path,
                chunk_size=args.chunk_size,
                overlap=overlap,
                cfg_retrieve=cfg_retrieve,
                cfg_api_key=cfg_api_key,
                cfg_env_key=cfg_env_key,
            )
        except Exception as e:
            print(f"  [ERROR] {e}")


if __name__ == "__main__":
    main()
