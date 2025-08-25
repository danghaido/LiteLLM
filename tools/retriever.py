# retriever.py
from functools import lru_cache
from typing import List, Tuple, Dict, Any
import json

import chromadb
from chonkie import ChromaHandshake, AutoEmbeddings
from FlagEmbedding import FlagReranker

from Phoenix.trace.tracing import tracer
from opentelemetry.trace import Status, StatusCode

from LiteLLM.common import CONFIG 


# ---------- singletons (init đúng 1 lần) ----------
@lru_cache(maxsize=1)
def _bootstrap(
    path: str = CONFIG.retrieve.path or "DB_TEST",
    collection_name: str = CONFIG.retrieve.collection_name or "journal_db",
    embedding_model: str = CONFIG.retrieve.embedding_model or "BAAI/bge-large-en-v1.5",
    reranker_model: str = CONFIG.retrieve.rerank_model or "BAAI/bge-reranker-v2-m3",
):
    # embeddings
    embeddings = AutoEmbeddings.get_embeddings(embedding_model)

    # chroma
    client = chromadb.PersistentClient(path=path)
    names = [c.name for c in client.list_collections()]
    if collection_name not in names:
        ChromaHandshake(client=client, collection_name=collection_name, embedding_model=embeddings)
    collection = client.get_collection(collection_name)

    # reranker
    reranker = FlagReranker(reranker_model, use_fp16=True, trust_remote_code=True)

    return embeddings, client, collection, reranker


def retrieve_chunks(query: str, top_k: int = 5, fetch_k: int = 50) -> List[str]:
    embeddings, _, collection, reranker = _bootstrap()

    with tracer.start_as_current_span("retrieve") as span:
        span.set_attribute("openinference.span.kind", "RETRIEVER")
        span.set_attribute("input.value", query)
        try:
            qvec = embeddings(query)
            res = collection.query(
                query_embeddings=[qvec],
                n_results=fetch_k,
                include=["documents", "metadatas"],
            )

            docs: List[str]      = res["documents"][0]
            metas: List[Dict]    = res["metadatas"][0]
            ids:   List[str]     = res["ids"][0]

            for i, (doc_id, text) in enumerate(zip(ids, docs)):
                notification = f"Retrieved doc {i}: {doc_id} :: {text[:120]}"
                span.add_event(notification)

            pairs = [[query, t] for t in docs]
            scores = reranker.compute_score(pairs, normalize=True)

            ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)
            top = ranked[:top_k]

            for idx, (doc_id, text, meta, score) in enumerate(top):
                span.set_attribute(f"retrieval.documents.{idx}.document.id", str(doc_id))
                span.set_attribute(f"retrieval.documents.{idx}.document.score", float(score))
                span.set_attribute(f"retrieval.documents.{idx}.document.content", text[:400])
                if isinstance(meta, dict):
                    for k, v in meta.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            span.set_attribute(f"retrieval.documents.{idx}.document.metadata.{k}", v)
                        else:
                            span.set_attribute(f"retrieval.documents.{idx}.document.metadata.{k}", json.dumps(v))

            span.set_attribute("retrieval.top_k", int(top_k))

            span.set_status(Status(StatusCode.OK))
            return [text for (_, text, _, _) in top]

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
