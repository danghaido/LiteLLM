# retriever.py
import json
import os
from functools import lru_cache
from typing import Dict, List

import chromadb
from chonkie import AutoEmbeddings, ChromaHandshake
from FlagEmbedding import FlagReranker
from litellm import embedding
from opentelemetry.trace import Status, StatusCode

from litellm_client.common import CONFIG
from phoenix_tools.trace.tracing import tracer


class LiteLLMEmbeddings:
    """Embedding adapter to make LiteLLM embedding() callable like local embeddings."""

    def __init__(self, model_name: str, api_key: str | None = None, base_url: str | None = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        response = embedding(
            model=self.model_name,
            input=texts,
            api_key=self.api_key,
            api_base=self.base_url,
        )

        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data", [])

        vectors: List[List[float]] = []
        for item in data or []:
            if isinstance(item, dict):
                vectors.append(item.get("embedding", []))
            else:
                vectors.append(getattr(item, "embedding", []))

        if not vectors:
            raise ValueError("LiteLLM embedding response does not contain vectors")
        return vectors

    def __call__(self, text_or_texts: str | List[str]):
        if isinstance(text_or_texts, str):
            return self._embed_many([text_or_texts])[0]
        return self._embed_many(text_or_texts)


def _get_retrieve_embeddings(local_model_name: str):
    retrieve_cfg = CONFIG.retrieve
    retrieve_type = str(getattr(retrieve_cfg, "type", "local")).lower()

    if retrieve_type != "cloud":
        return AutoEmbeddings.get_embeddings(local_model_name)

    cloud_cfg = getattr(retrieve_cfg, "cloud", None)
    cloud_model_name = (
        getattr(cloud_cfg, "model_name", None) or getattr(retrieve_cfg, "model_name", None) or local_model_name
    )
    cloud_api_key = (
        getattr(cloud_cfg, "api_key", None)
        or getattr(retrieve_cfg, "api_key", None)
        or getattr(CONFIG, "api_key", None)
    )
    cloud_base_url = (
        getattr(cloud_cfg, "base_url", None)
        or getattr(retrieve_cfg, "base_url", None)
        or getattr(CONFIG, "url_base", None)
    )

    if not cloud_api_key:
        env_key = (
            getattr(cloud_cfg, "env_key", None)
            or getattr(retrieve_cfg, "env_key", None)
            or getattr(CONFIG, "env_key", None)
        )
        if env_key:
            cloud_api_key = os.getenv(env_key)

    if not cloud_model_name:
        raise ValueError("Missing cloud embedding model_name in retrieve.cloud.model_name")

    return LiteLLMEmbeddings(
        model_name=cloud_model_name,
        api_key=cloud_api_key,
        base_url=cloud_base_url,
    )


# ---------- singletons (init đúng 1 lần) ----------
@lru_cache(maxsize=1)
def _bootstrap(
    path: str = CONFIG.retrieve.path or "DB_TEST",
    collection_name: str = CONFIG.retrieve.collection_name or "journal_db",
    embedding_model: str = CONFIG.retrieve.embedding_model or "BAAI/bge-large-en-v1.5",
    reranker_model: str = CONFIG.retrieve.rerank_model or "BAAI/bge-reranker-v2-m3",
):
    # embeddings
    embeddings = _get_retrieve_embeddings(embedding_model)

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
            docs: List[str] = res["documents"][0]
            metas: List[Dict] = res["metadatas"][0]
            ids: List[str] = res["ids"][0]

            for i, (doc_id, text) in enumerate(zip(ids, docs)):
                notification = f"Retrieved doc {i}: {doc_id} :: {text}"
                span.add_event(notification)

            pairs = [[query, t] for t in docs]
            scores = reranker.compute_score(pairs, normalize=True)

            ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)
            top = ranked[:top_k]

            for idx, (doc_id, text, meta, score) in enumerate(top):
                span.set_attribute(f"retrieval.documents.{idx}.document.id", str(doc_id))
                span.set_attribute(f"retrieval.documents.{idx}.document.score", float(score))
                span.set_attribute(f"retrieval.documents.{idx}.document.content", text)
                if isinstance(meta, dict):
                    for k, v in meta.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            span.set_attribute(f"retrieval.documents.{idx}.document.metadata.{k}", v)
                        else:
                            span.set_attribute(
                                f"retrieval.documents.{idx}.document.metadata.{k}",
                                json.dumps(v),
                            )

            span.set_attribute("retrieval.top_k", int(top_k))

            span.set_status(Status(StatusCode.OK))
            return [text for (_, text, _, _) in top]

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            return None
