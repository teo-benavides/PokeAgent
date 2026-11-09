"""Chroma vector store (persisted on disk) with a local fastembed model."""
import threading
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

from pokeagent.config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL


@lru_cache
def embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


_lock = threading.Lock()
_store: Chroma | None = None


def vectorstore() -> Chroma:
    """Open Chroma once. Locked because parallel tool calls run graph nodes in
    separate threads, and two threads opening the same Chroma folder at once crash."""
    global _store
    with _lock:
        if _store is None:
            _store = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=embeddings(),
                persist_directory=str(CHROMA_DIR),
            )
        return _store
