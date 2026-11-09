"""Embed the free-text parts of the Pokédex into Chroma (the RAG corpus).

Documents:
  - Pokédex entries (flavor text), chunked a few entries per document
  - Move descriptions / effects
  - Ability effects

Structured facts (types, stats, evolutions, learnsets) live in the knowledge
graph instead; the text here is what you'd *read* about a Pokémon.

Usage:  python -m ingest.build_vectors
"""
import json
import shutil

from langchain_core.documents import Document

from pokeagent.config import CHROMA_DIR, RAW_DIR
from pokeagent.vectorstore import vectorstore

ENTRIES_PER_CHUNK = 5


def title(name: str) -> str:
    return name.replace("-", " ").title()


def load(name: str):
    return json.loads((RAW_DIR / f"{name}.json").read_text(encoding="utf-8"))


def build_docs() -> list[Document]:
    docs = []

    for p in load("pokemon"):
        header = f"{title(p['name'])}, the {p['genus']} (#{p['id']:03d})."
        texts = p["flavor_texts"]
        for i in range(0, len(texts), ENTRIES_PER_CHUNK):
            chunk = texts[i:i + ENTRIES_PER_CHUNK]
            body = "\n".join(f"- ({e['version']}) {e['text']}" for e in chunk)
            docs.append(Document(
                page_content=f"{header} Pokédex entries:\n{body}",
                metadata={"kind": "pokedex_entry", "name": p["name"], "id": p["id"]},
            ))

    for m in load("moves"):
        parts = [f"Move: {title(m['name'])} ({m['type']}-type, {m['damage_class']})."]
        if m["description"]:
            parts.append(m["description"])
        if m["effect"]:
            effect = m["effect"].replace("$effect_chance", str(m["effect_chance"]))
            parts.append(f"Effect: {effect}")
        docs.append(Document(page_content=" ".join(parts),
                             metadata={"kind": "move", "name": m["name"]}))

    for a in load("abilities"):
        text = a["effect"] or a["short_effect"]
        if text:
            docs.append(Document(page_content=f"Ability: {title(a['name'])}. {text}",
                                 metadata={"kind": "ability", "name": a["name"]}))
    return docs


def main() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)   # rebuild from scratch, keeps it idempotent
    docs = build_docs()
    print(f"Embedding {len(docs)} documents (first run downloads the embedding model)...")
    store = vectorstore()
    ids = [f"{d.metadata['kind']}:{d.metadata['name']}:{i}" for i, d in enumerate(docs)]
    for i in range(0, len(docs), 256):
        store.add_documents(docs[i:i + 256], ids=ids[i:i + 256])
    print("Done. Sample search for 'sleeps a lot and is very heavy':")
    for d in store.similarity_search("sleeps a lot and is very heavy", k=3):
        print("  -", d.page_content[:110].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
