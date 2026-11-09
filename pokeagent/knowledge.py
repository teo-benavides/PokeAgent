"""The centralized informational pipeline: a LangGraph graph that combines RAG + KG.

        ┌──────────────┐
  START─┤ retrieve_docs├──┐
        └──────────────┘  │   ┌────────────┐
        ┌──────────────┐  ├──►│ synthesize ├──► END
  START─┤   query_kg   ├──┘   └────────────┘
        └──────────────┘

Both branches always run in parallel for every informational question; the
synthesizer decides what's relevant. There is no keyword routing anywhere.

The whole graph is exposed to the top-level agent as ONE tool: `ask_pokedex`.
It only reads (Chroma similarity search + a READ_ACCESS Neo4j session), so it
can never perform actions; those belong to the MCP server.
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from pokeagent.kg import SCHEMA, run_read
from pokeagent.llm import llm
from pokeagent.vectorstore import vectorstore

TOP_K_DOCS = 6
CYPHER_ATTEMPTS = 2


class KnowledgeState(TypedDict, total=False):
    question: str
    documents: list[str]
    cypher: str
    kg_results: list[dict]
    kg_error: str | None
    trace: Annotated[list[str], operator.add]   # what each branch did (useful in LangGraph Studio)
    answer: str


# ---------------------------------------------------------------- RAG branch
def retrieve_docs(state: KnowledgeState) -> dict:
    hits = vectorstore().similarity_search(state["question"], k=TOP_K_DOCS)
    docs = [f"[{d.metadata['kind']}: {d.metadata['name']}] {d.page_content}" for d in hits]
    return {"documents": docs, "trace": [f"rag: retrieved {len(docs)} passages"]}


# ----------------------------------------------------------------- KG branch
class CypherQuery(BaseModel):
    cypher: str = Field(description="A single read-only Cypher query, or an empty string if the "
                                    "graph has no relevant structured data for this question.")


CYPHER_PROMPT = """You translate questions about Gen 1 Pokémon into Cypher for a Neo4j graph.

Schema:
{schema}

Rules:
- Read-only queries only (MATCH / OPTIONAL MATCH / WITH / RETURN / ORDER BY / LIMIT).
- Return compact, named columns, and LIMIT results to at most 25 rows.
- Names are lowercase PokeAPI slugs; normalise the user's spelling (e.g. "Mr. Mime" -> "mr-mime").
- Fetch enough context to answer, e.g. types plus stats when comparing Pokémon.
- Return an empty string only if nothing in the schema relates to the question.

Examples:
Q: What is Charizard weak to?
A: MATCH (:Pokemon {{name:'charizard'}})-[:HAS_TYPE]->(d:Type)<-[:SUPER_EFFECTIVE_AGAINST]-(atk:Type)
   RETURN d.name AS defending_type, collect(atk.name) AS weak_to

Q: How does Eevee evolve?
A: MATCH (:Pokemon {{name:'eevee'}})-[e:EVOLVES_INTO]->(to:Pokemon)
   RETURN to.name AS evolves_into, e.trigger AS trigger, e.item AS item, e.min_level AS level

Q: Which electric moves can Pikachu learn by level-up?
A: MATCH (:Pokemon {{name:'pikachu'}})-[l:LEARNS {{method:'level-up'}}]->(m:Move)-[:OF_TYPE]->(:Type {{name:'electric'}})
   RETURN m.name AS move, l.level AS level, m.power AS power ORDER BY level

{feedback}Q: {question}
A:"""


def query_kg(state: KnowledgeState) -> dict:
    # json_schema = provider-native structured output (forced tool calling isn't supported by newer models)
    generator = llm().with_structured_output(CypherQuery, method="json_schema")
    feedback, trace, last_error, q = "", [], None, None
    for _ in range(CYPHER_ATTEMPTS):
        try:
            q = generator.invoke(CYPHER_PROMPT.format(schema=SCHEMA, question=state["question"],
                                                      feedback=feedback))
        except Exception as e:  # model/API failure: degrade to a RAG-only answer instead of crashing
            return {"cypher": "", "kg_results": [], "kg_error": f"query generation failed: {e}",
                    "trace": trace + [f"kg: generation failed ({type(e).__name__})"]}
        if not q.cypher.strip():
            return {"cypher": "", "kg_results": [], "kg_error": None,
                    "trace": trace + ["kg: no relevant graph query"]}
        try:
            rows = run_read(q.cypher)
            return {"cypher": q.cypher, "kg_results": rows, "kg_error": None,
                    "trace": trace + [f"kg: query returned {len(rows)} rows"]}
        except Exception as e:  # syntax error, write attempt rejected, etc. -> let the LLM fix it
            last_error = str(e)
            trace.append(f"kg: query failed ({type(e).__name__})")
            feedback = (f"Your previous query failed.\nQuery: {q.cypher}\nError: {e}\n"
                        f"Write a corrected query.\n\n")
    return {"cypher": q.cypher, "kg_results": [], "kg_error": last_error, "trace": trace}


# ---------------------------------------------------------------- synthesis
SYNTH_PROMPT = """You are a Pokédex. Answer the question using ONLY the evidence below.
Structured facts come from a knowledge graph (authoritative for types, stats, evolutions,
learnsets and type matchups; typings and the type chart are the modern ones, learnsets are from
Red/Blue). Passages come from Pokédex entries and move/ability descriptions.
Combine both when useful. If the evidence doesn't cover something, say so; don't invent facts.

Question: {question}

Knowledge graph query:
{cypher}
Knowledge graph results:
{kg}

Retrieved passages:
{docs}

Answer:"""


def synthesize(state: KnowledgeState) -> dict:
    kg = state.get("kg_results") or []
    kg_text = "\n".join(map(str, kg)) if kg else (
        f"(query failed: {state['kg_error']})" if state.get("kg_error") else "(no results)")
    prompt = SYNTH_PROMPT.format(
        question=state["question"],
        cypher=state.get("cypher") or "(none)",
        kg=kg_text,
        docs="\n".join(state.get("documents") or []) or "(none)",
    )
    return {"answer": llm().invoke(prompt).text}   # .text drops thinking blocks, keeps only the answer


def build_knowledge_graph():
    g = StateGraph(KnowledgeState)
    g.add_node("retrieve_docs", retrieve_docs)
    g.add_node("query_kg", query_kg)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "retrieve_docs")
    g.add_edge(START, "query_kg")
    g.add_edge(["retrieve_docs", "query_kg"], "synthesize")
    g.add_edge("synthesize", END)
    return g.compile(name="pokedex_knowledge")


knowledge_graph = build_knowledge_graph()


@tool
async def ask_pokedex(question: str) -> str:
    """Look up information about Gen 1 Pokémon: Pokédex lore and descriptions, types and
    type matchups, base stats, evolutions, moves and learnsets, abilities and habitats.
    Read-only: this tool never changes anything. Pass a complete, self-contained question."""
    result = await knowledge_graph.ainvoke({"question": question})
    return result["answer"]
