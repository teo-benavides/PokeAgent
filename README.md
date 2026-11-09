# PokeAgent: a Pokédex chatbot (RAG + Knowledge Graph + MCP)

A chatbot for Gen 1 Pokémon. It answers questions by combining a vector store
(Pokédex lore) with a Neo4j knowledge graph (types, stats, evolutions, moves).
It acts on the trainer's party through an MCP server.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
  Chat UI ─────────►│  pokedex_agent (LangChain create_agent)      │
 (Agent Chat UI     │  one LLM, flat tool list, native tool-calling│
  or terminal)      └──────────────┬───────────────────┬───────────┘
                                   │                   │
                  informational    │                   │  actions
                                   ▼                   ▼
            ┌─────────────────────────────┐   ┌───────────────────────────┐
            │ ask_pokedex                 │   │ MCP server (FastMCP, HTTP)│
            │ = pokedex_knowledge graph   │   │ view_team, add_to_team,   │
            │                             │   │ release_from_team,        │
            │  retrieve_docs ─┐           │   │ rename_team_member,       │
            │   (Chroma RAG)  ├► synthesize│   │ mark_as_caught,           │
            │  query_kg ──────┘           │   │ pokedex_progress          │
            │   (Neo4j, read-only)        │   │       │                   │
            └─────────────────────────────┘   │   SQLite (team.db)        │
                                              └───────────────────────────┘
```

| Requirement | How it's met |
|---|---|
| Centralized RAG + KG flow | `pokeagent/knowledge.py`: a LangGraph `StateGraph` runs RAG (Chroma) and KG (LLM-written Cypher on Neo4j) in parallel, then synthesizes. It's exposed to the agent as one tool, `ask_pokedex`. |
| Actions only through MCP | `mcp_server/server.py`: a FastMCP server with its own SQLite state. Tool input schemas are generated from type hints and Pydantic `Field`s. |
| Autonomous routing, no if/else | `pokeagent/agent.py`: the LLM chooses tools from their descriptions. No keyword rules or conditionals anywhere. |
| Strict separation | The knowledge graph only reads. Neo4j runs in a `READ_ACCESS` session, so even an LLM-written `CREATE` is rejected. The MCP server never touches Chroma or Neo4j. |
| Extensible without changing routing | MCP tools are discovered at runtime from `mcp_servers.json`. Add an `@mcp.tool()` function, or add another server to the JSON, and the agent picks it up. |
| Passive UI | Agent Chat UI (via `langgraph dev`) or `chat.py` only forwards messages. |

## Setup (once)

Requires Python 3.11+, Docker Desktop, and an LLM API key.

```bash
python -m venv .venv            # or: uv venv .venv --python 3.11
.venv\Scripts\activate          # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env          # then set LLM_MODEL and your API key
docker compose up -d            # Neo4j at http://localhost:7474 (neo4j / pokedex123)

python -m ingest.fetch_pokeapi  # download Gen 1 data from PokeAPI -> data/raw/
python -m ingest.build_vectors  # embed lore into Chroma -> data/chroma/
python -m ingest.build_graph    # load the knowledge graph into Neo4j
```

Embeddings run locally with `fastembed` (`BAAI/bge-small-en-v1.5`), so you don't
need an API key for them. The chat model is any `provider:model` that LangChain's
`init_chat_model` supports, e.g. `anthropic:claude-sonnet-5-5` or `openai:gpt-4.1-mini`.
To use another provider, install its LangChain package.

## Run

Terminal 1, the MCP action server:
```bash
python -m mcp_server.server --http       # http://localhost:8765/mcp
```

Terminal 2, then pick one UI:
```bash
langgraph dev                            # opens LangGraph Studio; use Agent Chat UI below
python chat.py --verbose                 # or: plain terminal chat, shows tool calls
```

**Agent Chat UI:** open https://agentchat.vercel.app, then set
Deployment URL `http://localhost:2024` and Assistant/Graph ID `pokedex_agent`.

## Try these

| Prompt | Expected route |
|---|---|
| "What is Gengar weak to, and what does the Pokédex say about it?" | `ask_pokedex` (KG for types, RAG for lore) |
| "How does Eevee evolve?" | `ask_pokedex` (KG) |
| "Which Pokémon lives in volcanoes?" | `ask_pokedex` (RAG) |
| "Add Pikachu to my team and nickname it Sparky" | `add_to_team` (MCP) |
| "Release whatever is in slot 2" | `release_from_team` (MCP) |
| "Look at my team. Which of them is best against Onix?" | `view_team` (MCP), then `ask_pokedex` |
| "Catch the Pokémon that evolves from Magikarp" | `ask_pokedex`, then `mark_as_caught` |

## Adding a new action (extensibility demo)

1. Add a function to `mcp_server/server.py`:
   ```python
   @mcp.tool()
   def swap_slots(a: int, b: int) -> dict:
       """Swap the Pokémon in two party slots."""
       ...
   ```
2. Restart the MCP server. The agent picks the tool up the next time it loads the graph,
   without any change to the agent or routing code.

## Optional: use the MCP server from VS Code or Copilot

With the HTTP server running, add this to `.vscode/mcp.json`:
```json
{ "servers": { "pokemon-trainer": { "type": "http", "url": "http://localhost:8765/mcp" } } }
```
You can also test the tools by hand with `mcp dev mcp_server/server.py`, which opens the MCP Inspector.

## Layout

```
ingest/          fetch_pokeapi.py, build_vectors.py, build_graph.py   (run once)
pokeagent/       config, llm, vectorstore, kg, knowledge (RAG+KG graph), agent
mcp_server/      server.py (actions; SQLite state in data/team.db)
chat.py          terminal UI
langgraph.json   exposes pokedex_agent and pokedex_knowledge to langgraph dev / Studio
mcp_servers.json MCP servers the agent loads tools from
```
