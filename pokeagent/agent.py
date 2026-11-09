"""Top-level agent: one LLM with a flat list of tools.

  tools = [ask_pokedex]                   <- the informational RAG+KG graph (pokeagent/knowledge.py)
        + every tool from every MCP server <- actions (discovered at runtime from mcp_servers.json)

The LLM's native tool-calling decides which to use. There are no if/else or
keyword rules. Adding a tool to the MCP server (or a new server to
mcp_servers.json) makes it available without touching this file.
"""
import json
import sys

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from pokeagent.config import MCP_SERVERS_FILE, ROOT
from pokeagent.knowledge import ask_pokedex
from pokeagent.llm import llm

SYSTEM_PROMPT = """You are a friendly Pokédex assistant for a Pokémon trainer (Gen 1 Pokémon).

You have tools for looking up Pokémon knowledge and tools for managing the trainer's
party and Pokédex. Choose tools based on their descriptions. You may call several tools
in sequence, e.g. look something up and then act on it, or check the party before
recommending something.

Ground every factual claim in tool results; don't rely on memory for Pokémon facts.
Never claim an action was performed unless a tool confirmed it. If a tool returns an
error, explain it to the user."""


def load_mcp_connections() -> dict:
    conns = json.loads(MCP_SERVERS_FILE.read_text(encoding="utf-8"))
    for c in conns.values():
        if c.get("transport") == "stdio":
            if c.get("command") == "python":
                c["command"] = sys.executable   # same venv as the agent
            c.setdefault("cwd", str(ROOT))
    return conns


async def load_action_tools():
    """Discover every tool on every configured MCP server (re-run on each graph load,
    so tools added to a server show up without code changes)."""
    conns = load_mcp_connections()
    try:
        return await MultiServerMCPClient(conns).get_tools()
    except Exception as e:
        urls = ", ".join(c.get("url") or c.get("command", "?") for c in conns.values())
        raise RuntimeError(
            f"Could not reach the MCP server(s) at {urls}.\n"
            f"Start it first, in another terminal:  python -m mcp_server.server --http"
        ) from e


async def build_agent(checkpointer=None):
    tools = [ask_pokedex, *await load_action_tools()]
    return create_agent(llm(), tools, system_prompt=SYSTEM_PROMPT,
                        checkpointer=checkpointer, name="pokedex_agent")


async def make_graph():
    """Entry point for `langgraph dev` (see langgraph.json). The platform handles persistence."""
    return await build_agent()
