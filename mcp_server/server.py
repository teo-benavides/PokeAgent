"""Pokémon Trainer MCP server: the ONLY place where actions happen.

State lives in its own SQLite file (data/team.db). This server shares nothing
with the informational pipeline (Chroma / Neo4j). It only reads the raw
Pokémon name list to validate input.

Add a new action by writing another @mcp.tool() function. The agent discovers
tools at startup, so no agent or routing code has to change.

Run:
  python -m mcp_server.server --http          # streamable HTTP on :8765/mcp (what the agent uses; also VS Code / Copilot)
  python -m mcp_server.server                 # stdio, for clients that launch the server themselves
  mcp dev mcp_server/server.py                # MCP Inspector, to test tools by hand
"""
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # allow `mcp dev mcp_server/server.py` as well as `-m`
from pokeagent.config import RAW_DIR, TEAM_DB  # noqa: E402

MAX_TEAM = 6
mcp = FastMCP("pokemon-trainer", port=8765)

POKEMON = {p["name"]: p for p in json.loads((RAW_DIR / "pokemon.json").read_text(encoding="utf-8"))}


def db() -> sqlite3.Connection:
    TEAM_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TEAM_DB)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS team (
            slot INTEGER PRIMARY KEY CHECK (slot BETWEEN 1 AND 6),
            pokemon TEXT NOT NULL,
            nickname TEXT,
            added_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS caught (
            pokemon TEXT PRIMARY KEY,
            caught_at TEXT NOT NULL
        );
    """)
    return conn


def normalise(name: str) -> str:
    """'Mr. Mime' -> 'mr-mime', 'Nidoran♀' -> 'nidoran-f'."""
    n = name.strip().lower().replace("♀", "-f").replace("♂", "-m")
    n = n.replace(".", "").replace("'", "").replace(" ", "-")
    if n not in POKEMON:
        raise ValueError(f"'{name}' is not a Gen 1 Pokémon.")
    return n


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def team_rows(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT slot, pokemon, nickname FROM team ORDER BY slot")]


# ----------------------------------------------------------------- tools
@mcp.tool()
def view_team() -> dict:
    """Show the trainer's current party (up to 6 Pokémon) with slot numbers and nicknames."""
    with db() as conn:
        team = team_rows(conn)
    return {"team": team, "size": len(team), "max": MAX_TEAM}


@mcp.tool()
def add_to_team(
    pokemon: Annotated[str, Field(description="Gen 1 Pokémon name, e.g. 'Pikachu' or 'Mr. Mime'")],
    nickname: Annotated[str | None, Field(description="Optional nickname")] = None,
) -> dict:
    """Add a Pokémon to the trainer's party in the first free slot. Fails if the party is full."""
    name = normalise(pokemon)
    with db() as conn:
        used = {r["slot"] for r in conn.execute("SELECT slot FROM team")}
        free = next((s for s in range(1, MAX_TEAM + 1) if s not in used), None)
        if free is None:
            raise ValueError("The party is full (6/6). Release a Pokémon first.")
        conn.execute("INSERT INTO team VALUES (?, ?, ?, ?)", (free, name, nickname, now()))
        conn.execute("INSERT OR IGNORE INTO caught VALUES (?, ?)", (name, now()))
        return {"added": name, "nickname": nickname, "slot": free, "team": team_rows(conn)}


@mcp.tool()
def release_from_team(
    slot: Annotated[int, Field(ge=1, le=MAX_TEAM, description="Party slot (1-6) to release")],
) -> dict:
    """Remove the Pokémon in the given party slot. Use view_team first to find the slot."""
    with db() as conn:
        row = conn.execute("SELECT * FROM team WHERE slot = ?", (slot,)).fetchone()
        if row is None:
            raise ValueError(f"Slot {slot} is empty.")
        conn.execute("DELETE FROM team WHERE slot = ?", (slot,))
        return {"released": dict(row)["pokemon"], "slot": slot, "team": team_rows(conn)}


@mcp.tool()
def rename_team_member(
    slot: Annotated[int, Field(ge=1, le=MAX_TEAM, description="Party slot (1-6)")],
    nickname: Annotated[str | None, Field(description="New nickname, or null to clear it")],
) -> dict:
    """Give (or clear) the nickname of the Pokémon in a party slot."""
    with db() as conn:
        cur = conn.execute("UPDATE team SET nickname = ? WHERE slot = ?", (nickname, slot))
        if cur.rowcount == 0:
            raise ValueError(f"Slot {slot} is empty.")
        return {"slot": slot, "nickname": nickname, "team": team_rows(conn)}


@mcp.tool()
def mark_as_caught(
    pokemon: Annotated[str, Field(description="Gen 1 Pokémon name")],
) -> dict:
    """Register a Pokémon as caught in the trainer's Pokédex (without adding it to the party)."""
    name = normalise(pokemon)
    with db() as conn:
        conn.execute("INSERT OR IGNORE INTO caught VALUES (?, ?)", (name, now()))
        total = conn.execute("SELECT count(*) FROM caught").fetchone()[0]
    return {"caught": name, "pokedex_progress": f"{total}/{len(POKEMON)}"}


@mcp.tool()
def pokedex_progress() -> dict:
    """Show which Pokémon the trainer has registered as caught and the completion count."""
    with db() as conn:
        caught = [r["pokemon"] for r in conn.execute("SELECT pokemon FROM caught ORDER BY pokemon")]
    return {"caught": caught, "progress": f"{len(caught)}/{len(POKEMON)}"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
