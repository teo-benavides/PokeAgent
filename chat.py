"""Terminal chat UI (passive: it only forwards messages and prints replies).

Usage:  python chat.py          (add --verbose to see tool calls)
"""
import asyncio
import sys
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from pokeagent.agent import build_agent


async def main() -> None:
    verbose = "--verbose" in sys.argv
    try:
        agent = await build_agent(checkpointer=InMemorySaver())
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("Pokédex agent ready. Ctrl+C or 'exit' to quit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        async for step in agent.astream({"messages": [("user", text)]}, config, stream_mode="updates"):
            for update in step.values():
                for msg in (update or {}).get("messages", []):
                    if verbose and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            print(f"  [tool call] {tc['name']}({tc['args']})")
                    elif verbose and msg.type == "tool":
                        print(f"  [tool result] {str(msg.content)[:200]}")
                    elif msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
                        print(f"\nbot> {msg.text if hasattr(msg, 'text') else msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
