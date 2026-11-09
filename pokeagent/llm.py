"""Chat model factory. Provider is chosen purely by the LLM_MODEL env var."""
from functools import lru_cache

from langchain.chat_models import init_chat_model

from pokeagent.config import LLM_MODEL


@lru_cache
def llm():
    # No temperature override: newer models (e.g. Claude Sonnet 5.5) reject non-default values.
    return init_chat_model(LLM_MODEL)
