"""Neo4j connection helpers and the graph schema description given to the LLM."""
from functools import lru_cache

from neo4j import GraphDatabase, READ_ACCESS

from pokeagent.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

# Hand-written schema: clearer for the LLM than an auto-introspected one.
SCHEMA = """
Nodes:
  (:Pokemon {id, name, genus, height_m, weight_kg, color, habitat, is_legendary,
             is_mythical, capture_rate, hp, attack, defense, special_attack,
             special_defense, speed, base_stat_total})
  (:Type {name})
  (:Move {name, power, accuracy, pp, damage_class})   // damage_class: physical | special | status
  (:Ability {name})
  (:Habitat {name})

Relationships:
  (:Pokemon)-[:HAS_TYPE {slot}]->(:Type)                  // slot 1 = primary type
  (:Pokemon)-[:EVOLVES_INTO {trigger, min_level, item}]->(:Pokemon)
  (:Pokemon)-[:LEARNS {method, level}]->(:Move)           // method: level-up | machine | egg | tutor (Red/Blue learnsets)
  (:Pokemon)-[:HAS_ABILITY {hidden}]->(:Ability)
  (:Pokemon)-[:LIVES_IN]->(:Habitat)
  (:Move)-[:OF_TYPE]->(:Type)
  (:Type)-[:SUPER_EFFECTIVE_AGAINST]->(:Type)             // attacking type deals 2x to defending type
  (:Type)-[:NOT_VERY_EFFECTIVE_AGAINST]->(:Type)          // 0.5x
  (:Type)-[:NO_EFFECT_AGAINST]->(:Type)                   // 0x

Conventions: all `name` values are lowercase, hyphenated PokeAPI slugs
(e.g. "mr-mime", "nidoran-f", "thunder-punch"). Only the 151 Gen 1 Pokémon exist.
Typings and the type chart are the CURRENT (modern) ones, including dark, steel and
fairy (e.g. Clefairy is fairy-type). Learnsets are from Red/Blue.
"""


@lru_cache
def driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def run_read(cypher: str, params: dict | None = None, limit: int = 50) -> list[dict]:
    """Run a query in a READ-ONLY session, so the informational path can never mutate data."""
    with driver().session(default_access_mode=READ_ACCESS) as s:
        result = s.run(cypher, params or {})
        return [r.data() for r in result][:limit]
