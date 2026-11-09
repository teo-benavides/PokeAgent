"""Load the structured Pokédex data into Neo4j (the knowledge graph).

Usage:  python -m ingest.build_graph      (Neo4j must be running: docker compose up -d)
"""
import json

from pokeagent.config import RAW_DIR
from pokeagent.kg import driver


def load(name: str):
    return json.loads((RAW_DIR / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    pokemon, moves, abilities = load("pokemon"), load("moves"), load("abilities")
    evolutions, types = load("evolutions"), load("types")

    for p in pokemon:
        p.update(p.pop("stats"))
        p["base_stat_total"] = sum(p[k] for k in
                                   ("hp", "attack", "defense", "special_attack", "special_defense", "speed"))

    with driver().session() as s:
        print("Clearing graph...")
        s.run("MATCH (n) DETACH DELETE n")
        for label in ("Pokemon", "Type", "Move", "Ability", "Habitat"):
            s.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")

        print("Types + type chart...")
        s.run("UNWIND $types AS t MERGE (:Type {name: t.name})", types=types)
        for rel, key in [("SUPER_EFFECTIVE_AGAINST", "super_effective_against"),
                         ("NOT_VERY_EFFECTIVE_AGAINST", "not_very_effective_against"),
                         ("NO_EFFECT_AGAINST", "no_effect_against")]:
            s.run(f"""
                UNWIND $types AS t UNWIND t.{key} AS target
                MATCH (a:Type {{name: t.name}}), (b:Type {{name: target}})
                MERGE (a)-[:{rel}]->(b)""", types=types)

        print("Moves + abilities...")
        s.run("""
            UNWIND $moves AS m
            MERGE (mv:Move {name: m.name})
              SET mv.power = m.power, mv.accuracy = m.accuracy, mv.pp = m.pp,
                  mv.damage_class = m.damage_class
            WITH mv, m MATCH (t:Type {name: m.type}) MERGE (mv)-[:OF_TYPE]->(t)""", moves=moves)
        s.run("UNWIND $abilities AS a MERGE (:Ability {name: a.name})", abilities=abilities)

        print("Pokémon...")
        s.run("""
            UNWIND $pokemon AS p
            MERGE (pk:Pokemon {name: p.name})
              SET pk.id = p.id, pk.genus = p.genus, pk.height_m = p.height_m,
                  pk.weight_kg = p.weight_kg, pk.color = p.color, pk.habitat = p.habitat,
                  pk.is_legendary = p.is_legendary, pk.is_mythical = p.is_mythical,
                  pk.capture_rate = p.capture_rate, pk.hp = p.hp, pk.attack = p.attack,
                  pk.defense = p.defense, pk.special_attack = p.special_attack,
                  pk.special_defense = p.special_defense, pk.speed = p.speed,
                  pk.base_stat_total = p.base_stat_total
            WITH pk, p
            UNWIND range(0, size(p.types) - 1) AS i
            MATCH (t:Type {name: p.types[i]})
            MERGE (pk)-[:HAS_TYPE {slot: i + 1}]->(t)""", pokemon=pokemon)
        s.run("""
            UNWIND $pokemon AS p UNWIND p.abilities AS a
            MATCH (pk:Pokemon {name: p.name}), (ab:Ability {name: a.name})
            MERGE (pk)-[:HAS_ABILITY {hidden: a.hidden}]->(ab)""", pokemon=pokemon)
        s.run("""
            UNWIND $pokemon AS p WITH p WHERE p.habitat IS NOT NULL
            MATCH (pk:Pokemon {name: p.name})
            MERGE (h:Habitat {name: p.habitat}) MERGE (pk)-[:LIVES_IN]->(h)""", pokemon=pokemon)

        print("Learnsets...")
        s.run("""
            UNWIND $pokemon AS p UNWIND p.learnset AS l
            MATCH (pk:Pokemon {name: p.name}), (mv:Move {name: l.move})
            MERGE (pk)-[r:LEARNS {method: l.method}]->(mv)
              SET r.level = CASE WHEN l.level > 0 THEN l.level ELSE null END""", pokemon=pokemon)

        print("Evolutions...")
        s.run("""
            UNWIND $evos AS e
            MATCH (a:Pokemon {name: e.from}), (b:Pokemon {name: e.to})
            MERGE (a)-[r:EVOLVES_INTO]->(b)
              SET r.trigger = e.trigger, r.min_level = e.min_level, r.item = e.item""", evos=evolutions)

        counts = s.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label").data()
        rels = s.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY rel").data()
    print("Nodes:", {c["label"]: c["n"] for c in counts})
    print("Relationships:", {r["rel"]: r["n"] for r in rels})


if __name__ == "__main__":
    main()
