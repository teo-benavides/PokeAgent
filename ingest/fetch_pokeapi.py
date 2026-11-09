"""Download Gen 1 data from PokeAPI and save it as clean JSON in data/raw/.

Raw HTTP responses are cached in data/raw/cache/, so re-running is cheap and
we respect PokeAPI's fair-use policy (cache locally, don't hammer the API).

Usage:  python -m ingest.fetch_pokeapi
"""
import asyncio
import hashlib
import json

import httpx

from pokeagent.config import RAW_DIR

API = "https://pokeapi.co/api/v2"
GEN1 = range(1, 152)
VERSION_GROUP = "red-blue"   # learnsets from the original games
CACHE = RAW_DIR / "cache"
SEM = asyncio.Semaphore(10)


async def get(client: httpx.AsyncClient, url: str) -> dict:
    if not url.startswith("http"):
        url = f"{API}/{url.lstrip('/')}"
    path = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    async with SEM:
        for attempt in range(4):
            try:
                r = await client.get(url)
                r.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)
    data = r.json()
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def english(entries: list[dict], field: str) -> str | None:
    for e in entries:
        if e["language"]["name"] == "en":
            return clean(e[field])
    return None


def clean(text: str) -> str:
    return " ".join(text.replace("\f", " ").replace("\n", " ").split())


def walk_chain(node: dict, out: list[dict]) -> None:
    """Flatten an evolution-chain tree into (from, to, details) edges."""
    for child in node["evolves_to"]:
        d = child["evolution_details"][0] if child["evolution_details"] else {}
        out.append({
            "from": node["species"]["name"],
            "to": child["species"]["name"],
            "trigger": (d.get("trigger") or {}).get("name"),
            "min_level": d.get("min_level"),
            "item": (d.get("item") or {}).get("name"),
            "min_happiness": d.get("min_happiness"),
        })
        walk_chain(child, out)


async def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "pokeagent-training"}) as client:
        print("Fetching Pokémon + species...")
        pokes = await asyncio.gather(*(get(client, f"pokemon/{i}") for i in GEN1))
        species = await asyncio.gather(*(get(client, f"pokemon-species/{i}") for i in GEN1))
        gen1_names = {p["name"] for p in pokes}

        pokemon, move_urls, ability_urls, chain_urls = [], {}, {}, set()
        for p, s in zip(pokes, species):
            # Unique English Pokédex entries across all game versions
            flavor, seen = [], set()
            for e in s["flavor_text_entries"]:
                t = clean(e["flavor_text"])
                if e["language"]["name"] == "en" and t.lower() not in seen:
                    seen.add(t.lower())
                    flavor.append({"version": e["version"]["name"], "text": t})

            learnset = []
            for m in p["moves"]:
                for d in m["version_group_details"]:
                    if d["version_group"]["name"] == VERSION_GROUP:
                        learnset.append({
                            "move": m["move"]["name"],
                            "method": d["move_learn_method"]["name"],
                            "level": d["level_learned_at"],
                        })
                        move_urls[m["move"]["name"]] = m["move"]["url"]

            for a in p["abilities"]:
                ability_urls[a["ability"]["name"]] = a["ability"]["url"]
            chain_urls.add(s["evolution_chain"]["url"])

            pokemon.append({
                "id": p["id"],
                "name": p["name"],
                "genus": next((g["genus"] for g in s["genera"] if g["language"]["name"] == "en"), None),
                "height_m": p["height"] / 10,
                "weight_kg": p["weight"] / 10,
                "types": [t["type"]["name"] for t in sorted(p["types"], key=lambda t: t["slot"])],
                "abilities": [{"name": a["ability"]["name"], "hidden": a["is_hidden"]} for a in p["abilities"]],
                "stats": {st["stat"]["name"].replace("-", "_"): st["base_stat"] for st in p["stats"]},
                "habitat": (s["habitat"] or {}).get("name"),
                "color": s["color"]["name"],
                "is_legendary": s["is_legendary"],
                "is_mythical": s["is_mythical"],
                "capture_rate": s["capture_rate"],
                "flavor_texts": flavor,
                "learnset": learnset,
            })

        print(f"Fetching {len(move_urls)} moves, {len(ability_urls)} abilities, "
              f"{len(chain_urls)} evolution chains, types...")
        move_data = await asyncio.gather(*(get(client, u) for u in move_urls.values()))
        ability_data = await asyncio.gather(*(get(client, u) for u in ability_urls.values()))
        chain_data = await asyncio.gather(*(get(client, u) for u in chain_urls))
        type_list = await get(client, "type?limit=100")
        type_data = await asyncio.gather(*(get(client, t["url"]) for t in type_list["results"]))

    moves = [{
        "name": m["name"],
        "type": m["type"]["name"],
        "power": m["power"],
        "accuracy": m["accuracy"],
        "pp": m["pp"],
        "damage_class": m["damage_class"]["name"],
        "effect": english(m["effect_entries"], "short_effect"),
        "effect_chance": m["effect_chance"],
        "description": english(m["flavor_text_entries"], "flavor_text"),
    } for m in move_data]

    abilities = [{
        "name": a["name"],
        "effect": english(a["effect_entries"], "effect"),
        "short_effect": english(a["effect_entries"], "short_effect"),
    } for a in ability_data]

    evolutions = []
    for c in chain_data:
        walk_chain(c["chain"], evolutions)
    evolutions = [e for e in evolutions if e["from"] in gen1_names and e["to"] in gen1_names]

    types = [{
        "name": t["name"],
        "super_effective_against": [x["name"] for x in t["damage_relations"]["double_damage_to"]],
        "not_very_effective_against": [x["name"] for x in t["damage_relations"]["half_damage_to"]],
        "no_effect_against": [x["name"] for x in t["damage_relations"]["no_damage_to"]],
    } for t in type_data if t["name"] not in ("unknown", "shadow", "stellar")]

    for name, obj in [("pokemon", pokemon), ("moves", moves), ("abilities", abilities),
                      ("evolutions", evolutions), ("types", types)]:
        (RAW_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {name}.json ({len(obj)} records)")


if __name__ == "__main__":
    asyncio.run(main())
