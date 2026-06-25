from typing import Dict, Any, Optional, List
import json, copy, os

from .card_types import CardDefinition, CardStatsCompat, TroopStats, BuildingStats, SpellStats
from .card_aliases import alias_card_map, resolve_card_name
from .factory.card_factory import card_from_gamedata


SYNTHETIC_CARDS = {
    'HealSpirit': {'id': 9991, 'name': 'HealSpirit', 'manaCost': 1, 'rarity': 'Common',
                   'tidType': 'TID_CARD_TYPE_CHARACTER',
                   'summonCharacterData': {'hitpoints': 350, 'damage': 0, 'range': 4000, 'hitSpeed': 1200,
                   'speed': 120, 'collisionRadius': 300, 'deployTime': 1000, 'loadTime': 500}},
    'FlyingMachine': {'id': 9992, 'name': 'FlyingMachine', 'manaCost': 4, 'rarity': 'Rare',
                      'tidType': 'TID_CARD_TYPE_CHARACTER',
                      'summonCharacterData': {'hitpoints': 800, 'damage': 85, 'range': 6000, 'hitSpeed': 1000,
                      'speed': 60, 'collisionRadius': 300, 'deployTime': 1000, 'loadTime': 500}},
    'Void': {'id': 9993, 'name': 'Void', 'manaCost': 3, 'rarity': 'Epic',
             'tidType': 'TID_CARD_TYPE_SPELL', 'radius': 2500,
             'summonCharacterData': {'damage': 270, 'hitSpeed': 1000}},
    'PrincessTower': {'id': 9994, 'name': 'PrincessTower', 'manaCost': 0, 'rarity': 'Common',
                      'tidType': 'TID_CARD_TYPE_BUILDING',
                      'summonCharacterData': {'hitpoints': 1400, 'damage': 50, 'range': 7500, 'hitSpeed': 800,
                      'collisionRadius': 1000, 'deployTime': 800, 'loadTime': 500}},
    'Cannoneer': {'id': 9995, 'name': 'Cannoneer', 'manaCost': 0, 'rarity': 'Common',
                  'tidType': 'TID_CARD_TYPE_BUILDING',
                  'summonCharacterData': {'hitpoints': 1600, 'damage': 140, 'range': 7000, 'hitSpeed': 1000,
                  'collisionRadius': 1000, 'deployTime': 800, 'loadTime': 500}},
    'DaggerDuchess': {'id': 9996, 'name': 'DaggerDuchess', 'manaCost': 0, 'rarity': 'Common',
                      'tidType': 'TID_CARD_TYPE_BUILDING',
                      'summonCharacterData': {'hitpoints': 1200, 'damage': 20, 'range': 6500, 'hitSpeed': 300,
                      'collisionRadius': 1000, 'deployTime': 800, 'loadTime': 500}},
}

HERO_CARDS = {}
hero_base = ['Goblins', 'EliteArcher', 'MegaMinion', 'BarbLog', 'Wizard', 'Knight',
             'IceGolemite', 'Giant', 'MiniPekka', 'Musketeer', 'Balloon', 'DarkPrince']
hero_names = ['HeroGoblins', 'HeroMagicArcher', 'HeroMegaMinion', 'HeroBarbarianBarrel',
              'HeroWizard', 'HeroKnight', 'HeroIceGolem', 'HeroGiant',
              'HeroMiniPekka', 'HeroMusketeer', 'HeroBalloon', 'HeroDarkPrince']
for hname, base in zip(hero_names, hero_base):
    HERO_CARDS[hname] = {'name': hname, 'manaCost': None, 'rarity': 'Legendary', 'card_type': 'Troop',
                         'base_card': base, 'is_hero': True}


_DEFAULT_DATA_FILE = None

def _find_gamedata() -> str:
    global _DEFAULT_DATA_FILE
    if _DEFAULT_DATA_FILE:
        return _DEFAULT_DATA_FILE
    candidates = [
        "gamedata.json",
    ]
    proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    candidates.append(os.path.join(proj, "gamedata.json"))
    for c in candidates:
        if os.path.exists(c):
            _DEFAULT_DATA_FILE = c
            return c
    return "gamedata.json"


class CardDataLoader:
    def __init__(self, data_file: str = None):
        self.data_file = data_file or _find_gamedata()
        self._cards: Dict[str, CardStatsCompat] = {}
        self._card_definitions: Dict[str, CardDefinition] = {}

    def load_card_definitions(self) -> Dict[str, CardDefinition]:
        """Load card definitions using the factory system."""
        if self._card_definitions:
            return self._card_definitions

        with open(self.data_file, 'r') as f:
            data = json.load(f)

        card_definitions: Dict[str, CardDefinition] = {}

        for entry in data.get("items", {}).get("spells", []):
            card_name = entry.get("name", "")
            if not card_name or card_name.startswith("King_"):
                continue
            if "manaCost" not in entry:
                continue

            try:
                card_definitions[card_name] = card_from_gamedata(entry)
            except Exception as exc:
                print(f"Warning: Could not load card definition for {card_name}: {exc}")

        # Add synthetic cards (HealSpirit, FlyingMachine, Void, Tower Troops)
        for name, raw in SYNTHETIC_CARDS.items():
            if name not in card_definitions:
                try:
                    card_definitions[name] = card_from_gamedata(raw)
                except Exception as exc:
                    print(f"Warning: Could not create synthetic card {name}: {exc}")

        # Add hero cards (clone their base card stats + hero flag)
        for hname, hdata in HERO_CARDS.items():
            if hname not in card_definitions:
                base_name = hdata['base_card']
                base_def = card_definitions.get(base_name)
                if base_def:
                    hraw = dict(base_def.raw)
                    hraw['name'] = hname
                    hraw['is_hero'] = True
                    try:
                        card_definitions[hname] = card_from_gamedata(hraw)
                    except Exception as exc:
                        print(f"Warning: Could not create hero card {hname}: {exc}")

        self._card_definitions = alias_card_map(card_definitions)
        return self._card_definitions

    def load_cards(self) -> Dict[str, CardStatsCompat]:
        """Materialize compatibility stats from card definitions."""
        definitions = self.load_card_definitions()
        cards = {
            name: CardStatsCompat.from_card_definition(card_def)
            for name, card_def in definitions.items()
        }
        self._cards = cards
        return cards

    def get_card(self, name: str) -> Optional[CardStatsCompat]:
        """Get card stats by name using compatibility wrappers."""
        if not self._cards:
            self.load_cards()
        resolved_name = resolve_card_name(name, self._cards)
        return self._cards.get(resolved_name)

    def get_card_definition(self, name: str) -> Optional[CardDefinition]:
        """Get card definition by name."""
        if not self._card_definitions:
            self.load_card_definitions()
        resolved_name = resolve_card_name(name, self._card_definitions)
        return self._card_definitions.get(resolved_name)

    def get_card_compat(self, name: str) -> Optional[CardStatsCompat]:
        """Alias for get_card to preserve API compatibility."""
        return self.get_card(name)
    
    def print_card_summary(self, name: str) -> None:
        """Print a detailed summary of a card's attributes"""
        card = self.get_card(name)
        if not card:
            print(f"Card '{name}' not found")
            return
            
        print(f"=== {card.name} ===")
        print(f"Type: {card.card_type} | Rarity: {card.rarity} | Cost: {card.mana_cost} elixir")
        if card.tribe:
            print(f"Tribe: {card.tribe}")
        if card.unlock_arena:
            print(f"Unlocks: {card.unlock_arena}")
            
        if card.hitpoints or card.damage:
            print(f"\\nCombat:")
            if card.hitpoints:
                print(f"  HP: {card.hitpoints}")
            if card.damage:
                print(f"  Damage: {card.damage}")
            if card.hit_speed:
                print(f"  Attack Speed: {card.hit_speed}ms")
                
        if card.range or card.sight_range or card.speed:
            print(f"\\nMovement & Range:")
            if card.range:
                print(f"  Attack Range: {card.range} tiles")
            if card.sight_range:
                print(f"  Sight Range: {card.sight_range} tiles")
            if card.speed:
                print(f"  Speed: {card.speed} tiles/min")
            if card.collision_radius:
                print(f"  Collision Radius: {card.collision_radius} tiles")
                
        if card.summon_count:
            print(f"\\nDeployment:")
            print(f"  Units Spawned: {card.summon_count}")
            if card.summon_radius:
                print(f"  Spawn Radius: {card.summon_radius} tiles")
            if card.summon_deploy_delay:
                print(f"  Spawn Delay: {card.summon_deploy_delay}ms")
                
        if card.attacks_ground is not None or card.attacks_air is not None:
            print(f"\\nTargeting:")
            if card.attacks_ground:
                print(f"  Attacks Ground: Yes")
            if card.attacks_air:
                print(f"  Attacks Air: Yes")
                
        if card.has_evolution:
            print(f"\\nEvolution: Available")
            
        if card.deploy_time or card.load_time:
            print(f"\\nTiming:")
            if card.deploy_time:
                print(f"  Deploy Time: {card.deploy_time}ms")
            if card.load_time:
                print(f"  Load Time: {card.load_time}ms")
