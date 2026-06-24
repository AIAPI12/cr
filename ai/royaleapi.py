import json, time, random
from typing import List, Dict, Optional

class RoyaleAPI:
    def __init__(self):
        self.base_url = "https://api.royaleapi.com/v1"
        self.cache = {}
        self.cache_time = {}

    def _get(self, endpoint: str, params: Dict = None):
        import urllib.request, urllib.parse
        url = f"{self.base_url}/{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        cache_key = url
        if cache_key in self.cache and time.time() - self.cache_time.get(cache_key, 0) < 3600:
            return self.cache[cache_key]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CR-AI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                self.cache[cache_key] = data
                self.cache_time[cache_key] = time.time()
                return data
        except Exception as e:
            return None

    def get_popular_decks(self, limit: int = 20) -> List[List[str]]:
        try:
            data = self._get("top/decks", {"limit": limit})
            if data:
                decks = []
                for entry in data:
                    cards = entry.get("cards", []) or entry.get("deck", [])
                    if len(cards) == 8:
                        decks.append(cards)
                return decks
        except:
            pass
        return []

    def get_card_stats(self) -> Dict:
        data = self._get("cards")
        if data:
            stats = {}
            for card in data:
                stats[card.get("name")] = {
                    "elixir": card.get("elixir", 0),
                    "rarity": card.get("rarity", "Common"),
                    "win_rate": card.get("winRate", 50),
                    "use_rate": card.get("useRate", 10),
                }
            return stats
        return {}


class DeckOptimizer:
    def __init__(self, self_play_system):
        self.sp = self_play_system
        self.royale_api = RoyaleAPI()
        self.population = []
        self.generation = 0
        self.fitness_cache = {}

    def seed_from_meta(self, count: int = 50):
        api_decks = self.royale_api.get_popular_decks(count)
        for d in api_decks:
            # Filter to cards our simulator supports
            from clasher.data import CardDataLoader
            loader = CardDataLoader()
            defs = loader.load_card_definitions()
            valid = [c for c in d if c in defs]
            if len(valid) == 8:
                self.population.append(valid)
        # Fill with random decks
        while len(self.population) < count:
            self.population.append(self.sp.get_random_deck())
        return self.population

    def evaluate(self, deck: List[str], num_games: int = 10) -> float:
        key = (tuple(deck), num_games)
        if key in self.fitness_cache:
            return self.fitness_cache[key]
        wr = self.sp.get_win_rate(deck, num_games)
        self.fitness_cache[key] = wr
        return wr

    def crossover(self, d1: List[str], d2: List[str]) -> List[str]:
        split = random.randint(2, 6)
        child = d1[:split] + d2[split:]
        child = list(dict.fromkeys(child))
        if len(child) < 8:
            pool = [c for c in d1 + d2 if c not in child]
            random.shuffle(pool)
            child.extend(pool[:8 - len(child)])
        return child[:8]

    def mutate(self, deck: List[str], mutation_rate: float = 0.2) -> List[str]:
        if random.random() > mutation_rate:
            return deck
        from clasher.data import CardDataLoader
        loader = CardDataLoader()
        defs = loader.load_card_definitions()
        all_cards = list(defs.keys())
        idx = random.randint(0, 7)
        new_card = random.choice([c for c in all_cards if c not in deck])
        deck = deck[:]
        deck[idx] = new_card
        return deck

    def evolve(self, generations: int = 10, pop_size: int = 20,
               elite_frac: float = 0.2, eval_games: int = 10) -> List[str]:
        if not self.population:
            self.seed_from_meta(pop_size)
        pop = self.population[:pop_size]
        for gen in range(generations):
            scored = [(self.evaluate(d, eval_games), d) for d in pop]
            scored.sort(key=lambda x: -x[0])
            print(f"[Evo] Gen {gen}: best={scored[0][0]:.2%} median={sum(s[0] for s in scored)/len(scored):.2%}")
            elite = [d for _, d in scored[:max(2, int(pop_size * elite_frac))]]
            new_pop = elite[:]
            while len(new_pop) < pop_size:
                p1 = random.choice(elite)
                p2 = random.choice(elite)
                child = self.crossover(p1, p2)
                child = self.mutate(child, 0.3)
                new_pop.append(child)
            pop = new_pop[:pop_size]
            self.generation = gen + 1
        best = max([(self.evaluate(d, eval_games * 2), d) for d in pop], key=lambda x: x[0])
        print(f"[Evo] Best deck: {best[1]} with {best[0]:.2%} win rate")
        self.fitness_cache.clear()
        return best[1]
