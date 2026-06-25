"""
Elo tracking and opponent pool for Clash Royale AI.
Tracks Elo ratings across checkpoints, maintains a win-rate matrix,
and implements Prioritized Fictitious Self-Play (PFSP) sampling.
"""
import json, os, math, random
from typing import Dict, List, Tuple, Optional


class EloTracker:
    def __init__(self, k=32, initial_elo=1200):
        self.k = k
        self.initial_elo = initial_elo
        self.ratings: Dict[str, float] = {}  # checkpoint_name -> rating
        self.match_history: List[Tuple[str, str, int]] = []  # (p1, p2, winner: 0/1/draw)

    def get_rating(self, name: str) -> float:
        return self.ratings.get(name, self.initial_elo)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, p1: str, p2: str, winner: int):
        """winner: 0 = p1 won, 1 = p2 won, -1 = draw"""
        r1 = self.get_rating(p1)
        r2 = self.get_rating(p2)
        e1 = self.expected_score(r1, r2)
        e2 = 1.0 - e1

        if winner == 0:
            s1, s2 = 1.0, 0.0
        elif winner == 1:
            s1, s2 = 0.0, 1.0
        else:
            s1, s2 = 0.5, 0.5

        self.ratings[p1] = r1 + self.k * (s1 - e1)
        self.ratings[p2] = r2 + self.k * (s2 - e2)
        self.match_history.append((p1, p2, winner))

    def win_rate_matrix(self, names: List[str]) -> Dict[str, Dict[str, float]]:
        """Build pairwise win-rate matrix from match history."""
        wins = {n: {m: 0 for m in names} for n in names}
        counts = {n: {m: 0 for m in names} for n in names}
        for p1, p2, w in self.match_history:
            if p1 in names and p2 in names:
                counts[p1][p2] += 1
                if w == 0:
                    wins[p1][p2] += 1
                elif w == 1:
                    wins[p2][p1] += 1
        matrix = {}
        for n in names:
            matrix[n] = {}
            for m in names:
                c = counts[n][m]
                matrix[n][m] = wins[n][m] / c if c > 0 else 0.5
        return matrix

    def elo_ladder(self) -> List[Tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: -x[1])

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'ratings': self.ratings,
                'history': [(p1, p2, w) for p1, p2, w in self.match_history],
            }, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.ratings = {k: float(v) for k, v in data.get('ratings', {}).items()}
        self.match_history = [(p1, p2, w) for p1, p2, w in data.get('history', [])]

    def print_ladder(self):
        print('=== Elo Ladder ===')
        for i, (name, rating) in enumerate(self.elo_ladder(), 1):
            print(f'  {i}. {name}: {rating:.0f}')
        print(f'  Total matches: {len(self.match_history)}')


class OpponentPool:
    """
    Simple PFSP (Prioritized Fictitious Self-Play) pool.
    Samples opponents weighted by how much the main agent loses to them.
    Stores past checkpoint snapshots.
    """
    def __init__(self, max_size=20, pfsp_weight=0.7):
        self.max_size = max_size
        self.pfsp_weight = pfsp_weight
        self.checkpoints: List[str] = []  # paths to saved models
        self.elo = EloTracker()

    def add_checkpoint(self, name: str, path: str):
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)

    def sample_opponent(self, main_agent_name: str) -> Optional[str]:
        """Sample an opponent weighted by PFSP (lose more -> sample more)."""
        if not self.checkpoints:
            return None

        main_rating = self.elo.get_rating(main_agent_name)
        weights = []
        for ckpt in self.checkpoints:
            ckpt_name = os.path.basename(ckpt)
            opp_rating = self.elo.get_rating(ckpt_name)
            expected_win = self.elo.expected_score(main_rating, opp_rating)

            # PFSP: weight by (1 - expected_win) = expected loss rate
            pfsp_weight = (1.0 - expected_win) ** 2

            # Mix with uniform
            weight = self.pfsp_weight * pfsp_weight + (1 - self.pfsp_weight) * 1.0
            weights.append(max(weight, 0.01))

        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(self.checkpoints, weights=weights)[0]
