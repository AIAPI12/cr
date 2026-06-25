import sys, os, random, math, copy, json, time
from typing import Dict, List, Tuple, Optional
from collections import deque

try:
    import torch; torch.set_num_threads(1)
except ImportError:
    torch = None

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
_SRC = os.path.join(_PROJ, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import logging
logging.disable(logging.CRITICAL)

from clasher.battle import BattleState, pick_tower_troops, TOWER_TROOP_STATS
from clasher.arena import Position
from clasher.spells import SPELL_REGISTRY

FEATURE_CHANNELS = 20
NUM_X = 18
NUM_Y_TROOP = 15
ACTION_DIM = 1 + 4 * NUM_X * NUM_Y_TROOP

BLOCKED_TILES = set([
    (0,14),(0,17),(17,14),(17,17),
    (0,0),(1,0),(2,0),(3,0),(4,0),(5,0),
    (12,0),(13,0),(14,0),(15,0),(16,0),(17,0),
    (0,31),(1,31),(2,31),(3,31),(4,31),(5,31),
    (12,31),(13,31),(14,31),(15,31),(16,31),(17,31),
])


class CRGame:
    def __init__(self, deck0: List[str] = None, deck1: List[str] = None, tower_troop: List[str] = None):
        tt = tower_troop if tower_troop else list(pick_tower_troops())
        self.battle = BattleState(tower_troop=tt)
        self._setup_decks(deck0, deck1)
        self.tower_troop = tt
        self._valid_cache = {}

    def _setup_decks(self, deck0: List[str], deck1: List[str]):
        if deck0: self._set_player_deck(0, deck0)
        if deck1: self._set_player_deck(1, deck1)

    def _set_player_deck(self, pid: int, deck: List[str]):
        p = self.battle.players[pid]
        p.deck = deck[:]
        p.hand = deck[:4]
        p.cycle_queue = deque(deck[4:])
        p.elixir = 5.0

    def step(self, actions: Dict[int, Tuple[int, float, float]]):
        for pid, (card_idx, x, y) in actions.items():
            if card_idx < 0: continue
            player = self.battle.players[pid]
            if card_idx >= len(player.hand): continue
            card_name = player.hand[card_idx]
            pos = Position(x, y)
            self.battle.deploy_card(pid, card_name, pos)

    def tick(self):
        self.battle.step()

    @property
    def game_over(self):
        return self.battle.game_over

    @property
    def winner(self):
        return self.battle.winner

    def get_state_tensor(self, pid: int):
        h, w, c = 8, 18, FEATURE_CHANNELS
        t = [[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(c)]
        b = self.battle
        p = b.players[pid]; op = b.players[1-pid]

        for y in range(h):
            for x in range(w):
                t[0][y][x] = p.king_tower_hp / 4824.0
                t[1][y][x] = op.king_tower_hp / 4824.0

        for e in b.entities.values():
            if not e.is_alive: continue
            if not hasattr(e, 'card_stats') or not e.card_stats: continue
            ex, ey = e.position.x, e.position.y
            hp = e.hitpoints / max(e.max_hitpoints, 1)
            tx = int(ex); ty = int(ey / 4)
            if tx < 0 or tx >= w or ty < 0 or ty >= h: continue
            ch = 2 if e.player_id == pid else 3
            t[ch][ty][tx] = min(1.0, t[ch][ty][tx] + hp * 0.5)

        for y in range(h):
            for x in range(w):
                t[4][y][x] = p.elixir / 10.0
                t[5][y][x] = op.elixir / 10.0

        for i, name in enumerate(p.hand[:4]):
            cs = b.card_loader.get_card(name)
            cost = cs.mana_cost if cs else 5
            for y in range(h):
                for x in range(w):
                    t[6+i][y][x] = cost / 10.0

        cd = p.get_crown_count() - op.get_crown_count()
        for y in range(h):
            for x in range(w):
                t[10][y][x] = max(-1, min(1, cd / 3.0))

        tr = max(0, 300 - b.time) / 300.0
        for y in range(h):
            for x in range(w):
                t[11][y][x] = tr

        if pid == 1:
            for c in range(c):
                t[c] = list(reversed(t[c]))

        return t

    def _get_tile_positions(self, pid: int):
        op = self.battle.players[1-pid]
        zones = []
        if pid == 0:
            zones.append((0, 1, 18, 15))
            zones.append((6, 0, 12, 6))
            if op.left_tower_hp <= 0: zones.append((0, 17, 9, 21))
            if op.right_tower_hp <= 0: zones.append((9, 17, 18, 21))
        else:
            zones.append((0, 17, 18, 31))
            zones.append((6, 26, 12, 32))
            if op.left_tower_hp <= 0: zones.append((0, 11, 9, 15))
            if op.right_tower_hp <= 0: zones.append((9, 11, 18, 15))
        out = []
        for x1, y1, x2, y2 in zones:
            for x in range(int(x1), int(x2)):
                for y in range(int(y1), int(y2)):
                    if (x, y) not in BLOCKED_TILES:
                        out.append((x + 0.5, y + 0.5))
        return out

    def get_valid_actions(self, pid: int):
        actions = [(-1, 0, 0)]
        player = self.battle.players[pid]
        tiles = self._get_tile_positions(pid)
        for ci in range(len(player.hand)):
            name = player.hand[ci]
            cs = self.battle.card_loader.get_card(name)
            if not cs or player.elixir < cs.mana_cost:
                continue
            is_spell = name in SPELL_REGISTRY
            if is_spell:
                for x in range(18):
                    for y in range(32):
                        actions.append((ci, x + 0.5, y + 0.5))
            else:
                for px, py in tiles:
                    actions.append((ci, px, py))
        return actions

    def step_n(self, n: int):
        for _ in range(n):
            if self.game_over: break
            self.tick()

    def clone(self):
        return copy.deepcopy(self)


class SelfPlaySystem:
    def __init__(self, network=None):
        self.network = network
        self.replay_buffer = []
        self.max_buffer_size = 200000
        self.deck_pool = self._decks()
        self._pos_cache = None
        self._pos_cache_pid = None

    def _decks(self):
        return [
            ['Knight','Archer','Musketeer','Fireball','Giant','Minions','Barbarians','Arrows'],
            ['HogRider','Musketeer','Fireball','Cannon','Skeletons','IceSpirits','Log','Knight'],
            ['Pekka','Bandit','BattleRam','EliteArcher','Poison','Zap','Minions','ElectroWizard'],
            ['Golem','BabyDragon','DarkWitch','RageBarbarian','Tornado','Lightning','MegaMinion','Barbarians'],
            ['Knight','GoblinBarrel','Princess','Log','Rocket','IceSpirits','Tornado','Tesla'],
            ['Xbow','Knight','Archer','Fireball','Log','IceSpirits','Skeletons','Tesla'],
            ['RoyalGiant','FirespiritHut','Lightning','Log','Barbarians','MegaMinion','Minions','Zap'],
            ['Miner','Poison','Knight','Minions','Skeletons','IceSpirits','Log','Archer'],
        ]

    def get_random_deck(self):
        return random.choice(self.deck_pool)

    def _batch_policy(self, states, valid_list, temp):
        import torch
        dev = next(self.network.parameters()).device
        st = torch.FloatTensor(states).to(dev)
        with torch.inference_mode():
            logits, vals = self.network(st)
        probs = torch.exp(logits).cpu().numpy()

        out = []
        for pi in range(len(states)):
            valid = valid_list[pi]
            if not valid:
                out.append((-1, 0, 0))
                continue
            vidx = []
            for a in valid:
                ci, x, y = a
                if ci < 0: i = 0
                else: i = 1 + ci * (NUM_X * NUM_Y_TROOP) + int(x - 0.5) * NUM_Y_TROOP + int(y // 4)
                vidx.append(min(i, len(probs[pi]) - 1))
            vp = [max(probs[pi][i], 1e-10) for i in vidx]
            total = sum(vp)
            vp = [p/total for p in vp]
            if temp > 0:
                vp = [math.log(p)/temp for p in vp]
                ex = [math.exp(p) for p in vp]
                s = sum(ex)
                vp = [p/s for p in ex]
            out.append(random.choices(valid, weights=vp)[0])
        return out

    def play_game_mcts(self, temperature=1.0, record=True, mcts_sims=50, c_puct=1.5, step_n=150):
        d0, d1 = self.get_random_deck(), self.get_random_deck()
        game = CRGame(d0, d1)
        traj = []

        from ai.mcts import mcts_search, encode_action, action_to_idx, ACTION_DIM
        import numpy as np

        for step in range(6000):
            if game.game_over:
                break

            policy_vecs = []
            actions = [(-1, 0, 0), (-1, 0, 0)]
            for pid in range(2):
                valid = game.get_valid_actions(pid)
                if not valid:
                    continue
                policy, act = mcts_search(
                    game=game, pid=pid, network=self.network,
                    game_clone_fn=lambda g: copy.deepcopy(g),
                    num_simulations=mcts_sims, c_puct=c_puct,
                    temperature=temperature, device='cpu',
                )
                actions[pid] = act
                if record:
                    traj.append({
                        'state': game.get_state_tensor(pid),
                        'pid': pid, 'action': act, 'valid': valid,
                        'policy': policy.tolist(),
                    })

            game.step({0: actions[0], 1: actions[1]})
            game.step_n(step_n)

        w = game.winner
        for t in traj:
            r = 0
            if w == t['pid']:
                r = 1
            elif w is not None:
                r = -1
            t['reward'] = r

        if record:
            self.replay_buffer.extend(traj)
            if len(self.replay_buffer) > self.max_buffer_size:
                self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]

        return {'winner': w, 'steps': step, 'decks': (d0, d1)}

    def play_game(self, use_network=True, temperature=1.0, record=True, step_n=150):
        d0, d1 = self.get_random_deck(), self.get_random_deck()
        game = CRGame(d0, d1)
        traj_raw = []

        for step in range(6000):
            if game.game_over: break
            if use_network and self.network:
                s0 = game.get_state_tensor(0)
                s1 = game.get_state_tensor(1)
                v0 = game.get_valid_actions(0)
                v1 = game.get_valid_actions(1)
                acts = self._batch_policy([s0, s1], [v0, v1], temperature)
                if record:
                    traj_raw.append({'state': s0, 'pid': 0, 'action': acts[0], 'valid': v0})
                    traj_raw.append({'state': s1, 'pid': 1, 'action': acts[1], 'valid': v1})
            else:
                acts = {}
                for pid in range(2):
                    valid = game.get_valid_actions(pid)
                    acts[pid] = random.choice(valid) if valid else (-1, 0, 0)
                    if record:
                        traj_raw.append({'state': game.get_state_tensor(pid), 'pid': pid,
                                         'action': acts[pid], 'valid': valid})
                acts = [acts[0], acts[1]]

            game.step({0: acts[0], 1: acts[1]})
            game.step_n(step_n)

        w = game.winner
        for t in traj_raw:
            r = 0
            if w == t['pid']: r = 1
            elif w is not None: r = -1
            t['reward'] = r

        if record:
            self.replay_buffer.extend(traj_raw)
            if len(self.replay_buffer) > self.max_buffer_size:
                self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]

        return {'winner': w, 'steps': step, 'decks': (d0, d1), 'trajectory': traj_raw}

    def play_game_ppo(self, use_network=True, temperature=1.0, step_n=150):
        """
        Play one game and return trajectories suitable for PPO.
        Returns (traj_p0, traj_p1) where each is a list of dicts with step-by-step data.
        Only the terminal step has non-zero reward and done=True.
        """
        d0, d1 = self.get_random_deck(), self.get_random_deck()
        game = CRGame(d0, d1)
        traj_p0 = []
        traj_p1 = []

        for step in range(6000):
            if game.game_over: break
            game_over_now = game.game_over
            if use_network and self.network:
                s0 = game.get_state_tensor(0)
                s1 = game.get_state_tensor(1)
                v0 = game.get_valid_actions(0)
                v1 = game.get_valid_actions(1)
                acts = self._batch_policy([s0, s1], [v0, v1], temperature)
                traj_p0.append({'state': s0, 'pid': 0, 'action': acts[0], 'valid': v0, 'reward': 0, 'done': False})
                traj_p1.append({'state': s1, 'pid': 1, 'action': acts[1], 'valid': v1, 'reward': 0, 'done': False})
            else:
                acts = {}
                for pid in range(2):
                    valid = game.get_valid_actions(pid)
                    acts[pid] = random.choice(valid) if valid else (-1, 0, 0)
                s0 = game.get_state_tensor(0)
                s1 = game.get_state_tensor(1)
                v0 = game.get_valid_actions(0)
                v1 = game.get_valid_actions(1)
                traj_p0.append({'state': s0, 'pid': 0, 'action': acts[0], 'valid': v0, 'reward': 0, 'done': False})
                traj_p1.append({'state': s1, 'pid': 1, 'action': acts[1], 'valid': v1, 'reward': 0, 'done': False})

            game.step({0: acts[0], 1: acts[1]})
            game.step_n(step_n)

        w = game.winner
        if traj_p0:
            last_p0 = traj_p0[-1]
            last_p1 = traj_p1[-1]
            if w == 0:
                last_p0['reward'] = 1
                last_p1['reward'] = -1
            elif w == 1:
                last_p0['reward'] = -1
                last_p1['reward'] = 1
            last_p0['done'] = True
            last_p1['done'] = True

        return {'winner': w, 'steps': step, 'decks': (d0, d1),
                'traj_p0': traj_p0, 'traj_p1': traj_p1}

    def sample_batch(self, n=256):
        batch = random.sample(self.replay_buffer, min(n, len(self.replay_buffer)))
        return [t['state'] for t in batch], [t['action'] for t in batch], [t['reward'] for t in batch]

    def get_win_rate(self, deck, n=10):
        wins = 0
        for _ in range(n):
            game = CRGame(deck, self.get_random_deck())
            for _ in range(6000):
                if game.game_over: break
                game.step({0: random.choice(game.get_valid_actions(0)) if game.get_valid_actions(0) else (-1,0,0),
                           1: random.choice(game.get_valid_actions(1)) if game.get_valid_actions(1) else (-1,0,0)})
                game.step_n(15)
            if game.winner == 0: wins += 1
        return wins / n
