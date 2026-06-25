import asyncio, json, os, sys, time, math, random
from typing import Dict, List, Any, Optional
from collections import deque
from contextlib import asynccontextmanager

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
_SRC = os.path.join(_PROJ, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ai.env import CRGame, SelfPlaySystem

STATE_FILE = "/tmp/cr_dashboard_state.json"


class TrainingMetrics(BaseModel):
    game: int
    total_games: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    gps: float
    pct: float
    eta_seconds: float
    elapsed_seconds: float
    timestamp: float = Field(default_factory=time.time)


class GameResult(BaseModel):
    game: int
    winner: int
    steps: int
    decks: tuple
    p0_steps: int
    p1_steps: int
    timestamp: float = Field(default_factory=time.time)


class SystemInfo(BaseModel):
    cpu_percent: float
    memory_percent: float
    gpu_memory: Optional[float] = None
    gpu_util: Optional[float] = None


class DashboardState:
    def __init__(self):
        self.metrics: List[TrainingMetrics] = []
        self.games: List[GameResult] = []
        self.current_game: Optional[GameResult] = None
        self.session_start: float = time.time()
        self.total_games_target: int = 100000
        self.step_n: int = 150
        self.config: Dict[str, Any] = {}
        self.system_info: SystemInfo = SystemInfo(cpu_percent=0, memory_percent=0)
        self.live_feed: deque = deque(maxlen=500)
        self.connected_clients: set = set()
        self.win_rates: Dict[int, float] = {0: 0, 1: 0, -1: 0}
        self.elo_ratings: Dict[str, float] = {}
        self.game: Optional[CRGame] = None
        self.playing: bool = False
        self.speed: int = 1
        self.game_count: int = 0
        self._game_task: Optional[asyncio.Task] = None
        self._load()
        self._load_network()

    def _load_network(self):
        ckpt = os.path.join(_PROJ, 'checkpoints', 'latest.pt')
        if os.path.exists(ckpt):
            try:
                from ai.network import Trainer
                self._trainer = Trainer(device='cpu', res_blocks=7, filters=64)
                self._trainer.load(ckpt)
                self._sp = SelfPlaySystem(network=self._trainer.network)
                self.add_live(f"Loaded checkpoint ({self._trainer.step_count} steps)", "info")
            except Exception as e:
                self.add_live(f"Could not load checkpoint: {e}", "warn")
                self._trainer = None
                self._sp = SelfPlaySystem()
        else:
            self._trainer = None
            self._sp = SelfPlaySystem()

    def _load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    data = json.load(f)
                self.metrics = [TrainingMetrics(**m) for m in data.get('metrics', [])]
                self.games = [GameResult(**g) for g in data.get('games', [])]
                self.session_start = data.get('session_start', time.time())
                self.total_games_target = data.get('total_games_target', 100000)
                self.step_n = data.get('step_n', 150)
                self.config = data.get('config', {})
                self.elo_ratings = data.get('elo_ratings', {})
            except Exception as e:
                print(f"Failed to load state: {e}")

    def save(self):
        try:
            data = {
                'metrics': [m.model_dump() for m in self.metrics],
                'games': [g.model_dump() for g in self.games],
                'session_start': self.session_start,
                'total_games_target': self.total_games_target,
                'step_n': self.step_n,
                'config': self.config,
                'elo_ratings': self.elo_ratings,
            }
            with open(STATE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Failed to save state: {e}")

    async def broadcast(self, event: str, data: Any):
        msg = json.dumps({"event": event, "data": data})
        dead = set()
        for ws in self.connected_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.connected_clients -= dead

    def add_live(self, msg: str, level: str = "info"):
        entry = {"msg": msg, "level": level, "time": time.time()}
        self.live_feed.append(entry)
        try:
            asyncio.get_running_loop()
            asyncio.create_task(self.broadcast("live", entry))
        except RuntimeError:
            pass


state = DashboardState()


def _game_to_state(game) -> dict:
    if not game or not game.battle:
        return {}
    b = game.battle
    ents = []
    tower_ids = set()
    for pid in range(2):
        p = b.players[pid]
        for t in (p.king_tower, p.left_tower, p.right_tower):
            if t:
                tower_ids.add(id(t))
    for e in b.entities.values():
        if not e.is_alive:
            continue
        cs = getattr(e, 'card_stats', None)
        name = getattr(cs, 'name', '') if cs else ''
        is_b = 'Building' in type(e).__name__
        is_tower = id(e) in tower_ids
        r = getattr(e, 'collision_radius', 0.5) or 0.5
        ents.append({
            'id': id(e) % 65536,
            'x': e.position.x, 'y': e.position.y,
            'player_id': e.player_id,
            'name': name,
            'hp_ratio': e.hitpoints / max(e.max_hitpoints, 1),
            'radius': r * 20,
            'is_building': is_b,
            'is_tower': is_tower,
            'is_air': getattr(e, 'is_air_unit', False),
        })

    towers = []
    for pid in range(2):
        p = b.players[pid]
        towers.append({'player_id': pid, 'type': 'king', 'hp_ratio': p.king_tower_hp / max(p.king_tower_max_hp, 1)})
        towers.append({'player_id': pid, 'type': 'left', 'hp_ratio': p.left_tower_hp / max(p.left_tower_max_hp, 1)})
        towers.append({'player_id': pid, 'type': 'right', 'hp_ratio': p.right_tower_hp / max(p.right_tower_max_hp, 1)})

    players = []
    for pid in range(2):
        p = b.players[pid]
        hand = []
        for name in p.hand[:4]:
            cs = b.card_loader.get_card(name)
            hand.append({'name': name, 'cost': cs.mana_cost if cs else 5})
        players.append({
            'elixir': p.elixir,
            'crowns': p.get_crown_count(),
            'hand': hand,
        })

    return {
        'time': b.time,
        'entities': ents,
        'towers': towers,
        'players': players,
        'projectiles': [],
        'effects': [],
        'game_over': b.game_over,
        'winner': b.winner,
    }


def _ai_action(game, pid):
    valid = game.get_valid_actions(pid)
    if not valid:
        return (-1, 0, 0)
    if state._trainer and state._trainer.network is not None:
        try:
            import torch
            st = game.get_state_tensor(pid)
            st_t = torch.FloatTensor(st).unsqueeze(0)
            with torch.no_grad():
                logits, _ = state._trainer.network(st_t)
            pol = torch.exp(logits).squeeze(0).cpu().numpy()
            idxs = []
            for a in valid:
                ci, x, y = a
                if ci < 0:
                    i = 0
                else:
                    i = 1 + ci * (18 * 15) + int(x - 0.5) * 15 + int(y // 4)
                idxs.append(min(i, len(pol) - 1))
            probs = [max(pol[i], 1e-10) for i in idxs]
            total = sum(probs)
            probs = [p / total for p in probs]
            return random.choices(valid, weights=probs)[0]
        except Exception:
            pass
    return random.choice(valid)


async def _run_game_loop():
    while state.playing:
        if state.game is None or state.game.game_over:
            d0, d1 = state._sp.get_random_deck(), state._sp.get_random_deck()
            state.game = CRGame(d0, d1)
            state.game_count += 1
            state.add_live(f"Game {state.game_count}: {d0[0]}.. vs {d1[0]}..", "info")

        for _ in range(state.speed):
            if state.game.game_over:
                w = state.game.winner
                state.add_live(f"Game {state.game_count} over - winner: {'Draw' if w is None else f'P{w}'}", "info")
                await state.broadcast("game_state", _game_to_state(state.game))
                await state.broadcast("metrics", {
                    'gps': 0, 'loss': 0, 'win_rate': 0,
                    'buffer': len(state._sp.replay_buffer), 'game': state.game_count,
                })
                break
            actions = {0: _ai_action(state.game, 0), 1: _ai_action(state.game, 1)}
            state.game.step(actions)
            state.game.step_n(30)

        await state.broadcast("game_state", _game_to_state(state.game))
        await asyncio.sleep(0.033)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    state.playing = False
    if state._game_task:
        state._game_task.cancel()
    state.save()


app = FastAPI(title="Clash Royale AI Dashboard", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ConnectionManager:
    async def connect(self, ws: WebSocket):
        await ws.accept()
        state.connected_clients.add(ws)
        gs = _game_to_state(state.game) if state.game else None
        await ws.send_text(json.dumps({
            "event": "init",
            "data": {
                "metrics": [m.model_dump() for m in state.metrics[-100:]],
                "games": [g.model_dump() for g in state.games[-100:]],
                "config": state.config,
                "session_start": state.session_start,
                "total_games_target": state.total_games_target,
                "live_feed": list(state.live_feed)[-50:],
                "win_rates": state.win_rates,
                "game_state": gs,
            }
        }))

    def disconnect(self, ws: WebSocket):
        state.connected_clients.discard(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.get("/")
async def root():
    return FileResponse("dashboard/index.html")


@app.get("/api/state")
async def get_state():
    return {
        "metrics": [m.model_dump() for m in state.metrics[-100:]],
        "games": [g.model_dump() for g in state.games[-100:]],
        "config": state.config,
        "session_start": state.session_start,
        "live_feed": list(state.live_feed)[-50:],
        "win_rates": state.win_rates,
        "game_state": _game_to_state(state.game) if state.game else None,
    }


@app.post("/api/metrics")
async def receive_metrics(m: TrainingMetrics):
    state.metrics.append(m)
    if state.games:
        wins = sum(1 for g in state.games if g.winner == 0)
        losses = sum(1 for g in state.games if g.winner == 1)
        total = len(state.games)
        state.win_rates = {0: wins / total, 1: losses / total, -1: (total - wins - losses) / total}
    state.save()
    await state.broadcast("metrics", m.model_dump())
    return {"ok": True}


@app.post("/api/game")
async def receive_game(g: GameResult):
    state.games.append(g)
    state.current_game = g
    state.save()
    await state.broadcast("game", g.model_dump())
    return {"ok": True}


@app.get("/api/live")
async def receive_live(msg: str, level: str = "info"):
    state.add_live(msg, level)
    return {"ok": True}


@app.post("/api/play")
async def play_game():
    if state.playing:
        return {"ok": False, "msg": "Already playing"}
    state.playing = True
    state._game_task = asyncio.create_task(_run_game_loop())
    state.add_live("Game simulation started", "info")
    return {"ok": True}


@app.post("/api/stop")
async def stop_game():
    state.playing = False
    if state._game_task:
        state._game_task.cancel()
        state._game_task = None
    state.add_live("Game simulation stopped", "warn")
    return {"ok": True}


@app.post("/api/speed")
async def set_speed(data: Dict[str, int]):
    state.speed = max(1, data.get("speed", 1))
    return {"ok": True, "speed": state.speed}


@app.get("/health")
async def health():
    return {"status": "ok", "uptime": time.time() - state.session_start}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
