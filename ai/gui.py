import sys, os, math, random, time, json
_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import pygame

from ai.env import CRGame, SelfPlaySystem, EVOLVED_CARDS, CHAMPION_CARDS
from ai.network import Trainer, CRNetwork, ACTION_DIM
import torch

pygame.init()

W, H = 1420, 920
ARENA_W, ARENA_H = 720, 640
AX, AY = 50, 70
TILE = 20
CARD_PANEL_Y = AY + ARENA_H + 20
CARD_START_X = 60

BLACK = (0,0,0); WHITE = (255,255,255)
RED_TEAM = (255,70,70); BLUE_TEAM = (50,130,255)
GREEN = (60,255,60); GOLD = (255,210,50); GRAY = (120,120,120)
LGRAY = (200,200,200); DARK = (22,22,34); DARKER = (14,14,24)
YELLOW = (255,255,100); ELIXIR_PURPLE = (180,60,220); ELIXIR_GLOW = (220,150,255)
COMMON = (160,160,160); RARE = (255,210,50); EPIC = (160,60,255)
LEGENDARY = (255,180,50); CHAMPION = (255,60,180)
RARITY_COLORS = {'Common': COMMON, 'Rare': RARE, 'Epic': EPIC, 'Legendary': LEGENDARY, 'Champion': CHAMPION}

# Load card sprites (try 150px first, fall back to 75px)
CARD_MAP = {}
CARD_SPRITES = {}

# GUI uses flat filename map (no path prefixes)
_map_path = os.path.join(_PROJ, 'dashboard', 'static', 'cards', 'card_map_gui.json')
if not os.path.exists(_map_path):
    _map_path = os.path.join(_PROJ, 'dashboard', 'static', 'cards', 'card_map.json')

# Search dirs in priority order
_sprite_dirs = [
    os.path.join(_PROJ, 'dashboard', 'static', 'assets', 'cards-150'),
    os.path.join(_PROJ, 'dashboard', 'static', 'cards'),
]

if os.path.exists(_map_path):
    with open(_map_path) as f:
        CARD_MAP = json.load(f)
    for cname, fn in CARD_MAP.items():
        for _dir in _sprite_dirs:
            fp = os.path.join(_dir, fn)
            if os.path.exists(fp):
                try:
                    CARD_SPRITES[cname] = pygame.image.load(fp).convert_alpha()
                    break
                except Exception:
                    pass

# Map entity names (from card_stats) to card map keys for sprite lookup
ENTITY_TO_CARD_KEY = {
    'Skeleton': 'Skeletons',
    'Barbarian': 'Barbarians',
    'Goblin': 'Goblins',
    'Minion': 'Minions',
    'FireSpirit': 'FireSpirits',
    'SpearGoblin': 'SpearGoblins',
    'Bat': 'Bats',
    'Golemite': 'Golem',
    'LavaPup': 'LavaHound',
    'SkeletonDragons': 'SkeletonDragons',
    'RageBarbarian': 'RageBarbarian',
    'GoblinBrawler': 'GoblinCage',
    'Archer': 'Archer',
    'IceSpirit': 'IceSpirits',
    'ElixirGolem2': 'ElixirGolem',
    'SuperLavaHound2': 'SuperLavaHound',
    'SuperMiniPekkaPancakes': 'SuperMiniPekka',
    'SkeletonContainer': 'SkeletonBalloon',
    'BalloonBomb': 'Balloon',
    'GiantSkeletonBomb': 'GiantSkeleton',
    'RageBarbarianBottle': 'RageBarbarian',
    'BombTowerBomb': 'BombTower',
}


class CRGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Clash Royale AI")
        self.font = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 16)
        self.tiny = pygame.font.Font(None, 13)
        self.big = pygame.font.Font(None, 36)
        self.massive = pygame.font.Font(None, 48)
        self.clock = pygame.time.Clock()

        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        self.trainer = Trainer(device=device, res_blocks=5, filters=48, dropout=0.01)
        ckpt = os.path.join(_PROJ, 'checkpoints', 'latest.pt')
        if os.path.exists(ckpt):
            self.trainer.load(ckpt)

        self.sp = SelfPlaySystem(network=self.trainer.network)
        self.game = None
        self.speed = 2
        self.paused = False
        self.ai_vs_ai = True
        self.game_count = 0
        self.wins = [0, 0]
        self.step_count = 0
        self.last_actions = ['', '']
        self.training_enabled = False
        self.batch_size = 256
        self.recent_games = []
        self.turbo = False
        self.step_n = 60
        self.menu = True
        self.training_games = 0
        self.training_active = False
        self._start_auto_training()

    def _start_auto_training(self):
        def _train_bg():
            from ai.train import run_selfplay_loop
            while True:
                run_selfplay_loop(50, turbo=True, compile_model=False)
                self.training_games += 50
                self.training_active = True
                time.sleep(0.1)
        import threading
        threading.Thread(target=_train_bg, daemon=True).start()
        self.training_active = True

    def new_game(self):
        deck0 = self.sp.get_random_deck()
        deck1 = self.sp.get_random_deck()
        self.game = CRGame(deck0, deck1)
        self.step_count = 0
        self.last_actions = ['', '']

    def ws(self, wx, wy):
        return int(AX + wx * TILE), int(AY + (32 - wy) * TILE)

    def draw_arena(self):
        pygame.draw.rect(self.screen, (30,80,20), (AX, AY, ARENA_W, ARENA_H))
        ry = AY + 15 * TILE
        pygame.draw.rect(self.screen, (40,100,140), (AX, ry, ARENA_W, TILE * 2 + 4))
        for bx in [2,3,4, 13,14,15]:
            px = AX + bx * TILE
            pygame.draw.rect(self.screen, (100,70,40), (px, ry, TILE, TILE * 2))

    def draw_towers(self):
        kpos = [(9, 1.5), (9, 30.5)]
        for i, (kx, ky) in enumerate(kpos):
            sx, sy = self.ws(kx, ky)
            owner = 0 if i == 0 else 1
            col = BLUE_TEAM if owner == 0 else RED_TEAM
            pygame.draw.circle(self.screen, (55,55,70), (sx-2, sy+2), 36)
            s = pygame.Surface((72,72), pygame.SRCALPHA)
            c = (*col, 200)
            pygame.draw.circle(s, c, (36,36), 34)
            pygame.draw.circle(s, WHITE, (36,36), 34, 2)
            self.screen.blit(s, (sx-36, sy-36))
            ct = self.small.render("K", True, GOLD)
            self.screen.blit(ct, (sx-ct.get_width()//2, sy-ct.get_height()//2))
        pt_pos = [(3.5, 5.5), (14.5, 5.5), (3.5, 26.5), (14.5, 26.5)]
        for i, (px, py) in enumerate(pt_pos):
            sx, sy = self.ws(px, py)
            owner = 0 if i < 2 else 1
            col = BLUE_TEAM if owner == 0 else RED_TEAM
            s = pygame.Surface((40,40), pygame.SRCALPHA)
            c = (*col, 180)
            pygame.draw.circle(s, c, (20,20), 18)
            pygame.draw.circle(s, WHITE, (20,20), 18, 1)
            self.screen.blit(s, (sx-20, sy-20))

    def draw_hp_bar(self, sx, sy, ratio, w=26, h=4, yo=-16):
        if ratio <= 0: return
        pygame.draw.rect(self.screen, (60,0,0), (sx - w//2, sy + yo, w, h))
        rw = max(2, int(w * ratio))
        if ratio > 0.6: c = (100+int(155*(1-ratio)/0.4), 220, 50)
        elif ratio > 0.3: c = (255, int(220*(ratio-0.3)/0.3), 40)
        else: c = (255, max(0, int(150*(ratio/0.3))), 40)
        pygame.draw.rect(self.screen, c, (sx - w//2, sy + yo, rw, h))

    def draw_elixir(self, x, y, val, maxv=10):
        bw, bh = 160, 18
        pygame.draw.rect(self.screen, (20,10,50), (x, y, bw, bh))
        fill = min(1.0, val / maxv)
        if fill > 0:
            for i in range(int(bw * fill)):
                mod = i % 4 < 2
                c = (180, 60, 220) if fill < 0.5 else (200, 120, 255)
                if not mod: c = (100, 30, 140) if fill < 0.5 else (120, 80, 180)
                pygame.draw.line(self.screen, c, (x+i, y+2), (x+i, y+bh-3))
        pygame.draw.rect(self.screen, (180,100,255), (x, y, bw, bh), 1)

    def draw_top_bar(self):
        if not self.game: return
        b = self.game.battle
        p0, p1 = b.players[0], b.players[1]
        mid = W // 2
        t0 = self.big.render(f"P0 {p0.get_crown_count()}/3", True, BLUE_TEAM)
        t1 = self.big.render(f"{p1.get_crown_count()}/3 P1", True, RED_TEAM)
        self.screen.blit(t0, (mid - 80 - t0.get_width(), 12))
        self.screen.blit(t1, (mid + 80, 12))
        ts = self.big.render(f"{b.time:.0f}s", True, WHITE)
        self.screen.blit(ts, (mid - ts.get_width()//2, 12))

    def draw_card_hand(self):
        if not self.game: return
        p0 = self.game.battle.players[0]
        cx = CARD_START_X
        for i, name in enumerate(p0.hand[:4]):
            cs = self.game.battle.card_loader.get_card(name)
            cost = cs.mana_cost if cs else 5
            rare = getattr(cs, 'rarity', 'Common') if cs else 'Common'
            aff = p0.elixir >= cost
            bw, bh = 64, 80
            x, y = cx, CARD_PANEL_Y
            rcol = RARITY_COLORS.get(rare, LGRAY)
            is_evo = name in EVOLVED_CARDS
            is_champ = name in CHAMPION_CARDS
            evo_col = (150, 50, 255) if is_evo else rcol
            final_col = (255, 180, 50) if is_champ else evo_col

            pygame.draw.rect(self.screen, (30,30,50), (x, y, bw, bh))
            pygame.draw.rect(self.screen, (40,40,65), (x+2, y+2, bw-4, bh-4))

            sprite = CARD_SPRITES.get(name)
            if sprite:
                s = pygame.transform.scale(sprite, (bw-4, bh-4))
                self.screen.blit(s, (x+2, y+2))
            else:
                n = self.tiny.render(name[:8], True, WHITE)
                self.screen.blit(n, (x + bw//2 - n.get_width()//2, y + bh//2 - 8))

            pygame.draw.rect(self.screen, final_col, (x, y, bw, bh), 2)
            if is_evo:
                pygame.draw.rect(self.screen, (200, 100, 255), (x+2, y+2, bw-4, 4))

            pygame.draw.circle(self.screen, ELIXIR_PURPLE, (x+14, y+14), 11)
            pygame.draw.circle(self.screen, ELIXIR_GLOW, (x+14, y+14), 10)
            ct = self.small.render(str(cost), True, WHITE)
            self.screen.blit(ct, (x+14-ct.get_width()//2, y+10))

            n = self.tiny.render(name[:8], True, WHITE)
            self.screen.blit(n, (x + bw//2 - n.get_width()//2, y + bh - 16))

            if not aff:
                overlay = pygame.Surface((bw, bh), pygame.SRCALPHA)
                overlay.fill((0,0,0,140))
                self.screen.blit(overlay, (x, y))
            cx += 68

        elix_x = cx + 20
        self.draw_elixir(elix_x, CARD_PANEL_Y + 30, p0.elixir)

    def _get_sprite_name(self, entity_name):
        if entity_name in CARD_SPRITES:
            return entity_name
        mapped = ENTITY_TO_CARD_KEY.get(entity_name)
        if mapped and mapped in CARD_SPRITES:
            return mapped
        return None

    def draw_entities(self):
        if not self.game: return
        b = self.game.battle
        for e in list(b.entities.values()):
            if not e.is_alive: continue
            try:
                sx, sy = self.ws(e.position.x, e.position.y)
            except Exception:
                continue
            cs = getattr(e, 'card_stats', None)
            raw_name = getattr(cs, 'name', '') if cs else ''
            sprite_key = self._get_sprite_name(raw_name)
            is_b = 'Building' in type(e).__name__
            is_tower = is_b and any(k in raw_name for k in ['Tower','King','Princess','Cannon','Duchess'])
            team = BLUE_TEAM if e.player_id == 0 else RED_TEAM
            hp_r = e.hitpoints / max(e.max_hitpoints, 1)
            col_r = getattr(e, 'collision_radius', None)
            if col_r is None and cs is not None:
                col_r = getattr(cs, 'collision_radius', 0.3)
            r = max(6, int((col_r or 0.3) * TILE * 2.2))
            s = pygame.Surface((r*2+4, r*2+4), pygame.SRCALPHA)

            if is_tower:
                self._draw_tower_entity(sx, sy, team, hp_r, r, raw_name)
                continue
            if is_b:
                self._draw_building(sx, sy, raw_name, team, hp_r, r)
                continue

            sprite = CARD_SPRITES.get(sprite_key) if sprite_key else None
            if sprite:
                s_size = r * 2 + 4
                scaled = pygame.transform.scale(sprite, (s_size, s_size))
                if e.player_id == 1:
                    tint = pygame.Surface((s_size, s_size), pygame.SRCALPHA)
                    tint.fill((255, 0, 0, 60))
                    scaled.blit(tint, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
                self.screen.blit(scaled, (sx - s_size//2, sy - s_size//2))
            else:
                air = getattr(e, 'is_air_unit', False)
                if air:
                    pts = [(r+2, 2), (r*2+2, r+2), (r+2, r*2+2), (2, r+2)]
                    pygame.draw.polygon(s, (*team, 220), pts)
                    pygame.draw.polygon(s, WHITE, pts, 1)
                else:
                    pygame.draw.circle(s, (*team, 220), (r+2, r+2), r)
                    pygame.draw.circle(s, WHITE, (r+2, r+2), r, 1)

            is_evo = raw_name in EVOLVED_CARDS or sprite_key in EVOLVED_CARDS
            is_champ = raw_name in CHAMPION_CARDS or sprite_key in CHAMPION_CARDS
            if is_champ:
                pygame.draw.circle(s, (255,180,50), (r+2, r+2), r+3, 2)
            elif is_evo:
                pygame.draw.circle(s, (200,100,255), (r+2, r+2), r+3, 2)

            if not sprite:
                self.screen.blit(s, (sx-r-2, sy-r-2))

            self.draw_hp_bar(sx, sy, hp_r, r*2+4, 5, -r-10)
            lbl = self.tiny.render(raw_name[:7], True, WHITE)
            self.screen.blit(lbl, (sx - lbl.get_width()//2, sy + r + 3))

    def _draw_tower_entity(self, sx, sy, team, hp_r, r, name):
        r_size = 32
        s = pygame.Surface((r_size*2, r_size*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*team, 200), (r_size, r_size), r_size)
        pygame.draw.circle(s, WHITE, (r_size, r_size), r_size, 2)
        self.screen.blit(s, (sx-r_size, sy-r_size))
        ct = self.tiny.render(f"{int(hp_r*100)}%", True, WHITE)
        self.screen.blit(ct, (sx - ct.get_width()//2, sy - 36))
        cpos = [(sx-8, sy+6), (sx-5, sy-8), (sx, sy-2), (sx+5, sy-8), (sx+8, sy+6)]
        pygame.draw.polygon(self.screen, GOLD, cpos)
        self.draw_hp_bar(sx, sy, hp_r, 52, 6, -38)

    def _draw_building(self, sx, sy, name, team, hp_r, r):
        r_size = max(10, int(r * 1.5))
        s = pygame.Surface((r_size*2, r_size*2), pygame.SRCALPHA)
        sprite = CARD_SPRITES.get(name)
        if sprite:
            scaled = pygame.transform.scale(sprite, (r_size*2, r_size*2))
            s.blit(scaled, (0,0))
        else:
            pygame.draw.rect(s, (*team, 200), (0, 0, r_size*2, r_size*2))
            pygame.draw.rect(s, WHITE, (0, 0, r_size*2, r_size*2), 1)
        self.screen.blit(s, (sx-r_size, sy-r_size))
        self.draw_hp_bar(sx, sy, hp_r, r_size*2, 5, -r_size-8)
        lbl = self.tiny.render(name[:5], True, WHITE)
        self.screen.blit(lbl, (sx - lbl.get_width()//2, sy + r_size + 2))

    def draw_info_panel(self):
        if not self.game: return
        px = AX + ARENA_W + 30
        py = AY
        pw = W - px - 20
        pygame.draw.rect(self.screen, DARKER, (px, py, pw, 300))
        pygame.draw.rect(self.screen, GRAY, (px, py, pw, 300), 1)
        txts = [
            (self.small, f"Game {self.game_count}", LGRAY),
            (self.small, f"Steps: {self.step_count}", LGRAY),
            (self.small, f"Speed: {self.speed}x timeScale: {self.step_n}", YELLOW if self.step_n > 50 else LGRAY),
            (self.small, f"Buffer: {len(self.sp.replay_buffer)}", LGRAY),
        ]
        for i, (fn, txt, col) in enumerate(txts):
            t = fn.render(txt, True, col)
            self.screen.blit(t, (px + 10, py + 10 + i * 20))

        y = py + 130
        for pid in range(2):
            t = self.tiny.render(f"P{pid} Hand:", True, LGRAY)
            self.screen.blit(t, (px + 10, y)); y += 16
            t = self.tiny.render(str(self.game.battle.players[pid].hand[:4]), True, WHITE)
            self.screen.blit(t, (px + 10, y)); y += 14

        y = py + 200
        for la in self.last_actions:
            if la:
                t = self.tiny.render(la, True, YELLOW)
                self.screen.blit(t, (px + 10, y)); y += 14

    def draw_bottom_panel(self):
        if not self.game: return
        p1 = self.game.battle.players[1]
        ex = W - 220
        self.draw_elixir(ex, CARD_PANEL_Y + 30, p1.elixir)
        recent = self.recent_games[-40:]
        dots = ''.join(['O' if w == 0 else 'X' if w == 1 else '-' for w in recent])
        wl = self.small.render(f"Wins: P0={self.wins[0]} P1={self.wins[1]}", True, LGRAY)
        self.screen.blit(wl, (ex, CARD_PANEL_Y + 5))
        dr = self.small.render(f"[{dots}]", True, GRAY)
        self.screen.blit(dr, (ex, CARD_PANEL_Y + 75))

    def draw_controls(self):
        y = H - 50
        controls = [
            f"SPACE: pause  S/D: speed  UP/DOWN: timeScale  1: AIvsAI/random",
            f"T: train  R: restart  ESC: quit  Step {self.trainer.step_count}  timeScale: {self.step_n}",
        ]
        for i, c in enumerate(controls):
            t = self.tiny.render(c, True, GRAY)
            self.screen.blit(t, (AX, y + i * 16))

    def draw_menu(self):
        self.screen.fill(DARK)
        title = self.massive.render("CLASH ROYALE AI", True, GOLD)
        self.screen.blit(title, (W//2 - title.get_width()//2, 120))
        subtitle = self.font.render("AlphaZero Self-Play Training", True, LGRAY)
        self.screen.blit(subtitle, (W//2 - subtitle.get_width()//2, 170))
        if self.training_active:
            st = self.big.render(f"Training... {self.training_games} games", True, WHITE)
            self.screen.blit(st, (W//2 - st.get_width()//2, 250))
        else:
            options = [
                (1, "Watch AI vs AI"),
                (2, "Train & Watch"),
                (3, "Turbo Train (no render)"),
                (4, "Eval AI vs Random"),
            ]
            for key, label in options:
                y = 260 + key * 50
                c = LGRAY
                if key == 1: c = BLUE_TEAM
                elif key == 2: c = GOLD
                elif key == 3: c = (255,100,100)
                elif key == 4: c = GREEN
                t = self.big.render(f"[{key}]  {label}", True, c)
                self.screen.blit(t, (W//2 - t.get_width()//2, y))
        info = self.small.render(f"Step {self.trainer.step_count} | Cards: {len(CARD_SPRITES)} | Sprites loaded", True, GRAY)
        self.screen.blit(info, (W//2 - info.get_width()//2, 520))

    def handle_events(self):
        if self.menu:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: return False
                    if event.key == pygame.K_1:
                        self.menu = False; self.training_enabled = False; self.new_game()
                    if event.key == pygame.K_2:
                        self.menu = False; self.training_enabled = True; self.new_game()
                    if event.key == pygame.K_3:
                        self.training_active = True
                        import threading
                        def train_thread():
                            from ai.train import run_selfplay_loop
                            run_selfplay_loop(50, turbo=True)
                            self.training_games += 50
                            self.training_active = False
                            self.menu = False; self.training_enabled = False; self.new_game()
                        threading.Thread(target=train_thread, daemon=True).start()
                    if event.key == pygame.K_4:
                        self.menu = False; self.training_enabled = False; self.new_game()
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: self.menu = True; self.training_active = False
                if event.key == pygame.K_SPACE: self.paused = not self.paused
                if event.key == pygame.K_r: self.new_game(); self.game_count += 1
                if event.key == pygame.K_s: self.speed = min(60, self.speed + 1)
                if event.key == pygame.K_d: self.speed = max(0, self.speed - 1)
                if event.key == pygame.K_t: self.training_enabled = not self.training_enabled
                if event.key == pygame.K_1: self.ai_vs_ai = not self.ai_vs_ai
                if event.key == pygame.K_UP: self.step_n = min(300, self.step_n + 10)
                if event.key == pygame.K_DOWN: self.step_n = max(10, self.step_n - 10)
                if event.key == pygame.K_0: self.speed = 0
        return True

    def step_ai(self):
        if not self.game or self.game.game_over:
            if self.game and self.game.game_over:
                w = self.game.winner
                if w is not None: self.wins[w] += 1
                self.game_count += 1
                self.recent_games.append(self.game.winner)
                if len(self.recent_games) > 50: self.recent_games.pop(0)
                if self.training_enabled and len(self.sp.replay_buffer) >= self.batch_size:
                    states, actions, rewards = self.sp.sample_batch(self.batch_size)
                    self.trainer.train_step(states, actions, rewards)
                    self.trainer.save(os.path.join(_PROJ, 'checkpoints', 'latest.pt'))
            self.new_game()
            return

        self.step_count += 1
        actions = {}
        for pid in range(2):
            valid = self.game.get_valid_actions(pid)
            if not valid:
                actions[pid] = (-1, 0, 0)
                continue
            if self.ai_vs_ai or self.training_enabled:
                state = self.game.get_state_tensor(pid)
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.trainer.device)
                with torch.no_grad():
                    logits, _ = self.trainer.network(state_t)
                pol = torch.exp(logits).squeeze(0).cpu().numpy()
                idxs = []
                for a in valid:
                    ci, x, y = a
                    if ci < 0: i = 0
                    else: i = 1 + ci * (18 * 15) + int(x - 0.5) * 15 + int(y // 4)
                    idxs.append(min(i, ACTION_DIM - 1))
                probs = [pol[i] for i in idxs]
                total = sum(probs)
                if total > 0: probs = [p/total for p in probs]
                else: probs = [1.0/len(probs)] * len(probs)
                idx = random.choices(range(len(valid)), weights=probs)[0]
                action = valid[idx]
            else:
                action = random.choice(valid)
            actions[pid] = action
            if action[0] >= 0:
                cname = self.game.battle.players[pid].hand[action[0]]
                self.last_actions[pid] = f"P{pid}: {cname} ({action[1]:.0f},{action[2]:.0f})"
            else:
                self.last_actions[pid] = ""
        self.game.step(actions)
        self.game.step_n(self.step_n)

    def run(self):
        running = True
        tick_counter = 0
        while running:
            running = self.handle_events()
            if self.menu:
                self.draw_menu()
                pygame.display.flip()
                self.clock.tick(30)
                continue

            dt = self.clock.tick(60)
            if not self.paused:
                tick_counter += 1
                if self.speed >= 1:
                    skip = self.speed // 2
                    if tick_counter >= max(1, 3 - skip):
                        if self.turbo:
                            for _ in range(5): self.step_ai()
                        else:
                            self.step_ai()
                        tick_counter = 0

            self.screen.fill(DARK)
            self.draw_top_bar()
            self.draw_arena()
            self.draw_towers()
            self.draw_entities()
            self.draw_card_hand()
            self.draw_bottom_panel()
            self.draw_info_panel()
            self.draw_controls()
            pygame.display.flip()
        pygame.quit()

if __name__ == '__main__':
    CRGUI().run()
