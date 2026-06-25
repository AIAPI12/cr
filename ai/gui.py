import sys, os, math, random, time, json
_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import pygame

from ai.env import CRGame, SelfPlaySystem
from ai.network import Trainer, CRNetwork
import torch

pygame.init()

W, H = 1420, 920
ARENA_W, ARENA_H = 720, 640
AX, AY = 50, 70
TILE = 20
CARD_PANEL_Y = AY + ARENA_H + 20
CARD_START_X = 60

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,60,60)
RED_DARK = (160,30,30)
RED_TEAM = (255,70,70)
BLUE = (40,100,255)
BLUE_TEAM = (50,130,255)
BLUE_DARK = (20,60,200)
GREEN = (60,255,60)
GOLD = (255,210,50)
GRAY = (120,120,120)
LGRAY = (200,200,200)
DARK = (22,22,34)
DARKER = (14,14,24)
GRASS_GREEN = (30,80,20)
RIVER_BLUE = (40,100,140)
BRIDGE = (100,70,40)
HP_GREEN = (100,220,50)
HP_YELLOW = (240,200,30)
HP_RED = (240,50,50)
YELLOW = (255,255,100)
ELIXIR_PURPLE = (180,60,220)
ELIXIR_GLOW = (220,150,255)
COMMON = (160,160,160)
RARE = (255,210,50)
EPIC = (160,60,255)
LEGENDARY = (255,180,50)
CHAMPION = (255,60,180)

RARITY_COLORS = {'Common': COMMON, 'Rare': RARE, 'Epic': EPIC, 'Legendary': LEGENDARY, 'Champion': CHAMPION}

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
        self.trainer = Trainer(device=device, res_blocks=5, filters=48, dropout=0.0)
        ckpt = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'latest.pt')
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
        self.turbo = True
        self.fps_history = []
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
        pygame.draw.rect(self.screen, GRASS_GREEN, (AX, AY, ARENA_W, ARENA_H))
        ry = AY + 15 * TILE
        pygame.draw.rect(self.screen, RIVER_BLUE, (AX, ry, ARENA_W, TILE * 2 + 4))
        pygame.draw.rect(self.screen, (20,60,100), (AX, ry, ARENA_W, TILE * 2 + 4), 1)
        for bx in [2,3,4, 13,14,15]:
            px = AX + bx * TILE
            pygame.draw.rect(self.screen, BRIDGE, (px, ry, TILE, TILE * 2))

        kpos = [(9, 1.5), (9, 30.5)]
        for kx, ky in kpos:
            sx, sy = self.ws(kx, ky)
            pygame.draw.circle(self.screen, (55,55,70), (sx-2, sy+2), 36)
            pygame.draw.circle(self.screen, (80,80,100), (sx, sy), 36)
            pygame.draw.circle(self.screen, (60,60,80), (sx, sy), 34)
            pygame.draw.circle(self.screen, (120,120,140), (sx, sy), 36, 2)

        pt_pos = [(3.5, 5.5), (14.5, 5.5), (3.5, 26.5), (14.5, 26.5)]
        for px, py in pt_pos:
            sx, sy = self.ws(px, py)
            pygame.draw.circle(self.screen, (60,60,80), (sx, sy), 20)
            pygame.draw.circle(self.screen, (80,80,100), (sx, sy), 20)
            pygame.draw.circle(self.screen, (100,100,120), (sx, sy), 20, 1)

    def draw_hp_bar(self, sx, sy, ratio, w=26, h=4, yo=-16):
        if ratio <= 0: return
        bg = (60,0,0)
        pygame.draw.rect(self.screen, bg, (sx - w//2, sy + yo, w, h))
        rw = max(2, int(w * ratio))
        if ratio > 0.6:    c = (int(100 + 155*(1-ratio)/0.4), 220, 50)
        elif ratio > 0.3:  c = (255, int(220*(ratio-0.3)/0.3), 40)
        else:              c = (255, max(0, int(150*(ratio/0.3))), 40)
        pygame.draw.rect(self.screen, c, (sx - w//2, sy + yo, rw, h))

    def draw_elixir(self, x, y, val, maxv=10):
        bw, bh = 160, 18
        pygame.draw.rect(self.screen, (20,10,50), (x, y, bw, bh))
        fill = min(1.0, val / maxv)
        if fill > 0:
            c1 = (180, 60, 220) if fill < 0.5 else (200, 120, 255)
            c2 = (100, 30, 140) if fill < 0.5 else (120, 80, 180)
            for i in range(int(bw * fill)):
                mod = (i % 4 < 2)
                c = c1 if mod else c2
                pygame.draw.line(self.screen, c, (x+i, y+2), (x+i, y+bh-3))
        pygame.draw.rect(self.screen, (180,100,255), (x, y, bw, bh), 1)
        val_s = self.small.render(f"{val:.1f}", True, WHITE)
        self.screen.blit(val_s, (x + bw + 8, y))

    def draw_top_bar(self):
        if not self.game:
            return
        b = self.game.battle
        p0, p1 = b.players[0], b.players[1]
        mid = W // 2

        t0 = self.big.render(f"P0 {p0.get_crown_count()}/3", True, BLUE_TEAM)
        t1 = self.big.render(f"{p1.get_crown_count()}/3 P1", True, RED_TEAM)
        self.screen.blit(t0, (mid - 80 - t0.get_width(), 12))
        self.screen.blit(t1, (mid + 80, 12))

        ts = self.big.render(f"{b.time:.0f}s", True, WHITE)
        self.screen.blit(ts, (mid - ts.get_width()//2, 12))

        tt = getattr(self.game, 'tower_troop', ['?','?'])
        ti = self.tiny.render(f"Towers: {tt[0]} vs {tt[1]}", True, GRAY)
        self.screen.blit(ti, (mid - ti.get_width()//2, 44))

    def draw_card_hand(self):
        if not self.game:
            return
        p0 = self.game.battle.players[0]
        sx = CARD_START_X
        cx = sx
        for i, name in enumerate(p0.hand[:4]):
            cs = self.game.battle.card_loader.get_card(name)
            cost = cs.mana_cost if cs else 5
            rare = getattr(cs, 'rarity', 'Common') if cs else 'Common'
            aff = p0.elixir >= cost
            bw, bh = 64, 80
            x, y = cx, CARD_PANEL_Y

            rcol = RARITY_COLORS.get(rare, LGRAY)
            pygame.draw.rect(self.screen, (30,30,50), (x, y, bw, bh))
            pygame.draw.rect(self.screen, (40,40,65), (x+2, y+2, bw-4, bh-4))
            pygame.draw.rect(self.screen, rcol, (x, y, bw, bh), 2)

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

        p0h = self.tiny.render(f"P0 HP: {p0.king_tower_hp:.0f}", True, BLUE_TEAM)
        self.screen.blit(p0h, (elix_x, CARD_PANEL_Y + 55))

    def draw_entities(self):
        if not self.game:
            return
        b = self.game.battle
        for e in b.entities.values():
            if not e.is_alive:
                continue
            sx, sy = self.ws(e.position.x, e.position.y)
            cs = getattr(e, 'card_stats', None)
            name = getattr(cs, 'name', '') if cs else ''
            rare = getattr(cs, 'rarity', '') if cs else ''
            is_hero = (getattr(cs, 'has_evolution', False) or name.startswith('Hero') or rare == 'Champion')
            is_b = 'Building' in type(e).__name__
            is_tower = is_b and any(k in name for k in ['Tower','King','Princess','Cannon','Duchess'])
            team = BLUE_TEAM if e.player_id == 0 else RED_TEAM
            hp_r = e.hitpoints / max(e.max_hitpoints, 1)

            if is_tower:
                r = 32
                s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*team, 200), (r, r), r)
                pygame.draw.circle(s, WHITE, (r, r), r, 2)
                self.screen.blit(s, (sx-r, sy-r))
                ct = self.tiny.render(f"{int(e.hitpoints)}", True, WHITE)
                self.screen.blit(ct, (sx - ct.get_width()//2, sy - 36))
                cpos = [(sx-8, sy+6), (sx-5, sy-8), (sx, sy-2), (sx+5, sy-8), (sx+8, sy+6)]
                pygame.draw.polygon(self.screen, GOLD, cpos)
                self.draw_hp_bar(sx, sy, hp_r, 52, 6, -38)
                continue

            if is_b:
                r = max(10, int((getattr(e, 'collision_radius', 0.5) or 0.5) * TILE * 1.8))
                s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
                pygame.draw.rect(s, (*team, 200), (0, 0, r*2, r*2))
                pygame.draw.rect(s, WHITE, (0, 0, r*2, r*2), 1)
                self.screen.blit(s, (sx-r, sy-r))
                self.draw_hp_bar(sx, sy, hp_r, r*2, 5, -r-8)
                lbl = self.tiny.render(name[:5], True, WHITE)
                self.screen.blit(lbl, (sx - lbl.get_width()//2, sy + r + 2))
                continue

            r = max(6, int((getattr(e, 'collision_radius', 0.3) or 0.3) * TILE * 2.2))
            air = getattr(e, 'is_air_unit', False)
            s = pygame.Surface((r*2+4, r*2+4), pygame.SRCALPHA)

            if air:
                pts = [(r+2, 2), (r*2+2, r+2), (r+2, r*2+2), (2, r+2)]
                pygame.draw.polygon(s, (*team, 220), pts)
                pygame.draw.polygon(s, WHITE, pts, 1)
            else:
                pygame.draw.circle(s, (*team, 220), (r+2, r+2), r)
                pygame.draw.circle(s, WHITE, (r+2, r+2), r, 1)

            if is_hero and r >= 6:
                if air:
                    pygame.draw.polygon(s, GOLD, pts, 2)
                else:
                    pygame.draw.circle(s, GOLD, (r+2, r+2), r+2, 2)

            self.screen.blit(s, (sx-r-2, sy-r-2))
            self.draw_hp_bar(sx, sy, hp_r, r*2+6, 5, -r-10)
            lbl = self.tiny.render(name[:7], True, WHITE)
            self.screen.blit(lbl, (sx - lbl.get_width()//2, sy + r + 3))

    def draw_info_panel(self):
        if not self.game:
            return
        px = AX + ARENA_W + 30
        py = AY
        pw = W - px - 20

        pygame.draw.rect(self.screen, DARKER, (px, py, pw, 300))
        pygame.draw.rect(self.screen, GRAY, (px, py, pw, 300), 1)

        txts = [
            (self.small, f"Game {self.game_count}", LGRAY),
            (self.small, f"Steps: {self.step_count}", LGRAY),
            (self.small, f"Speed: {self.speed}x{' TURBO' if self.turbo else ''}", GOLD if self.turbo else LGRAY),
            (self.small, f"Buffer: {len(self.sp.replay_buffer)}", LGRAY),
            (self.small, "", LGRAY),
        ]

        for i, (fn, txt, col) in enumerate(txts):
            t = fn.render(txt, True, col)
            self.screen.blit(t, (px + 10, py + 10 + i * 20))

        leg = [
            "P0 Hand:", self.game.battle.players[0].hand[:4],
            "P1 Hand:", self.game.battle.players[1].hand[:4],
        ]
        y = py + 130
        for item in leg:
            if isinstance(item, str):
                t = self.tiny.render(item, True, LGRAY)
                self.screen.blit(t, (px + 10, y))
                y += 16
            else:
                t = self.tiny.render(str(item), True, WHITE)
                self.screen.blit(t, (px + 10, y))
                y += 14

        # Last actions
        y = py + 200
        for la in self.last_actions:
            if la:
                t = self.tiny.render(la, True, YELLOW)
                self.screen.blit(t, (px + 10, y))
                y += 14

    def draw_bottom_panel(self):
        if not self.game:
            return
        p1 = self.game.battle.players[1]
        ex = W - 220
        self.draw_elixir(ex, CARD_PANEL_Y + 30, p1.elixir)
        p1h = self.tiny.render(f"P1 HP: {p1.king_tower_hp:.0f}", True, RED_TEAM)
        self.screen.blit(p1h, (ex, CARD_PANEL_Y + 55))

        recent = self.recent_games[-40:]
        dots = ''.join(['O' if w == 0 else 'X' if w == 1 else '-' for w in recent])
        wl = self.small.render(f"Wins: P0={self.wins[0]} P1={self.wins[1]}", True, LGRAY)
        self.screen.blit(wl, (ex, CARD_PANEL_Y + 5))
        dr = self.small.render(f"[{dots}]", True, GRAY)
        self.screen.blit(dr, (ex, CARD_PANEL_Y + 75))

    def draw_controls(self):
        y = H - 50
        controls = [
            f"SPACE: pause  S/D: speed  F: turbo  1: AIvsAI/random",
            f"T: train {'ON' if self.training_enabled else 'OFF'}  R: restart  ESC: quit  Step {self.trainer.step_count}",
        ]
        if self.turbo:
            controls[1] += " [TURBO]"
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

        info = self.small.render(f"Step {self.trainer.step_count} | Buffer {len(self.sp.replay_buffer)} | MPS: {torch.backends.mps.is_available()}", True, GRAY)
        self.screen.blit(info, (W//2 - info.get_width()//2, 520))
        info2 = self.small.render(f"Decks: {len(self.sp.deck_pool)} | Cards: {len(self.game.battle.card_loader._cards) if self.game else '?'}", True, GRAY)
        self.screen.blit(info2, (W//2 - info2.get_width()//2, 540))

    def handle_events(self):
        if self.menu:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_1:
                        self.menu = False
                        self.training_enabled = False
                        self.new_game()
                    if event.key == pygame.K_2:
                        self.menu = False
                        self.training_enabled = True
                        self.new_game()
                    if event.key == pygame.K_3:
                        self.training_active = True
                        import threading
                        def train_thread():
                            from ai.train import run_selfplay_loop
                            run_selfplay_loop(50, turbo=True)
                            self.training_games += 50
                            self.training_active = False
                            self.menu = False
                            self.training_enabled = False
                            self.new_game()
                        threading.Thread(target=train_thread, daemon=True).start()
                    if event.key == pygame.K_4:
                        self.menu = False
                        self.training_enabled = False
                        self.new_game()
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.menu = True
                    self.training_active = False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_r:
                    self.new_game()
                    self.game_count += 1
                if event.key == pygame.K_s:
                    self.speed = min(60, self.speed + 1)
                if event.key == pygame.K_d:
                    self.speed = max(0, self.speed - 1)
                if event.key == pygame.K_t:
                    self.training_enabled = not self.training_enabled
                if event.key == pygame.K_1:
                    self.ai_vs_ai = not self.ai_vs_ai
                if event.key == pygame.K_f:
                    self.turbo = not self.turbo
                if event.key == pygame.K_0:
                    self.speed = 0
        return True

    def step_ai(self):
        if not self.game or self.game.game_over:
            if self.game and self.game.game_over:
                w = self.game.winner
                if w is not None:
                    self.wins[w] += 1
                self.game_count += 1
                self.recent_games.append(self.game.winner)
                if len(self.recent_games) > 50:
                    self.recent_games.pop(0)
                if self.training_enabled and len(self.sp.replay_buffer) >= self.batch_size:
                    states, actions, rewards = self.sp.sample_batch(self.batch_size)
                    self.trainer.train_step(states, actions, rewards)
                    self.trainer.save(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'latest.pt'))
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
                    if ci < 0:
                        i = 0
                    else:
                        i = 1 + ci * (18 * 15) + int(x - 0.5) * 15 + int(y // 4)
                    idxs.append(min(i, len(pol)-1))
                probs = [pol[i] for i in idxs]
                total = sum(probs)
                if total > 0:
                    probs = [p/total for p in probs]
                else:
                    probs = [1.0/len(probs)] * len(probs)
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
        self.game.step_n(30 if not self.turbo else 120)

        if self.training_enabled and random.random() < 0.05 and len(self.sp.replay_buffer) >= self.batch_size:
            states, actions, rewards = self.sp.sample_batch(self.batch_size)
            self.trainer.train_step(states, actions, rewards)

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
                if self.speed == 0:
                    pass
                elif self.speed == 1:
                    if tick_counter >= 2:
                        self.step_ai()
                        tick_counter = 0
                else:
                    skip = self.speed // 2
                    if self.turbo:
                        skip = max(skip, 20)
                    if tick_counter >= max(1, 3 - skip):
                        if self.turbo:
                            for _ in range(5):
                                self.step_ai()
                        else:
                            self.step_ai()
                        tick_counter = 0

            self.screen.fill(DARK)
            self.draw_top_bar()
            self.draw_arena()
            self.draw_entities()
            self.draw_card_hand()
            self.draw_bottom_panel()
            self.draw_info_panel()
            self.draw_controls()

            pygame.display.flip()

        pygame.quit()

if __name__ == '__main__':
    CRGUI().run()
