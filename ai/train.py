import sys, os, time, json, random, math, threading
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai.env import CRGame, SelfPlaySystem
from ai.network import Trainer, CRNetwork
import torch

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0: return f'{h}h{m:02d}m'
    if m > 0: return f'{m}m{s:02d}s'
    return f'{s}s'


def run_selfplay_loop(num_games=100, batch_size=64, save_every=25, turbo=True,
                      lr=3e-4, res_blocks=5, filters=48, dropout=0.0, compile_model=False,
                      device='cpu', use_mcts=False, mcts_sims=50, eval_every=0):
    print(f'[Train] Device: {device}  Turbo: {turbo}  MCTS: {use_mcts} sims={mcts_sims}')
    print(f'[Train] lr={lr} blocks={res_blocks} filters={filters} dropout={dropout} compile={compile_model}')

    trainer = Trainer(lr=lr, device=device, res_blocks=res_blocks, filters=filters, dropout=dropout, compile_model=compile_model)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    if os.path.exists(checkpoint_path):
        trainer.load(checkpoint_path)
        print(f'[Train] Loaded checkpoint (step {trainer.step_count})')

    sp = SelfPlaySystem(network=trainer.network)
    start = time.time()
    losses = []
    checkpoint_drive_path = os.path.join(os.path.dirname(CHECKPOINT_DIR), '..', 'colab_drive_sync', 'latest.pt')

    for game_idx in range(num_games):
        temp = 1.0 if game_idx < min(500, num_games // 20) else 0.5

        if use_mcts:
            sp.play_game_mcts(temperature=temp, record=True, mcts_sims=mcts_sims)
        else:
            sp.play_game(temperature=temp, record=True)

        # Training steps
        if len(sp.replay_buffer) >= batch_size:
            for _ in range(3 if turbo else 1):
                batch = random.sample(sp.replay_buffer, batch_size)
                states = [t['state'] for t in batch]
                actions = [t['action'] for t in batch]
                rewards = [t['reward'] for t in batch]
                policy_targets = [t.get('policy') for t in batch] if use_mcts and 'policy' in batch[0] else None
                metrics = trainer.train_step(states, actions, rewards, policy_targets=policy_targets)
                losses.append(metrics['loss'])

        # Save checkpoint
        if (game_idx + 1) % save_every == 0:
            trainer.save(checkpoint_path)

        # Periodic eval
        if eval_every > 0 and (game_idx + 1) % eval_every == 0:
            wr = evaluate_vs_baseline(trainer, num_games=min(50, max(10, eval_every // 10)))
            trainer.save(checkpoint_path)

        # ETA every 50 games (but not every line during fast runs)
        elapsed = time.time() - start
        gps = (game_idx + 1) / max(elapsed, 0.01)
        avg_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1) if losses else 0
        remaining = (num_games - game_idx - 1) / max(gps, 0.001)
        pct = (game_idx + 1) / num_games * 100

        if (game_idx + 1) % max(1, min(50, num_games // 100)) == 0 or game_idx == num_games - 1:
            print(f'[{game_idx+1:6d}/{num_games}] {pct:5.1f}% L={avg_loss:.3f} '
                  f'B={len(sp.replay_buffer):5d} {gps:.1f}g/s '
                  f'ETA={format_time(remaining)} Tot={format_time(elapsed)}')

    trainer.save(checkpoint_path)
    elapsed = time.time() - start
    print(f'[Done] {num_games} games in {elapsed:.1f}s ({num_games/elapsed:.1f} games/s)')
    return trainer


def evaluate_vs_baseline(trainer, num_games=20):
    sp = SelfPlaySystem(network=trainer.network)
    wins = 0
    eval_start = time.time()
    for i in range(num_games):
        deck0 = sp.get_random_deck()
        deck1 = sp.get_random_deck()
        game = CRGame(deck0, deck1)
        for _ in range(9000):
            if game.game_over:
                break
            actions = {}
            for pid in range(2):
                valid = game.get_valid_actions(pid)
                if pid == 0:
                    state = game.get_state_tensor(0)
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
                    with torch.no_grad():
                        policy_logits, _ = trainer.network(state_t)
                    policy = torch.exp(policy_logits).squeeze(0).cpu().numpy()
                    valid_indices = []
                    for a in valid:
                        ci, x, y = a
                        if ci < 0:
                            i = 0
                        else:
                            i = 1 + ci * (18 * 15) + int(x - 0.5) * 15 + int(y // 4)
                        valid_indices.append(min(i, len(policy) - 1))
                    probs = [max(policy[i], 1e-10) for i in valid_indices]
                    total = sum(probs)
                    probs = [p / total for p in probs]
                    temp = 0.5
                    probs = [math.log(p) / temp for p in probs]
                    ex = [math.exp(p) for p in probs]
                    s = sum(ex)
                    probs = [p / s for p in ex]
                    idx = random.choices(range(len(valid)), weights=probs)[0]
                    actions[0] = valid[idx]
                else:
                    actions[pid] = random.choice(valid) if valid else (-1, 0, 0)
            game.step(actions)
            game.step_n(150)
        if game.winner == 0:
            wins += 1
    wr = wins / num_games
    elapsed = time.time() - eval_start
    print(f'[Eval] {num_games} games, win rate vs random: {wr:.2%} ({elapsed:.1f}s {num_games/elapsed:.1f}g/s)')
    return wr


if __name__ == '__main__':
    import sys
    if '--eval' in sys.argv:
        trainer = Trainer(device='mps' if torch.backends.mps.is_available() else 'cpu')
        ckpt = os.path.join(CHECKPOINT_DIR, 'latest.pt')
        if os.path.exists(ckpt):
            trainer.load(ckpt)
        evaluate_vs_baseline(trainer, 20)
    else:
        n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        run_selfplay_loop(n_games)
