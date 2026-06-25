#!/usr/bin/env python3
"""
Kaggle training script for Clash Royale AlphaZero.
Runs raw policy (no MCTS) for max throughput.
Syncs checkpoints to/from Oracle VM for persistence across sessions.

Usage on Kaggle:
  1. Create notebook with Internet ON
  2. Add Kaggle secret for ORACLE_HOST, ORACLE_SSH_KEY, ORACLE_USER
  3. Run: python kaggle_train.py --games 250000 --step-n 400
"""
import sys, os, time, json, subprocess, argparse, shutil, tempfile

_PROJ = os.path.abspath(os.path.dirname(__file__))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
_SRC = os.path.join(_PROJ, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import torch
except ImportError:
    os.system('pip install torch numpy -q')
    import torch

CHECKPOINT_DIR = os.path.join(_PROJ, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IN_KAGGLE = os.path.exists('/kaggle')
IN_COLAB = 'COLAB_GPU' in os.environ or 'COLAB_JUPYTER_IP' in os.environ

# Oracle VM config (set via env or Kaggle secrets)
ORACLE_HOST = os.environ.get('ORACLE_HOST', '')
ORACLE_USER = os.environ.get('ORACLE_USER', 'ubuntu')
ORACLE_SSH_KEY = os.environ.get('ORACLE_SSH_KEY', '')
ORACLE_PORT = os.environ.get('ORACLE_PORT', '22')
ORACLE_CHECKPOINT_PATH = os.environ.get('ORACLE_CHECKPOINT_PATH', '/home/ubuntu/cr-checkpoints/')

# Path already set up above

from ai.env import CRGame, SelfPlaySystem
from ai.network import Trainer
from ai.train import run_selfplay_loop, evaluate_vs_baseline


def oracle_sync(local_path, filename, direction='push'):
    if not ORACLE_HOST:
        return
    remote = f'{ORACLE_USER}@{ORACLE_HOST}:{ORACLE_CHECKPOINT_PATH}{filename}'
    try:
        if direction == 'push':
            if os.path.exists(local_path):
                subprocess.run(
                    ['scp', '-i', ORACLE_SSH_KEY, '-P', ORACLE_PORT,
                     '-o', 'StrictHostKeyChecking=no',
                     '-o', 'UserKnownHostsFile=/dev/null',
                     local_path, remote],
                    capture_output=True, timeout=30)
        else:
            subprocess.run(
                ['scp', '-i', ORACLE_SSH_KEY, '-P', ORACLE_PORT,
                 '-o', 'StrictHostKeyChecking=no',
                 '-o', 'UserKnownHostsFile=/dev/null',
                 remote, local_path],
                capture_output=True, timeout=30)
    except Exception:
        pass


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0: return f'{h}h{m:02d}m'
    if m > 0: return f'{m}m{s:02d}s'
    return f'{s}s'


def main():
    parser = argparse.ArgumentParser(description='Kaggle AlphaZero Training')
    parser.add_argument('--games', type=int, default=150000,
                       help='Number of self-play games (default: 150k)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size (default: 128)')
    parser.add_argument('--blocks', type=int, default=7,
                       help='Residual blocks (default: 7)')
    parser.add_argument('--filters', type=int, default=64,
                       help='Convolution filters (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate (default: 0.0)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--save-every', type=int, default=1000,
                       help='Save checkpoint every N games (default: 1000)')
    parser.add_argument('--eval', type=int, default=2000,
                       help='Evaluate vs random every N games (default: 2000, 0=off)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    parser.add_argument('--step-n', type=int, default=150,
                       help='Game steps per decision (default: 150)')
    parser.add_argument('--no-oracle', action='store_true',
                       help='Disable Oracle VM sync')
    parser.add_argument('--ppo-epochs', type=int, default=4,
                       help='PPO epochs per batch (default: 4)')
    parser.add_argument('--mini-batch-size', type=int, default=256,
                       help='PPO mini-batch size (default: 256)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Kaggle] Device: {device}  Games: {args.games:,}')
    print(f'[Kaggle] Model: {args.blocks}b {args.filters}f  Step-n: {args.step_n}')
    print(f'[Kaggle] PPO: {args.ppo_epochs}ep {args.mini_batch_size}mb {args.batch_size}b')
    print(f'[Kaggle] Oracle: {"disabled" if args.no_oracle else "enabled (" + ORACLE_HOST + ")"}')

    if not args.no_oracle and ORACLE_HOST and args.resume:
        print(f'[Kaggle] Pulling checkpoint from Oracle VM...')
        oracle_sync(
            os.path.join(CHECKPOINT_DIR, 'latest.pt'),
            'latest.pt', 'pull')

    base_speed = 4.5 if device == 'cuda' else 0.5
    speed_factor = 150.0 / args.step_n
    est_gps = base_speed * speed_factor
    est_seconds = args.games / est_gps
    print(f'[Kaggle] Est. throughput: ~{est_gps:.1f} games/s')
    print(f'[Kaggle] Est. total time: {format_time(est_seconds)}')
    if est_seconds > 32400:
        print(f'[Kaggle] NOTE: Exceeds 9h session. Will resume from checkpoint.')

    start_wall = time.time()
    trainer = run_selfplay_loop(
        num_games=args.games,
        batch_size=args.batch_size,
        save_every=args.save_every,
        turbo=True,
        lr=args.lr,
        res_blocks=args.blocks,
        filters=args.filters,
        dropout=args.dropout,
        device=device,
        use_mcts=False,
        mcts_sims=50,
        eval_every=args.eval,
        step_n=args.step_n,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        use_amp=True,
    )
    elapsed = time.time() - start_wall
    actual_gps = args.games / elapsed if elapsed > 0 else 0

    ckpt_path = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    trainer.save(ckpt_path)

    if not args.no_oracle and ORACLE_HOST:
        print(f'[Kaggle] Pushing checkpoint to Oracle VM...')
        oracle_sync(ckpt_path, 'latest.pt', 'push')

    print(f'\n[Kaggle] === Final MCTS evaluation vs random ===')
    wr = evaluate_vs_baseline(trainer, num_games=max(20, args.eval // 10), step_n=args.step_n)
    print(f'[Kaggle] === Final win rate: {wr:.2%} ===')

    print(f'\n[Kaggle] === DONE ===')
    print(f'[Kaggle] Games: {args.games:,}  Time: {format_time(elapsed)}  Speed: {actual_gps:.1f} g/s')
    print(f'[Kaggle] Checkpoint: {ckpt_path}')
    print(f'[Kaggle] Download and run locally: python main.py gui')


if __name__ == '__main__':
    main()
