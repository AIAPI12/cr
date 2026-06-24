#!/usr/bin/env python3
"""
Colab training script for Clash Royale AlphaZero.
Usage:
  1. Mount Drive manually first:
     from google.colab import drive
     drive.mount('/content/drive')
  2. python colab_train.py --games 100000 --turbo

Runs raw policy (no MCTS) for max throughput.
Periodically evaluates with MCTS for accurate win-rate measurement.
"""
import sys, os, time, json, random, math, argparse, shutil
from pathlib import Path

try:
    import torch
except ImportError:
    os.system('pip install torch numpy -q')
    import torch

PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ)
CHECKPOINT_DIR = os.path.join(PROJ, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Detect Colab environment
IN_COLAB = 'COLAB_GPU' in os.environ or 'COLAB_JUPYTER_IP' in os.environ
DRIVE_DIR = '/content/drive/MyDrive/cr'
REMOTE_DIR = CHECKPOINT_DIR  # default to local

if IN_COLAB:
    if os.path.isdir('/content/drive/MyDrive'):
        os.makedirs(DRIVE_DIR, exist_ok=True)
        REMOTE_DIR = DRIVE_DIR
        print('[Colab] Drive already mounted at /content/drive/MyDrive')
    else:
        print('[Colab] Drive not mounted. Checkpoints saved locally only.')
        print('  To enable persistent saves, run this cell FIRST:')
        print('  from google.colab import drive; drive.mount(\'/content/drive\')')
else:
    print('[Colab] Not in Colab, using local checkpoints')

sys.path.insert(0, os.path.join(PROJ, 'src'))

from ai.env import CRGame, SelfPlaySystem
from ai.network import Trainer
from ai.train import run_selfplay_loop, evaluate_vs_baseline


def sync_checkpoint(local, remote, direction='push'):
    if direction == 'push' and os.path.exists(local):
        shutil.copy2(local, remote)
        print(f'[Sync] Pushed {local} -> {remote}')
    elif direction == 'pull' and os.path.exists(remote):
        shutil.copy2(remote, local)
        print(f'[Sync] Pulled {remote} -> {local}')


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0: return f'{h}h{m:02d}m'
    if m > 0: return f'{m}m{s:02d}s'
    return f'{s}s'


def main():
    parser = argparse.ArgumentParser(description='Colab AlphaZero Training')
    parser.add_argument('--games', type=int, default=100000,
                       help='Number of self-play games (default: 100k)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size (default: 128)')
    parser.add_argument('--mcts-sims', type=int, default=50,
                       help='MCTS sims for eval (default: 50)')
    parser.add_argument('--blocks', type=int, default=5,
                       help='Residual blocks (default: 5)')
    parser.add_argument('--filters', type=int, default=48,
                       help='Convolution filters (default: 48)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate (default: 0.0)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--save-every', type=int, default=500,
                       help='Save checkpoint every N games (default: 500)')
    parser.add_argument('--eval', type=int, default=1000,
                       help='Evaluate vs random every N games (default: 1000, 0=off)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    parser.add_argument('--no-turbo', action='store_false', dest='turbo',
                       help='Disable turbo (1 train step per game instead of 3)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n[Colab] Device: {device}  Games: {args.games:,}')
    print(f'[Colab] Model: {args.blocks}b {args.filters}f')
    print(f'[Colab] Turbo: {args.turbo}  Batch: {args.batch_size}')
    print(f'[Colab] Eval every {args.eval} games | Save every {args.save_every}')
    print(f'[Colab] ---')

    # Resume checkpoint from Drive
    if args.resume:
        sync_checkpoint(
            os.path.join(CHECKPOINT_DIR, 'latest.pt'),
            os.path.join(REMOTE_DIR, 'latest.pt'),
            'pull'
        )

    # Estimate: ~3 games/sec on T4, ~0.5 games/sec on CPU
    if device == 'cuda':
        est_gps = 3.0
        est_label = 'T4 GPU'
    else:
        est_gps = 0.5
        est_label = 'CPU'
    est_seconds = args.games / est_gps
    print(f'[Colab] Est. throughput: ~{est_gps:.1f} games/s ({est_label})')
    print(f'[Colab] Est. total time: {format_time(est_seconds)} ({format_time(est_seconds/3600)}h wall)')
    if est_seconds > 43200:
        print(f'[Colab] WARNING: This exceeds the 12h Colab limit!')
        print(f'[Colab] Consider reducing --games or use --resume to continue later')

    start_wall = time.time()
    trainer = run_selfplay_loop(
        num_games=args.games,
        batch_size=args.batch_size,
        save_every=args.save_every,
        turbo=args.turbo,
        lr=args.lr,
        res_blocks=args.blocks,
        filters=args.filters,
        dropout=args.dropout,
        device=device,
        use_mcts=False,  # Raw policy for speed, MCTS for eval only
        mcts_sims=args.mcts_sims,
        eval_every=args.eval,
    )
    elapsed = time.time() - start_wall
    actual_gps = args.games / elapsed if elapsed > 0 else 0

    # Push checkpoint
    sync_checkpoint(
        os.path.join(CHECKPOINT_DIR, 'latest.pt'),
        os.path.join(REMOTE_DIR, 'latest.pt'),
        'push'
    )

    # Final MCTS evaluation
    print(f'\n[Colab] === Final MCTS-{args.mcts_sims} evaluation vs random ===')
    wr = evaluate_vs_baseline(trainer, num_games=max(20, args.eval // 10))
    print(f'[Colab] === Final win rate: {wr:.2%} ===')

    print(f'\n[Colab] === DONE ===')
    print(f'[Colab] Games: {args.games:,}  Time: {format_time(elapsed)}  Speed: {actual_gps:.1f} g/s')
    print(f'[Colab] Checkpoint: {CHECKPOINT_DIR}/latest.pt')
    if REMOTE_DIR != CHECKPOINT_DIR:
        print(f'[Colab] Drive copy: {REMOTE_DIR}/latest.pt')
    print(f'[Colab] Download and run locally: python main.py gui')


if __name__ == '__main__':
    main()
