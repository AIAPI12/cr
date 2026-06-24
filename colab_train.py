#!/usr/bin/env python3
"""
Colab training script for Clash Royale AlphaZero.
Mounts Google Drive for checkpoint persistence, uses GPU for fast MCTS training.
"""
import sys, os, time, json, random, math, argparse
from pathlib import Path

# Install any missing deps
try:
    import torch
except ImportError:
    os.system('pip install torch numpy')
    import torch

PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ)
CHECKPOINT_DIR = os.path.join(PROJ, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Colab: mount Google Drive for persistent checkpoints
DRIVE_DIR = '/content/drive/MyDrive/clash-royale-ai'
try:
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs(DRIVE_DIR, exist_ok=True)
    REMOTE_DIR = DRIVE_DIR
    print('[Colab] Google Drive mounted')
except ImportError:
    REMOTE_DIR = CHECKPOINT_DIR
    print('[Colab] Not in Colab, using local checkpoints')

sys.path.insert(0, os.path.join(PROJ, 'src'))

from ai.env import CRGame, SelfPlaySystem
from ai.network import Trainer
from ai.train import run_selfplay_loop, evaluate_vs_baseline


def sync_checkpoint(local: str, remote: str, direction: str = 'push'):
    """Sync checkpoint between local and remote."""
    if direction == 'push' and os.path.exists(local):
        import shutil
        shutil.copy2(local, remote)
        print(f'[Sync] Pushed {local} -> {remote}')
    elif direction == 'pull' and os.path.exists(remote):
        import shutil
        shutil.copy2(remote, local)
        print(f'[Sync] Pulled {remote} -> {local}')


def main():
    parser = argparse.ArgumentParser(description='Colab AlphaZero Training')
    parser.add_argument('--games', type=int, default=200,
                       help='Number of self-play games (default: 200)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--mcts-sims', type=int, default=50,
                       help='MCTS simulations per move (default: 50)')
    parser.add_argument('--c-puct', type=float, default=1.5,
                       help='MCTS exploration constant (default: 1.5)')
    parser.add_argument('--blocks', type=int, default=5,
                       help='Residual blocks (default: 5)')
    parser.add_argument('--filters', type=int, default=48,
                       help='Convolution filters (default: 48)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate (default: 0.0)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--save-every', type=int, default=25,
                       help='Save checkpoint every N games (default: 25)')
    parser.add_argument('--eval', type=int, default=20,
                       help='Evaluate vs random every N games (default: 20, 0=off)')
    parser.add_argument('--no-mcts', action='store_true',
                       help='Disable MCTS (use raw policy instead)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    parser.add_argument('--no-turbo', action='store_false', dest='turbo',
                       help='Disable turbo (1 train step per game instead of 3)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[Colab] Device: {device}  Games: {args.games}')
    print(f'[Colab] Model: {args.blocks}blocks {args.filters}filters')
    print(f'[Colab] MCTS: {"on" if not args.no_mcts else "off"} sims={args.mcts_sims}')

    # Pull checkpoint from Google Drive if resuming
    if args.resume:
        sync_checkpoint(
            os.path.join(CHECKPOINT_DIR, 'latest.pt'),
            os.path.join(REMOTE_DIR, 'latest.pt'),
            'pull'
        )

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
        use_mcts=not args.no_mcts,
        mcts_sims=args.mcts_sims,
    )

    # Push checkpoint to Google Drive
    sync_checkpoint(
        os.path.join(CHECKPOINT_DIR, 'latest.pt'),
        os.path.join(REMOTE_DIR, 'latest.pt'),
        'push'
    )

    # Final evaluation
    print('\n[Colab] Final evaluation vs random baseline...')
    wr = evaluate_vs_baseline(trainer, num_games=args.eval if args.eval > 0 else 10)
    print(f'[Colab] Final win rate vs random: {wr:.2%}')


if __name__ == '__main__':
    main()
