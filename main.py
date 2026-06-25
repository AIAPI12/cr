#!/usr/bin/env python3
import sys, os, shutil

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BANNER = r"""
   _____ _           _    ____                   _   _
  / ____| |         | |  |  _ \                 | | (_)
 | |    | |__   __ _| |_ | |_) |_   _ _ __   ___| |_ _  ___  _ __
 | |    | '_ \ / _` | __||  _ <| | | | '_ \ / __| __| |/ _ \| '_ \
 | |____| | | | (_| | |_ | |_) | |_| | | | | (__| |_| | (_) | | | |
  \_____|_| |_|\__,_|\__||____/ \__, |_| |_|\___|\__|_|\___/|_| |_|
                                  __/ |
                                 |___/
"""

def print_help():
    print("Usage: python main.py <command> [options]")
    print("")
    print("Commands:")
    print("  gui              Launch the self-play GUI visualization")
    print("  train [N]        Run N self-play games (turbo default, --noturbo, --lr, --blocks, --filters)")
    print("  eval             Evaluate AI vs random baseline")
    print("  evolve [gen]     Evolve decks using genetic algorithm")
    print("  serve            Start continuous training server")
    print("  sync             Sync card stats from RoyaleAPI")
    print("")
    print("Defaults: blocks=5 filters=48 lr=3e-4 dropout=0.0 turbo=True device=cpu")
    print("")

def cmd_gui():
    from ai.gui import CRGUI
    gui = CRGUI()
    gui.run()

def _parse_args():
    args = {}
    for i, a in enumerate(sys.argv):
        if a.startswith('--'):
            k = a[2:]
            v = sys.argv[i+1] if i+1 < len(sys.argv) and not sys.argv[i+1].startswith('--') else True
            try: v = int(v)
            except:
                try: v = float(v)
                except: pass
            args[k] = v
    return args

def cmd_train():
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    a = _parse_args()
    turbo = a.get('turbo', True) if 'noturbo' not in a else False
    from ai.train import run_selfplay_loop
    device = a.get('device', 'cpu')
    run_selfplay_loop(n, turbo=turbo, lr=a.get('lr', 3e-4),
                      res_blocks=a.get('blocks', 5),
                      filters=a.get('filters', 48),
                      dropout=a.get('dropout', 0.0),
                      compile_model=a.get('compile', False),
                      device=device,
                      use_mcts=a.get('mcts', False),
                      mcts_sims=a.get('mcts-sims', 50))

def cmd_eval():
    from ai.train import evaluate_vs_baseline
    from ai.network import Trainer
    import torch
    _dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    trainer = Trainer(device=_dev)
    ckpt = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    if os.path.exists(ckpt):
        trainer.load(ckpt)
    evaluate_vs_baseline(trainer, 20)

def cmd_evolve():
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    from ai.env import SelfPlaySystem
    from ai.royaleapi import DeckOptimizer
    from ai.network import Trainer
    import torch
    _dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    trainer = Trainer(device=_dev)
    ckpt = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    if os.path.exists(ckpt):
        trainer.load(ckpt)
    sp = SelfPlaySystem(network=trainer.network)
    opt = DeckOptimizer(sp)
    best = opt.evolve(generations=n_gen)
    print(f"\n=== Best Deck Found ===")
    for i, c in enumerate(best):
        print(f"  {i+1}. {c}")
    # Save
    with open(os.path.join(CHECKPOINT_DIR, 'best_deck.json'), 'w') as f:
        json.dump({'deck': best, 'generation': n_gen}, f)
    print(f"Deck saved to checkpoints/best_deck.json")

def cmd_sync():
    from ai.royaleapi import RoyaleAPI
    api = RoyaleAPI()
    stats = api.get_card_stats()
    if stats:
        print(f"[Sync] Got {len(stats)} cards from RoyaleAPI")
        # Save to a cache file for reference
        cache_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'api_stats.json')
        with open(cache_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[Sync] Saved to {cache_path}")
    else:
        print("[Sync] No data from RoyaleAPI (offline or rate-limited)")

def cmd_serve():
    import time
    from ai.train import run_selfplay_loop
    a = _parse_args()
    device = a.get('device', 'cpu')
    print("[Serve] Continuous self-play training. Ctrl+C to stop.")
    game_count = 0
    try:
        while True:
            run_selfplay_loop(50, turbo=True, lr=a.get('lr', 3e-4),
                              res_blocks=a.get('blocks', 5),
                              filters=a.get('filters', 48),
                              dropout=a.get('dropout', 0.0),
                              compile_model=a.get('compile', False),
                              device=device,
                              use_mcts=a.get('mcts', False),
                              mcts_sims=a.get('mcts-sims', 50))
            game_count += 50
            print(f"[Serve] {game_count} games played. Buffer filling. Resuming...")
    except KeyboardInterrupt:
        print(f"\n[Serve] Stopped after {game_count} games.")

if __name__ == '__main__':
    import json
    if len(sys.argv) < 2:
        print(BANNER)
        print_help()
        sys.exit(1)

    cmd = sys.argv[1]
    commands = {
        'gui': cmd_gui,
        'train': cmd_train,
        'eval': cmd_eval,
        'evolve': cmd_evolve,
        'serve': cmd_serve,
        'sync': cmd_sync,
    }
    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print_help()
