import os, sys, subprocess, shutil

ROOT = '/content/cr'
os.makedirs(ROOT, exist_ok=True)
os.chdir(ROOT)

# Check if gamedata.json exists
if not os.path.exists('gamedata.json'):
    paths = ['/content/gamedata.json', '/content/drive/MyDrive/clash-royale-ai/gamedata.json']
    for p in paths:
        if os.path.exists(p):
            shutil.copy2(p, 'gamedata.json')
            break
    if not os.path.exists('gamedata.json'):
        print('ERROR: gamedata.json not found. Upload it first.')
        sys.exit(1)

print('Patching files...')

# env.py
e = open('ai/env.py').read()
e = e.replace(
    "import builtins\n_original_print = builtins.print\ndef _quiet_print(*args, **kwargs):\n    if args and isinstance(args[0], str):\n        if any(s in args[0] for s in ['[Lifecycle]', '[Detect]', '[Attach]', '[Mechanic]', '[Warn]', '====', 'Warning']):\n            return\n    _original_print(*args, **kwargs)\nbuiltins.print = _quiet_print",
    "import builtins\nbuiltins.print = lambda *a, **k: None"
)
e = e.replace('game.step_n(150)', 'game.step_n(step_n)')
e = e.replace(
    'def play_game_mcts(self, temperature=1.0, record=True, mcts_sims=50, c_puct=1.5):',
    'def play_game_mcts(self, temperature=1.0, record=True, mcts_sims=50, c_puct=1.5, step_n=150):'
)
e = e.replace(
    'def play_game(self, use_network=True, temperature=1.0, record=True):',
    'def play_game(self, use_network=True, temperature=1.0, record=True, step_n=150):'
)
open('ai/env.py', 'w').write(e)

# train.py
t = open('ai/train.py').read()
t = t.replace(
    "device='cpu', use_mcts=False, mcts_sims=50, eval_every=0):",
    "device='cpu', use_mcts=False, mcts_sims=50, eval_every=0, step_n=150):"
)
t = t.replace(
    'sp.play_game_mcts(temperature=temp, record=True, mcts_sims=mcts_sims)\n        else:\n            sp.play_game(temperature=temp, record=True)',
    'sp.play_game_mcts(temperature=temp, record=True, mcts_sims=mcts_sims, step_n=step_n)\n        else:\n            sp.play_game(temperature=temp, record=True, step_n=step_n)'
)
t = t.replace('def evaluate_vs_baseline(trainer, num_games=20):', 'def evaluate_vs_baseline(trainer, num_games=20, step_n=150):')
t = t.replace('game.step_n(150)\n        if game.winner == 0:', 'game.step_n(step_n)\n        if game.winner == 0:')
open('ai/train.py', 'w').write(t)

# colab_train.py
c = open('colab_train.py').read()
c = c.replace(
    "parser.add_argument('--no-turbo', action='store_false', dest='turbo',\n                       help='Disable turbo (1 train step per game instead of 3)')\n    args = parser.parse_args()",
    "parser.add_argument('--no-turbo', action='store_false', dest='turbo',\n                       help='Disable turbo (1 train step per game instead of 3)')\n    parser.add_argument('--step-n', type=int, default=150,\n                       help='Steps per decision')\n    args = parser.parse_args()"
)
c = c.replace('eval_every=args.eval,\n    )', 'eval_every=args.eval,\n        step_n=args.step_n,\n    )')
c = c.replace('est_gps = 3.0', 'speed_factor = 150.0 / args.step_n\n        est_gps = 3.0 * speed_factor')
c = c.replace('est_gps = 0.5', 'speed_factor = 150.0 / args.step_n\n        est_gps = 0.5 * speed_factor')
c = c.replace(
    'evaluate_vs_baseline(trainer, num_games=max(20, args.eval // 10))',
    'evaluate_vs_baseline(trainer, num_games=max(20, args.eval // 10), step_n=args.step_n)'
)
open('colab_train.py', 'w').write(c)

print('Patched OK')
print('Starting training...')

# Run training
sys.stdout.flush()
cmd = [sys.executable, 'colab_train.py'] + sys.argv[1:]
os.execvp(sys.executable, cmd)
