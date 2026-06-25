# === CLASH ROYALE AI - KAGGLE: paste into 1 cell & run ===
# First run: just paste & run. Resume: see instructions at bottom.
import os, sys, subprocess, shutil, json, time

os.system('pip install torch numpy pygame -q')
import torch
os.system('pip install kagglehub -q')

if not os.path.exists('/kaggle/working/cr'):
    os.makedirs('/kaggle/working/cr', exist_ok=True)
    # If you have your own repo, replace URL below. Default uses a template.
    subprocess.run(['git', 'clone', '--depth', '1',
        'https://github.com/AIAPI12/cr',
        '/kaggle/working/cr'], capture_output=True)

PROJ = '/kaggle/working/cr'
os.chdir(PROJ)
sys.path.insert(0, os.path.join(PROJ, 'src'))
sys.path.insert(0, PROJ)

CHECKPOINT_DIR = os.path.join(PROJ, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Restore checkpoint from Dataset if resuming
CKPT_SRC = '/kaggle/input/cr-checkpoint/latest.pt'
resume = '--resume' in sys.argv
if resume and os.path.exists(CKPT_SRC):
    shutil.copy2(CKPT_SRC, os.path.join(CHECKPOINT_DIR, 'latest.pt'))
    print(f'[Resume] Restored from Dataset cr-checkpoint')

# Run training
from ai.train import run_selfplay_loop, evaluate_vs_baseline, format_time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_games = 130000
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    n_games = int(sys.argv[1])

print(f'[Kaggle] Device: {device}  Games: {n_games:,}')
print(f'[Kaggle] Model: 7b 64f  AMP: enabled  Step-n: 150')
print(f'[Kaggle] PPO: 4ep  batch:128  mini-batch:256')
est_gps = 4.5 if device == 'cuda' else 0.5
print(f'[Kaggle] Est time: {n_games/est_gps:.0f}s ({n_games/est_gps/3600:.1f}h)')
print()

start = time.time()
trainer = run_selfplay_loop(
    num_games=n_games, batch_size=128, save_every=1000,
    lr=3e-4, res_blocks=7, filters=64, dropout=0.0,
    device=device, eval_every=2000, step_n=150,
    ppo_epochs=4, mini_batch_size=256, use_amp=True,
)
elapsed = time.time() - start
gps = n_games / elapsed if elapsed > 0 else 0

trainer.save(os.path.join(CHECKPOINT_DIR, 'latest.pt'))

wr = evaluate_vs_baseline(trainer, num_games=50, step_n=150)

print(f'\n[Kaggle] === DONE ===')
print(f'[Kaggle] Games: {n_games:,}  Time: {format_time(elapsed)}  Speed: {gps:.1f} g/s')
print(f'[Kaggle] Win rate vs random: {wr:.2%}')
print(f'[Kaggle] Checkpoint: /kaggle/working/cr/checkpoints/latest.pt')

"""
=== RESUME INSTRUCTIONS ===
1. After session ends, go to /kaggle/working/cr/checkpoints/
2. Download latest.pt to your computer
3. Create a new Kaggle Dataset named "cr-checkpoint" with that file
4. In a new notebook, use this same cell but add: --resume

=== GAME COUNT FOR 30 HOURS ===
Kaggle GPU limit: 9h/session, 30h/week
With ~4.5 games/s on T4 GPU:
  Per session (9h):  ~145,000 games
  4 sessions (~30h): ~580,000 games (need resume each time)

Use: python kaggle_notebook.py 145000  (for one 9h session)
Use: python kaggle_notebook.py 580000  (with --resume for ~30h total)
"""
