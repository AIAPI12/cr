import os, sys, subprocess, shutil, json, time
os.system('pip install torch numpy -q')
import torch

PROJ = '/kaggle/working/cr'
if not os.path.exists(PROJ):
    os.makedirs(PROJ, exist_ok=True)
    subprocess.run(['git', 'clone', '--depth', '1',
        'https://github.com/AIAPI12/cr',
        PROJ], capture_output=True)

os.chdir(PROJ)
sys.path.insert(0, os.path.join(PROJ, 'src'))
sys.path.insert(0, PROJ)

CHECKPOINT_DIR = os.path.join(PROJ, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CKPT_SRC = '/kaggle/input/cr-checkpoint/latest.pt'
resume = '--resume' in sys.argv
if resume and os.path.exists(CKPT_SRC):
    shutil.copy2(CKPT_SRC, os.path.join(CHECKPOINT_DIR, 'latest.pt'))
    print(f'[Resume] Restored from Dataset cr-checkpoint')

from ai.train import run_selfplay_loop, evaluate_vs_baseline, format_time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_games = 167000
dashboard_url = os.environ.get('DASHBOARD_URL', '')

print(f'[Kaggle] Device: {device}  Games: {n_games:,}')
print(f'[Kaggle] Model: 7b 64f | step_n=60 | PPO: 4ep batch:256 | Workers: 8')
if dashboard_url:
    print(f'[Kaggle] Dashboard: {dashboard_url}')
print()

start = time.time()
trainer = run_selfplay_loop(
    num_games=n_games,
    batch_size=256,
    save_every=2000,
    lr=3e-4,
    res_blocks=7,
    filters=64,
    dropout=0.01,
    device=device,
    eval_every=5000,
    step_n=60,
    ppo_epochs=4,
    mini_batch_size=256,
    use_amp=(device == 'cuda'),
    compile_model=(device == 'cuda'),
    workers=8,
    dashboard_url=dashboard_url,
)

elapsed = time.time() - start
gps = n_games / elapsed if elapsed > 0 else 0

trainer.save(os.path.join(CHECKPOINT_DIR, 'latest.pt'))
print(f'\n[Kaggle] DONE: {n_games:,}g in {format_time(elapsed)} ({gps:.1f} g/s)')

if dashboard_url:
    try:
        import urllib.request
        msg = f'Kaggle done: {n_games:,}g in {format_time(elapsed)} ({gps:.1f} g/s)'
        req = urllib.request.Request(f'{dashboard_url}/api/live?msg={urllib.parse.quote(msg)}&level=info')
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass
