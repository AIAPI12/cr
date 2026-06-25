import os, sys, subprocess, shutil, json, time
os.system('pip install torch numpy pygame -q')
import torch
if torch.cuda.is_available():
    os.system('pip install kagglehub -q')
    try:
        with open('/kaggle/input/cr-opt/cr_opt.txt') as f:
            exec(f.read())
    except: pass
if not os.path.exists('/kaggle/working/cr'):
    os.makedirs('/kaggle/working/cr', exist_ok=True)
    subprocess.run(['git', 'clone', '--depth', '1',
        'https://github.com/AIAPI12/cr',
        '/kaggle/working/cr'], capture_output=True)
PROJ = '/kaggle/working/cr'
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
n_games = 130000
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    n_games = int(sys.argv[1])

print(f'[Kaggle] Device: {device}  Games: {n_games:,}')
print(f'[Kaggle] Model: 7b 64f | step_n=60 | PPO: 3ep batch:256')
est_gps = 6.5 if device == 'cuda' else 0.5
print(f'[Kaggle] Est: {n_games/est_gps:.0f}s ({n_games/est_gps/3600:.1f}h)')
print()

start = time.time()
trainer = run_selfplay_loop(
    num_games=n_games, batch_size=256, save_every=1000,
    lr=3e-4, res_blocks=7, filters=64, dropout=0.0,
    device=device, eval_every=2000, step_n=60,
    ppo_epochs=3, mini_batch_size=256, use_amp=False,
    compile_model=True,
)
elapsed = time.time() - start
gps = n_games / elapsed if elapsed > 0 else 0

trainer.save(os.path.join(CHECKPOINT_DIR, 'latest.pt'))
wr = evaluate_vs_baseline(trainer, num_games=50, step_n=60)

print(f'\n[Kaggle] DONE: {n_games:,}g in {format_time(elapsed)} ({gps:.1f} g/s)')
print(f'[Kaggle] Win rate vs rand: {wr:.2%}')
print(f'[Kaggle] Ckpt: /kaggle/working/cr/checkpoints/latest.pt')
