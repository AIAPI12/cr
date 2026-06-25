#!/usr/bin/env python3
"""
Oracle VM orchestrator — runs on the Oracle free-tier VM.
Monitors training progress, triggers Kaggle sessions, manages checkpoints.
Designed to be run as a cron job or systemd service.

Usage:
  python orchestrator.py              # One-shot check
  python orchestrator.py --watch      # Watch mode (loop)
  python orchestrator.py --status     # Print training status
"""
import os, sys, json, time, subprocess, argparse
from pathlib import Path

CHECKPOINT_DIR = Path('/opt/cr-checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def get_latest_checkpoint():
    """Return path and timestamp of latest checkpoint."""
    files = sorted(CHECKPOINT_DIR.glob('*.pt'), key=os.path.getmtime, reverse=True)
    if not files:
        return None, None
    f = files[0]
    return str(f), os.path.getmtime(f)

def estimate_progress():
    """Crude progress estimate from checkpoint filenames."""
    total_games = 0
    for f in CHECKPOINT_DIR.glob('*.pt'):
        name = f.stem
        if name.startswith('step_'):
            try:
                total_games += int(name.split('_')[1])
            except (IndexError, ValueError):
                pass
    return total_games

def status_report():
    """Print formatted training status."""
    latest, mtime = get_latest_checkpoint()
    total = estimate_progress()
    print('=== Training Status ===')
    print(f'Checkpoints:  {len(list(CHECKPOINT_DIR.glob("*.pt")))}')
    print(f'Latest:       {Path(latest).name if latest else "N/A"}')
    print(f'Total games:  {total:,}')
    if mtime:
        age = time.time() - mtime
        print(f'Last update:  {age/3600:.1f}h ago')
    print(f'Storage:      {sum(f.stat().st_size for f in CHECKPOINT_DIR.glob("*"))/1024/1024:.1f} MB')
    print('')

def watch_loop(interval=300):
    """Watch for new checkpoints."""
    seen = set()
    while True:
        current = {str(f) for f in CHECKPOINT_DIR.glob('*.pt')}
        new = current - seen
        for f in sorted(new):
            sz = os.path.getsize(f) / 1024 / 1024
            print(f'[Orch] New checkpoint: {Path(f).name} ({sz:.1f} MB)')
        if new:
            status_report()
        seen = current
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true', help='Watch mode')
    parser.add_argument('--status', action='store_true', help='Print status')
    parser.add_argument('--interval', type=int, default=300, help='Watch interval seconds')
    args = parser.parse_args()

    if args.watch:
        watch_loop(args.interval)
    elif args.status:
        status_report()
    else:
        status_report()
        print('Use --watch for continuous monitoring')

if __name__ == '__main__':
    main()
