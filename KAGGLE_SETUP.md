# Kaggle Training Setup

## Prerequisites
1. Oracle VM running with `orchestrator/setup_oracle.sh` (optional but recommended)
2. Add Kaggle Secrets:
   - Go to your notebook → Add-ons → Secrets
   - `ORACLE_HOST` = your Oracle VM IP
   - `ORACLE_SSH_KEY` = base64-encoded SSH private key (optional, omit for no-oracle mode)

## Notebook Cells

### Cell 1: Setup
```python
# Cell 1 — Clone repo and install
!git clone https://github.com/AIAPI12/cr.git /kaggle/working/cr
%cd /kaggle/working/cr
!pip install torch numpy -q
```

### Cell 2: Patch + Run (first session)
```python
# Cell 2 — Start training
!python kaggle_train.py --games 250000 --step-n 400 --save-every 1000 --eval 5000
```

### Cell 2B: Resume (subsequent sessions)
```python
# Cell 2B — Resume from Oracle VM checkpoint
!python kaggle_train.py --games 500000 --step-n 400 --save-every 1000 --eval 5000 --resume
```

### Cell 2C: No Oracle (standalone Kaggle, no VM)
```python
# Cell 2C — Standalone (no Oracle sync, saves locally)
!python kaggle_train.py --games 100000 --step-n 400 --save-every 500 --eval 2000 --no-oracle
```

### Cell 3: Evaluation
```python
# Cell 3 — Evaluate latest checkpoint
!python main.py eval
```

### Cell 4: Download checkpoint
```python
# Cell 4 — Download checkpoint for local use
from IPython.display import FileLink
FileLink('/kaggle/working/cr/checkpoints/latest.pt')
```

## Notes
- Kaggle gives 30h/week GPU (T4/P100), 12h sessions, 73GB persistent storage
- Internet must be enabled (click "Internet" toggle in Kaggle notebook settings)
- Checkpoints sync to Oracle VM if `ORACLE_HOST` secret is set
- Without Oracle, checkpoints persist only during the Kaggle session
- Use `--step-n 400` for ~8 games/s on T4 (~17h for 500K games)
