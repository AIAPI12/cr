# Kaggle Notebook — Clash Royale AI Training

**URL:** `https://www.kaggle.com/` → Create → New Notebook

**Settings:**
- Internet: **ON** (click Internet toggle)
- Accelerator: **GPU T4 x2**
- Persistence: Files in `/kaggle/working/cr/` persist across sessions

---

## Cell 1 — Clone + Install

```python
!git clone https://github.com/AIAPI12/cr.git /kaggle/working/cr
%cd /kaggle/working/cr
!pip install torch numpy -q
```

---

## Cell 2 — Train (First Session)

```python
!python kaggle_train.py --games 250000 --step-n 400 --save-every 1000 --eval 5000 --no-oracle
```

**Flags:**
- `--games 250000` = 250K games (~8h with step-n 400)
- `--step-n 400` = ~8 games/s on T4, ~45 decisions/game
- `--no-oracle` = no Oracle VM needed (saves locally in Kaggle persistent storage)
- `--resume` = continue from latest checkpoint

---

## Cell 2B — Resume (Subsequent Session)

```python
!python kaggle_train.py --games 500000 --step-n 400 --save-every 1000 --eval 5000 --no-oracle --resume
```

---

## Cell 3 — Eval

```python
!python main.py eval
```

---

## Cell 4 — Download Checkpoint

```python
from IPython.display import FileLink
FileLink('/kaggle/working/cr/checkpoints/latest.pt')
```

---

## Cell 5 — GUI (if you want to watch)

```python
# Install pygame for visualization
!pip install pygame -q
!python main.py gui
```

---

## Kaggle Quota Management

| Metric | Limit |
|--------|-------|
| GPU quota | 30h/week |
| Session max | 12h |
| Persistent storage | 73 GB |
| Auto-save | Yes (Kaggle auto-saves) |

**Use `--games` to fit one session:**
- `--step-n 400` → ~8 games/s → 250K games = ~8.7h ✅
- `--step-n 400` → 350K games = ~12.2h ❌ (exceeds 12h session)
- `--step-n 600` → ~12 games/s → 350K games = ~8.1h ✅ (but dumber play)
- `--step-n 300` → ~6 games/s → 250K games = ~11.6h ✅ (smarter play)

---

## Without Kaggle (local GPU)

```bash
pip install torch numpy -q
python kaggle_train.py --games 1000 --device cuda --save-every 100 --eval 200 --no-oracle
python main.py gui
```
