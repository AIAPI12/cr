import sys, os, time, json, random, math, concurrent.futures
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai.env import CRGame, SelfPlaySystem
from ai.network import Trainer, CRNetwork, ACTION_DIM
import torch
import torch.nn.functional as F
torch.set_num_threads(4)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DASHBOARD_URL = os.environ.get('DASHBOARD_URL', 'http://localhost:8000')
TOTAL_GAMES_TARGET = 167000

EVOLVED_CARDS = {
    'Archers','BabyDragon','Barbarians','BattleRam','Bats','Bomber','Cannon',
    'DartGoblin','ElectroDragon','Executioner','Firecracker','Furnace',
    'GiantSnowball','GoblinBarrel','GoblinCage','GoblinDrill','GoblinGiant',
    'Hunter','IceSpirit','InfernoDragon','Knight','Lumberjack','MegaKnight',
    'MinionHorde','Mortar','Musketeer','Pekka','Princess','RoyalGhost',
    'RoyalGiant','RoyalHogs','RoyalRecruits','SkeletonArmy','SkeletonBarrel',
    'Skeletons','Tesla','Valkyrie','WallBreakers','Witch','Wizard','Zap',
}

CHAMPION_CARDS = {
    'ArcherQueen','SkeletonKing','GoldenKnight','MightyMiner','Monk',
    'LittlePrince','BossBandit','Goblinstein',
}


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0: return f'{h}h{m:02d}m'
    if m > 0: return f'{m}m{s:02d}s'
    return f'{s}s'


def _worker_play(worker_id, num_games, state_dict, res_blocks, filters, dropout, device, temperature_schedule, step_n, dashboard_url):
    import torch
    torch.set_num_threads(2)
    dev = 'cpu'
    net = CRNetwork(res_blocks=res_blocks, filters=filters, dropout=dropout).to(dev)
    net.load_state_dict(state_dict)
    net.eval()
    sp = SelfPlaySystem(network=net)

    all_trajs_p0 = []
    all_trajs_p1 = []
    win_count = 0
    total_steps = 0

    for gi in range(num_games):
        temp = temperature_schedule(gi)
        result = sp.play_game_ppo(temperature=temp, step_n=step_n)
        traj_p0 = result.get('traj_p0', [])
        traj_p1 = result.get('traj_p1', [])
        if traj_p0 and traj_p1:
            all_trajs_p0.append(traj_p0)
            all_trajs_p1.append(traj_p1)
        if result.get('winner') == 0:
            win_count += 1
        total_steps += result.get('steps', 0)

    return {
        'worker_id': worker_id,
        'traj_p0': all_trajs_p0,
        'traj_p1': all_trajs_p1,
        'wins': win_count,
        'games': num_games,
        'steps': total_steps,
    }


def post_metrics(metrics, url=None):
    target = url or DASHBOARD_URL
    if not target:
        return
    try:
        import urllib.request
        data = json.dumps(metrics).encode()
        req = urllib.request.Request(
            f'{target}/api/metrics',
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass


def post_live(msg, level='info', url=None):
    target = url or DASHBOARD_URL
    if not target:
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            f'{target}/api/live?msg={urllib.parse.quote(msg)}&level={level}',
            method='GET',
        )
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        pass


def run_selfplay_loop(num_games=TOTAL_GAMES_TARGET, batch_size=256, save_every=500,
                      lr=3e-4, res_blocks=5, filters=48, dropout=0.01, compile_model=False,
                      device='cpu', use_mcts=False, mcts_sims=50, eval_every=2000, step_n=150,
                      ppo_epochs=4, mini_batch_size=128, use_amp=False,
                      workers=6, dashboard_url=None):
    use_amp = use_amp and device != 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f'[Train] Device: {device}  AMP: {use_amp}  Workers: {workers}')
    print(f'[Train] lr={lr} blocks={res_blocks} filters={filters} dropout={dropout} step_n={step_n}')
    print(f'[Train] Target: {num_games:,} games  PPO epochs: {ppo_epochs}  batch: {mini_batch_size}')
    if dashboard_url:
        print(f'[Train] Dashboard: {dashboard_url}')

    trainer = Trainer(lr=lr, device=device, res_blocks=res_blocks,
                      filters=filters, dropout=dropout, compile_model=compile_model,
                      use_amp=use_amp)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest.pt')
    if os.path.exists(checkpoint_path):
        trainer.load(checkpoint_path)
        print(f'[Train] Loaded checkpoint (step {trainer.step_count})')

    def temp_schedule(gi):
        return 1.0 if gi < min(500, num_games // 20) else 0.5

    start = time.time()
    losses = []
    total_games_played = 0
    total_wins = 0
    metrics_log = []

    games_per_batch = max(1, batch_size // 32)
    games_per_worker = max(1, games_per_batch // workers) if workers > 0 else games_per_batch

    while total_games_played < num_games:
        remaining = num_games - total_games_played
        batch_games = min(games_per_batch * workers, remaining)

        if workers > 1 and device != 'mps':
            state_dict = trainer.network.state_dict()
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []
                for wi in range(workers):
                    w_games = max(1, batch_games // workers)
                    if wi == workers - 1:
                        w_games = batch_games - sum(max(1, batch_games // workers) for _ in range(wi))
                    if w_games <= 0:
                        continue
                    f = executor.submit(
                        _worker_play, wi, w_games, state_dict,
                        res_blocks, filters, dropout, device,
                        lambda gi, offset=total_games_played: temp_schedule(offset + gi),
                        step_n, dashboard_url,
                    )
                    futures.append(f)

                trajectory_buffer = []
                batch_wins = 0
                batch_steps = 0
                for f in concurrent.futures.as_completed(futures):
                    try:
                        result = f.result(timeout=300)
                        for tlist in [result['traj_p0'], result['traj_p1']]:
                            trajectory_buffer.extend(tlist)
                        batch_wins += result['wins']
                        batch_steps += result['steps']
                    except Exception as e:
                        print(f'[Worker] ERROR: {e}')

        else:
            sp = SelfPlaySystem(network=trainer.network)
            trajectory_buffer = []
            batch_wins = 0
            batch_steps = 0
            for gi in range(games_per_batch):
                temp = temp_schedule(total_games_played + gi)
                result = sp.play_game_ppo(temperature=temp, step_n=step_n)
                traj_p0 = result.get('traj_p0', [])
                traj_p1 = result.get('traj_p1', [])
                if traj_p0 and traj_p1:
                    trajectory_buffer.append(traj_p0)
                    trajectory_buffer.append(traj_p1)
                if result.get('winner') == 0:
                    batch_wins += 1
                batch_steps += result.get('steps', 0)

        total_games_played += batch_games
        total_wins += batch_wins

        if len(trajectory_buffer) > 0:
            metrics = trainer.train_ppo(
                trajectory_buffer,
                ppo_epochs=ppo_epochs,
                mini_batch_size=mini_batch_size,
            )
            if metrics['loss'] > 0:
                losses.append(metrics['loss'])

        elapsed = time.time() - start
        gps = total_games_played / max(elapsed, 0.01)
        avg_loss = sum(losses[-100:]) / max(len(losses[-100:]), 1) if losses else 0
        remaining_time = (num_games - total_games_played) / max(gps, 0.001)
        pct = total_games_played / num_games * 100
        win_rate = total_wins / max(total_games_played, 1)

        if total_games_played % max(1, min(200, num_games // 100)) == 0 or total_games_played >= num_games:
            print(f'[{total_games_played:7d}/{num_games}] {pct:5.1f}% L={avg_loss:.3f} '
                  f'WR={win_rate:.1%} {gps:.1f}g/s '
                  f'ETA={format_time(remaining_time)} Tot={format_time(elapsed)} Step={trainer.step_count}')

        if total_games_played % save_every <= batch_games or total_games_played >= num_games:
            trainer.save(checkpoint_path)
            if dashboard_url:
                try:
                    post_metrics({
                        'game': total_games_played,
                        'total_games': num_games,
                        'loss': avg_loss,
                        'policy_loss': metrics.get('policy_loss', 0),
                        'value_loss': metrics.get('value_loss', 0),
                        'entropy': metrics.get('entropy', 0),
                        'approx_kl': metrics.get('approx_kl', 0),
                        'gps': gps,
                        'pct': pct,
                        'eta_seconds': remaining_time,
                        'elapsed_seconds': elapsed,
                        'win_rate': win_rate,
                    }, url=dashboard_url)
                except Exception:
                    pass
                metrics_log.append({
                    'game': total_games_played,
                    'loss': avg_loss,
                    'gps': gps,
                    'win_rate': win_rate,
                })

        if eval_every > 0 and total_games_played % eval_every <= batch_games:
            wr = evaluate_vs_baseline(trainer, num_games=min(50, max(10, eval_every // 10)))
            trainer.save(checkpoint_path)
            if dashboard_url:
                try:
                    post_live(f'Eval: {wr:.1%} win rate vs random', 'info', url=dashboard_url)
                except Exception:
                    pass

    trainer.save(checkpoint_path)
    elapsed = time.time() - start
    print(f'[Done] {total_games_played} games in {elapsed:.1f}s ({total_games_played/elapsed:.1f} games/s)')
    return trainer


def evaluate_vs_baseline(trainer, num_games=20, step_n=150):
    sp = SelfPlaySystem(network=trainer.network)
    wins = 0
    eval_start = time.time()
    for i in range(num_games):
        deck0 = sp.get_random_deck()
        deck1 = sp.get_random_deck()
        game = CRGame(deck0, deck1)
        for _ in range(9000):
            if game.game_over:
                break
            actions = {}
            for pid in range(2):
                valid = game.get_valid_actions(pid)
                if pid == 0:
                    state = game.get_state_tensor(0)
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
                    with torch.no_grad():
                        policy_logits, _ = trainer.network(state_t)
                    policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                    valid_indices = []
                    for a in valid:
                        ci, x, y = a
                        if ci < 0:
                            i = 0
                        else:
                            i = 1 + ci * (18 * 15) + int(x - 0.5) * 15 + int(y // 4)
                        valid_indices.append(min(i, ACTION_DIM - 1))
                    probs = [max(policy[i], 1e-10) for i in valid_indices]
                    total = sum(probs)
                    probs = [p / total for p in probs]
                    temp = 0.5
                    probs = [math.log(p) / temp for p in probs]
                    ex = [math.exp(p) for p in probs]
                    s = sum(ex)
                    probs = [p / s for p in ex]
                    idx = random.choices(range(len(valid)), weights=probs)[0]
                    actions[0] = valid[idx]
                else:
                    actions[pid] = random.choice(valid) if valid else (-1, 0, 0)
            game.step(actions)
            game.step_n(step_n)
        if game.winner == 0:
            wins += 1
    wr = wins / num_games
    elapsed = time.time() - eval_start
    print(f'[Eval] {num_games} games, win rate vs random: {wr:.2%} ({elapsed:.1f}s {num_games/elapsed:.1f}g/s)')
    return wr


if __name__ == '__main__':
    if '--eval' in sys.argv:
        if torch.cuda.is_available():
            _dev = 'cuda'
        elif torch.backends.mps.is_available():
            _dev = 'mps'
        else:
            _dev = 'cpu'
        trainer = Trainer(device=_dev)
        ckpt = os.path.join(CHECKPOINT_DIR, 'latest.pt')
        if os.path.exists(ckpt):
            trainer.load(ckpt)
        evaluate_vs_baseline(trainer, 20)
    else:
        n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
        n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        if torch.cuda.is_available():
            dev = 'cuda'
        elif torch.backends.mps.is_available():
            dev = 'mps'
        else:
            dev = 'cpu'
        run_selfplay_loop(n_games, workers=n_workers, device=dev)
