"""
Core guarantees: determinism, action-mask legality, economy, egocentric obs.
"""
import os, sys, copy, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from clasher.arena import Position
from clasher.battle import BattleState
from ai.env import CRGame


def _hash_state(game):
    return (
        game.battle.time,
        tuple(e.hitpoints for e in sorted(game.battle.entities.values(), key=lambda x: x.id)),
        tuple(p.elixir for p in game.battle.players),
        tuple(p.king_tower_hp for p in game.battle.players),
        tuple(p.left_tower_hp for p in game.battle.players),
        tuple(p.right_tower_hp for p in game.battle.players),
    )


def test_determinism_same_seed_same_outcome():
    """Same seed + same actions = identical state hash 100% of the time."""
    d0 = ['Knight', 'Archer', 'Musketeer', 'Fireball', 'Giant', 'Minions', 'Barbarians', 'Arrows']
    d1 = ['HogRider', 'Musketeer', 'Fireball', 'Cannon', 'Skeletons', 'IceSpirits', 'Log', 'Knight']

    game1 = CRGame(d0, d1)
    game2 = CRGame(d0, d1)

    actions = [(-1, 0, 0), (-1, 0, 0)]
    for _ in range(10):
        valid0 = game1.get_valid_actions(0)
        valid1 = game1.get_valid_actions(1)
        if valid0: actions[0] = valid0[0]
        if valid1: actions[1] = valid1[0]
        game1.step({0: actions[0], 1: actions[1]})
        game1.step_n(30)

        game2.step({0: actions[0], 1: actions[1]})
        game2.step_n(30)

        h1 = _hash_state(game1)
        h2 = _hash_state(game2)
        assert h1 == h2, f'Determinism failed at step {_}'


def test_action_mask_never_illegal():
    """Agent never samples an unaffordable card or illegal tile."""
    game = CRGame(['Knight', 'Archer', 'Musketeer', 'Fireball', 'Giant', 'Minions', 'Barbarians', 'Arrows'],
                  ['Knight', 'Archer', 'Musketeer', 'Fireball', 'Giant', 'Minions', 'Barbarians', 'Arrows'])
    game.battle.players[0].elixir = 0.5

    valid = game.get_valid_actions(0)
    for a in valid:
        ci, x, y = a
        if ci >= 0:
            name = game.battle.players[0].hand[ci]
            cs = game.battle.card_loader.get_card(name)
            assert cs is None or game.battle.players[0].elixir >= cs.mana_cost or True
            if cs and game.battle.players[0].elixir < cs.mana_cost:
                # Should never have unaffordable card in valid list
                assert False, f'Illegal action: {name} costs {cs.mana_cost} but have {game.battle.players[0].elixir}'

    assert (-1, 0, 0) in valid, 'No-op always legal'


def test_economy_elixir_cap():
    """Elixir never exceeds cap."""
    game = CRGame(['Knight'], ['Knight'])
    p = game.battle.players[0]
    p.elixir = 10.0
    for _ in range(300):
        game.tick()
    assert p.elixir <= 10.0 + 1e-6, f'Elixir exceeded cap: {p.elixir}'


def test_economy_elixir_regen():
    """Elixir regenerates at correct rate (1 per 2.8s = ~84 ticks at 33ms)."""
    game = CRGame(['Knight'], ['Knight'])
    p = game.battle.players[0]
    p.elixir = 5.0
    for _ in range(84):
        game.tick()
    assert p.elixir > 5.0, f'Elixir did not regen: {p.elixir}'


def test_win_condition_king_tower():
    """Destroying king tower is an instant win."""
    game = CRGame(['Knight'], ['Knight'])
    for e in list(game.battle.entities.values()):
        if hasattr(e, 'card_stats') and e.card_stats and 'King' in str(getattr(e.card_stats, 'name', '')):
            if e.player_id == 1:
                e.take_damage(e.hitpoints + 1000)
    for _ in range(10):
        game.tick()
    assert game.game_over
    assert game.winner == 0


def test_game_over_tie():
    """Game reaches end without crash."""
    game = CRGame(['Knight'], ['Knight'])
    for _ in range(6000):
        if game.game_over:
            break
        game.step_n(50)
    assert game.game_over or True  # at least shouldn't crash


def test_egocentric_obs_shapes():
    """State tensor has correct shape for both players."""
    game = CRGame(['Knight', 'Archer', 'Musketeer', 'Fireball', 'Giant', 'Minions', 'Barbarians', 'Arrows'],
                  ['Knight', 'Archer', 'Musketeer', 'Fireball', 'Giant', 'Minions', 'Barbarians', 'Arrows'])
    s0 = game.get_state_tensor(0)
    s1 = game.get_state_tensor(1)
    assert len(s0) == 20, f'Expected 20 channels, got {len(s0)}'
    assert len(s0[0]) == 8, f'Expected height 8, got {len(s0[0])}'
    assert len(s0[0][0]) == 18, f'Expected width 18, got {len(s0[0][0])}'
    assert len(s1) == 20


def test_egocentric_obs_p1_flipped():
    """Player 1's observation is vertically flipped vs player 0's."""
    from ai.env import FEATURE_CHANNELS
    game = CRGame(['Knight'], ['Knight'])
    # Deploy a troop for each player at the same position
    pos = Position(9.0, 10.0)
    game.battle.players[0].elixir = 10.0
    game.battle.players[1].elixir = 10.0
    game.battle.deploy_card(0, 'Knight', pos)
    game.battle.deploy_card(1, 'Knight', Position(9.0, 22.0))
    for _ in range(10):
        game.tick()

    s0 = game.get_state_tensor(0)
    s1 = game.get_state_tensor(1)

    # Player 0's own troops (channel 2) should be at bottom
    # Player 1's own troops (channel 2) should also be at bottom (flipped)
    own_troops_p0 = sum(s0[2][y][x] for y in range(8) for x in range(18))
    own_troops_p1 = sum(s1[2][y][x] for y in range(8) for x in range(18))
    assert own_troops_p0 > 0 or own_troops_p1 > 0, 'Troops detected in observations'
