import math
import random
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import defaultdict


NUM_X = 18
NUM_Y_TROOP = 15
ACTION_DIM = 1 + 4 * NUM_X * NUM_Y_TROOP


class MCTSNode:
    __slots__ = ('prior', 'visit_count', 'value_sum', 'children', 'parent',
                 'action', 'action_key', 'is_expanded', 'virtual_loss')

    def __init__(self, prior: float = 0.0, action_key: str = None, parent=None):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: List['MCTSNode'] = []
        self.parent = parent
        self.action_key = action_key
        self.is_expanded = False
        self.virtual_loss = 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        if self.visit_count == 0:
            return float('inf')
        parent_visits = self.parent.visit_count if self.parent else 0
        q = self.value()
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count + self.virtual_loss)
        return q + u

    def best_child(self, c_puct: float) -> Optional['MCTSNode']:
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb_score(c_puct))

    def select_leaf(self, c_puct: float) -> 'MCTSNode':
        node = self
        while node.is_expanded and node.children:
            node = node.best_child(c_puct)
        return node

    def expand(self, action_probs: Dict[str, float]):
        for action_key, prob in action_probs.items():
            child = MCTSNode(prior=prob, action_key=action_key, parent=self)
            self.children.append(child)
        self.is_expanded = True

    def backup(self, value: float):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent


def encode_action(action) -> str:
    ci, x, y = action
    if ci < 0:
        return 'pass'
    xi = int(x - 0.5)
    yi = int(y // 4)
    return f'{ci}_{xi}_{yi}'


def decode_action(key: str):
    if key == 'pass':
        return (-1, 0, 0)
    parts = key.split('_')
    ci = int(parts[0])
    xi = int(parts[1])
    yi = int(parts[2])
    return (ci, xi + 0.5, yi * 4 + 0.5)


def action_to_idx(action) -> int:
    ci, x, y = action
    if ci < 0:
        return 0
    xi = int(x - 0.5)
    yi = int(y // 4)
    idx = 1 + ci * (NUM_X * NUM_Y_TROOP) + xi * NUM_Y_TROOP + yi
    return min(idx, ACTION_DIM - 1)


def idx_to_action(idx: int):
    if idx <= 0:
        return (-1, 0, 0)
    idx -= 1
    ci = idx // (NUM_X * NUM_Y_TROOP)
    rem = idx % (NUM_X * NUM_Y_TROOP)
    xi = rem // NUM_Y_TROOP
    yi = rem % NUM_Y_TROOP
    return (ci, xi + 0.5, yi * 4 + 0.5)


def get_network_action_probs(network, state_tensor, valid_actions, device='cpu'):
    """Get action probability vector from network for a single state."""
    with torch.inference_mode():
        st = torch.FloatTensor(state_tensor).unsqueeze(0).to(device)
        logits, value = network(st)
        probs = torch.exp(logits).squeeze(0).cpu().numpy()

    action_probs = {}
    for a in valid_actions:
        idx = action_to_idx(a)
        p = max(probs[idx], 1e-10)
        action_probs[encode_action(a)] = p

    total = sum(action_probs.values())
    if total > 0:
        for k in action_probs:
            action_probs[k] /= total

    return action_probs, value.item()


def mcts_search(
    game,
    pid: int,
    network,
    game_clone_fn: Callable,
    num_simulations: int = 50,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_weight: float = 0.25,
    temperature: float = 1.0,
    device: str = 'cpu',
) -> Tuple[np.ndarray, Any]:
    """Run MCTS search and return (policy_vector, selected_action)."""

    root = MCTSNode(prior=1.0)
    opp_pid = 1 - pid

    for sim in range(num_simulations):
        leaf = root.select_leaf(c_puct)
        sim_game = game
        sim_actions = []

        # Replay actions from root to leaf
        if leaf.action_key is not None:
            nodes_path = []
            node = leaf
            while node.parent is not None:
                nodes_path.append(node)
                node = node.parent
            nodes_path.reverse()

            sim_game = game_clone_fn(game)
            for n in nodes_path:
                action = decode_action(n.action_key)
                sim_actions.append(action)

                # Get opponent action using network
                opp_valid = sim_game.get_valid_actions(opp_pid)
                if opp_valid:
                    opp_ap, _ = get_network_action_probs(network, sim_game.get_state_tensor(opp_pid), opp_valid, device)
                    opp_action_key = max(opp_ap, key=opp_ap.get)
                    opp_action = decode_action(opp_action_key)
                else:
                    opp_action = (-1, 0, 0)

                sim_game.step({pid: action, opp_pid: opp_action})

                if sim_game.game_over:
                    break
                sim_game.step_n(150)

            if sim_game.game_over:
                w = sim_game.winner
                value = 1.0 if w == pid else (-1.0 if w is not None else 0.0)
                leaf.backup(value)
                continue

        # Get policy and value from network at leaf
        state_tensor = sim_game.get_state_tensor(pid)
        valid_actions = sim_game.get_valid_actions(pid)
        action_probs, leaf_value = get_network_action_probs(network, state_tensor, valid_actions, device)

        # Add Dirichlet noise at root
        if sim == 0 and leaf is root:
            alpha = [dirichlet_alpha] * len(action_probs)
            noise = np.random.dirichlet(alpha)
            for i, k in enumerate(action_probs):
                action_probs[k] = (1 - dirichlet_weight) * action_probs[k] + dirichlet_weight * noise[i]

        leaf.expand(action_probs)
        leaf.backup(-leaf_value)

    # Build visit distribution as policy vector
    policy_vec = np.zeros(ACTION_DIM, dtype=np.float32)
    if root.children:
        total_visits = sum(c.visit_count for c in root.children)
        if total_visits > 0:
            for child in root.children:
                if child.visit_count > 0:
                    action = decode_action(child.action_key)
                    idx = action_to_idx(action)
                    policy_vec[idx] = child.visit_count / total_visits

    # Select action
    valid = game.get_valid_actions(pid)
    if not valid:
        return policy_vec, (-1, 0, 0)

    if temperature < 0.01:
        # Greedy: most visited
        if root.children:
            best = max(root.children, key=lambda c: c.visit_count)
            return policy_vec, decode_action(best.action_key)
        return policy_vec, random.choice(valid)

    # Temperature-weighted sampling from children
    children_with_visits = [c for c in root.children if c.visit_count > 0]
    if not children_with_visits:
        return policy_vec, random.choice(valid)

    weights = [c.visit_count ** (1.0 / temperature) for c in children_with_visits]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    chosen = random.choices(children_with_visits, weights=probs)[0]
    return policy_vec, decode_action(chosen.action_key)
