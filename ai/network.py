import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import numpy as np

FEATURE_CHANNELS = 20
GRID_H, GRID_W = 8, 18
NUM_X = 18
NUM_Y_TROOP = 15
ACTION_DIM = 1 + 4 * NUM_X * NUM_Y_TROOP

class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x

class CRNetwork(nn.Module):
    def __init__(self, channels=FEATURE_CHANNELS, res_blocks=5, filters=48, dropout=0.0):
        super().__init__()
        self.conv_input = nn.Conv2d(channels, filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(filters)
        self.res_blocks = nn.ModuleList([ResidualBlock(filters, dropout) for _ in range(res_blocks)])

        self.policy_conv = nn.Conv2d(filters, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * GRID_H * GRID_W, ACTION_DIM)

        self.value_conv = nn.Conv2d(filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_H * GRID_W, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, value

    def get_action_log_probs(self, logits, actions):
        """Compute log probabilities of taken actions (masked only for valid)."""
        log_probs_all = F.log_softmax(logits, dim=1)
        aidx = []
        for a in actions:
            ci, x, y = a
            if ci < 0:
                aidx.append(0)
            else:
                xi = int(x - 0.5)
                yi = int(y // 4)
                idx = 1 + ci * (NUM_X * NUM_Y_TROOP) + xi * NUM_Y_TROOP + yi
                aidx.append(min(idx, ACTION_DIM - 1))
        at = torch.LongTensor(aidx).to(logits.device)
        return log_probs_all.gather(1, at.unsqueeze(1)).squeeze(1)

    def get_masked_entropy(self, logits, valid_actions_list):
        """Compute entropy over valid actions only (masked softmax)."""
        probs_all = F.softmax(logits, dim=1)
        entropies = []
        for b in range(logits.size(0)):
            valid = valid_actions_list[b]
            if not valid:
                entropies.append(torch.tensor(0.0, device=logits.device))
                continue
            vidx = []
            for a in valid:
                ci, x, y = a
                if ci < 0: i = 0
                else: i = 1 + ci * (NUM_X * NUM_Y_TROOP) + int(x - 0.5) * NUM_Y_TROOP + int(y // 4)
                vidx.append(min(i, ACTION_DIM - 1))
            vp = probs_all[b, vidx]
            vp = vp / (vp.sum() + 1e-10)
            entropies.append(-(vp * torch.log(vp + 1e-10)).sum())
        return torch.stack(entropies)


class Trainer:
    def __init__(self, lr=3e-4, weight_decay=1e-4, device='cpu',
                 res_blocks=5, filters=48, dropout=0.0, compile_model=False,
                 use_amp=False):
        self.device = device
        self.use_amp = use_amp and device != 'cpu' and torch.cuda.is_available()
        torch.set_num_threads(1)
        self.network = CRNetwork(res_blocks=res_blocks, filters=filters, dropout=dropout).to(device)
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.network = torch.compile(self.network)
                print(f'[Network] Compiled model')
            except Exception as e:
                print(f'[Network] Compile skipped: {e}')
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2)
        try:
            self.scaler = torch.amp.GradScaler(device) if self.use_amp else None
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.grad_clip = 3.0
        self.step_count = 0
        # PPO params
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_kl = 0.02
        self.gamma = 0.99
        self.gae_lambda = 0.95

    def save(self, path):
        state = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step_count,
        }
        if self.scaler:
            state['scaler'] = self.scaler.state_dict()
        torch.save(state, path)

    def load(self, path):
        if not os.path.exists(path): return
        ckpt = torch.load(path, map_location=self.device)
        try:
            self.network.load_state_dict(ckpt['network'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.step_count = ckpt.get('step', 0)
            if self.scaler and 'scaler' in ckpt:
                self.scaler.load_state_dict(ckpt['scaler'])
        except RuntimeError as e:
            print(f'[Warn] Architecture mismatch, starting fresh: {e}')

    def compute_gae(self, rewards, values, dones):
        """Compute GAE(λ) advantages."""
        advantages = []
        gae = 0
        next_val = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            next_val = values[t]
        return advantages

    def train_ppo(self, trajectory_lists, ppo_epochs=3, mini_batch_size=64):
        flat_states = []
        flat_actions = []
        flat_valid = []
        flat_adv = []
        flat_ret = []
        flat_old_logp = []

        for traj in trajectory_lists:
            if len(traj) < 2:
                continue
            states = torch.FloatTensor([t['state'] for t in traj]).to(self.device)
            actions = [t['action'] for t in traj]
            rewards = [t['reward'] for t in traj]
            dones = [t.get('done', False) for t in traj]
            valid_list = [t.get('valid', [(-1, 0, 0)]) for t in traj]

            with torch.no_grad():
                logits, values = self.network(states)
                values_np = values.squeeze().cpu().numpy().tolist()

            advantages = self.compute_gae(rewards, values_np, dones)
            returns = [adv + v for adv, v in zip(advantages, values_np)]

            with torch.no_grad():
                old_logp = self.network.get_action_log_probs(
                    self.network(states)[0], actions).detach()

            flat_states.extend(states.cpu())
            flat_actions.extend(actions)
            flat_valid.extend(valid_list)
            flat_adv.extend(advantages)
            flat_ret.extend(returns)
            flat_old_logp.extend(old_logp.cpu())

        n = len(flat_states)
        if n == 0:
            return {'loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0,
                    'approx_kl': 0, 'value_mean': 0, 'reward_mean': 0}

        adv_t = torch.FloatTensor(flat_adv).to(self.device)
        ret_t = torch.FloatTensor(flat_ret).to(self.device)
        old_logp_t = torch.stack(flat_old_logp).to(self.device)
        states_t = torch.stack(flat_states).to(self.device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        updates = 0

        indices = list(range(n))
        for epoch in range(ppo_epochs):
            random.shuffle(indices)
            epoch_kl = 0
            epoch_updates = 0
            for start in range(0, n, mini_batch_size):
                mb_idx = indices[start:start + mini_batch_size]
                mb_states = states_t[mb_idx]
                mb_actions = [flat_actions[i] for i in mb_idx]
                mb_valid = [flat_valid[i] for i in mb_idx]
                mb_old = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                amp_device_type = self.device if self.device != 'mps' else 'cpu'
                with torch.amp.autocast(device_type=amp_device_type, enabled=self.use_amp):
                    logits, vals = self.network(mb_states)
                    new_logp = self.network.get_action_log_probs(logits, mb_actions)
                    ratio = torch.exp(new_logp - mb_old)

                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    entropy = self.network.get_masked_entropy(logits, mb_valid)
                    entropy_loss = -self.entropy_coef * entropy.mean()
                    value_loss = self.value_coef * F.mse_loss(vals.squeeze(), mb_ret)
                    loss = policy_loss + entropy_loss + value_loss

                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                    self.optimizer.step()
                self.scheduler.step()
                self.step_count += 1

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                epoch_kl += approx_kl
                epoch_updates += 1
                updates += 1

                if approx_kl > self.max_kl:
                    break

            if epoch_updates > 0 and epoch_kl / epoch_updates > self.max_kl:
                break

        n_up = max(updates, 1)
        return {
            'loss': total_loss / n_up,
            'policy_loss': total_policy_loss / n_up,
            'value_loss': total_value_loss / n_up,
            'entropy': total_entropy / n_up,
            'approx_kl': total_approx_kl / n_up,
            'value_mean': 0,
            'reward_mean': sum(sum(t['reward'] for t in traj) for traj in trajectory_lists) / max(len(trajectory_lists), 1),
        }

    def train_step(self, states, actions, rewards, policy_targets=None):
        """Legacy single-step training (for backwards compat / MCTS mode)."""
        B = len(states)
        st = torch.FloatTensor(states).to(self.device)
        rt = torch.FloatTensor(rewards).to(self.device)

        logits, vals = self.network(st)

        if policy_targets is not None:
            pt = torch.FloatTensor(policy_targets).to(self.device)
            pl = -(pt * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        else:
            aidx = []
            for a in actions:
                ci, x, y = a
                if ci < 0:
                    aidx.append(0)
                else:
                    xi = int(x - 0.5)
                    yi = int(y // 4)
                    idx = 1 + ci * (NUM_X * NUM_Y_TROOP) + xi * NUM_Y_TROOP + yi
                    aidx.append(min(idx, ACTION_DIM - 1))
            at = torch.LongTensor(aidx).to(self.device)
            pl = F.nll_loss(F.log_softmax(logits, dim=1), at)

        vl = F.mse_loss(vals.squeeze(), rt)
        loss = pl + vl

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step_count += 1

        return {
            'loss': loss.item(),
            'policy_loss': pl.item(),
            'value_loss': vl.item(),
            'value_mean': vals.mean().item(),
            'reward_mean': rt.mean().item(),
        }

    def sample_action(self, state, valid_actions, temperature=1.0):
        with torch.no_grad():
            logits, val = self.network(state.unsqueeze(0))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        vidx = []
        for a in valid_actions:
            ci, x, y = a
            if ci < 0:
                i = 0
            else:
                i = 1 + ci * (NUM_X * NUM_Y_TROOP) + int(x - 0.5) * NUM_Y_TROOP + int(y // 4)
            vidx.append(min(i, len(probs)-1))
        vp = [max(probs[i], 1e-10) for i in vidx]
        total = sum(vp)
        vp = [p/total for p in vp]
        if temperature > 0:
            vp = [math.log(p)/temperature for p in vp]
            ex = [math.exp(p) for p in vp]
            s = sum(ex)
            vp = [p/s for p in ex]
        return random.choices(valid_actions, weights=vp)[0]
