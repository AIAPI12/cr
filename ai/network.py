import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os

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
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value


class Trainer:
    def __init__(self, lr=3e-4, weight_decay=1e-4, device='cpu',
                 res_blocks=5, filters=48, dropout=0.0, compile_model=False):
        self.device = device
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
        self.grad_clip = 3.0
        self.step_count = 0

    def save(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step_count,
        }, path)

    def load(self, path):
        if not os.path.exists(path): return
        ckpt = torch.load(path, map_location=self.device)
        try:
            self.network.load_state_dict(ckpt['network'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.step_count = ckpt.get('step', 0)
        except RuntimeError as e:
            print(f'[Warn] Architecture mismatch (arch changed?), starting fresh: {e}')

    def train_step(self, states, actions, rewards, policy_targets=None):
        B = len(states)
        st = torch.FloatTensor(states).to(self.device)
        rt = torch.FloatTensor(rewards).to(self.device)

        logits, vals = self.network(st)

        if policy_targets is not None:
            # MCTS mode: cross-entropy with visit distribution as target
            pt = torch.FloatTensor(policy_targets).to(self.device)
            pl = -(pt * logits).sum(dim=1).mean()
        else:
            # Single action mode: NLL loss
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
            pl = F.nll_loss(logits, at)

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
        probs = torch.exp(logits).squeeze(0).cpu().numpy()
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
