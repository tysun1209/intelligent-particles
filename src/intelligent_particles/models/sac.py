"""Scalar-action Soft Actor-Critic for bounded steering."""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


class SAC:
    def __init__(self, features, hidden, device, *, gamma=0.99, alpha=0.1, tau=0.005):
        self.device = torch.device(device)
        self.gamma, self.alpha, self.tau = gamma, alpha, tau
        self.actor = mlp(features, hidden, 2).to(self.device)  # mean and log standard deviation
        self.critics = nn.ModuleList([mlp(features + 1, hidden, 1) for _ in range(2)]).to(self.device)
        self.targets = deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=3e-4)

    def _sample(self, obs):
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        normal = torch.distributions.Normal(mean, log_std.clamp(-5, 2).exp())
        raw = normal.rsample()
        log_jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return raw.tanh(), normal.log_prob(raw) - log_jacobian

    @torch.no_grad()
    def normalized_turn(self, obs, *, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if deterministic:
            return self.actor(obs)[:, 0].tanh().cpu().numpy()
        return self._sample(obs)[0][:, 0].cpu().numpy()

    def get_deterministic_action_batch(self, obs):
        return self.normalized_turn(obs, deterministic=True)[:, None]

    def update(self, batch):
        obs, action, reward, following, done = (
            torch.as_tensor(field, dtype=torch.float32, device=self.device) for field in batch
        )
        with torch.no_grad():
            turn, logp = self._sample(following)
            pair = torch.cat((following, turn), dim=1)
            value = torch.minimum(self.targets[0](pair), self.targets[1](pair)) - self.alpha * logp
            target = reward + self.gamma * (1 - done) * value
        pair = torch.cat((obs, action), dim=1)
        loss = sum(F.mse_loss(q(pair), target) for q in self.critics)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        turn, logp = self._sample(obs)
        pair = torch.cat((obs, turn), dim=1)
        self.critics.requires_grad_(False)
        value = torch.minimum(self.critics[0](pair), self.critics[1](pair))
        self.actor_optimizer.zero_grad()
        (self.alpha * logp - value).mean().backward()
        self.actor_optimizer.step()
        self.critics.requires_grad_(True)
        with torch.no_grad():
            for target_param, param in zip(self.targets.parameters(), self.critics.parameters()):
                target_param.lerp_(param, self.tau)
