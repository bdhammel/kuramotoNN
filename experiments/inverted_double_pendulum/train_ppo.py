"""Train an InvertedDoublePendulum policy with PPO, logging to wandb.

    python train_ppo.py --total-steps 1000000
    python train_ppo.py --policy kuramoto
    python train_ppo.py --wandb-mode disabled   # skip wandb entirely

Same actor (GaussianPolicy wrapping the mlp or kuramoto mean-net) and the same
checkpoint format as train.py's REINFORCE, so eval.py works unmodified on
checkpoints from either trainer -- only the training algorithm differs. PPO
adds one thing REINFORCE doesn't have: a critic (state-value baseline), used
only to reduce the policy gradient's variance and discarded at eval time.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.distributions import Normal

from pymoto import calibrate, create_model

from model import Critic, GaussianPolicy, PolicyNet

ENV_ID = "InvertedDoublePendulum-v5"
OBS_DIM = 9
ACTION_DIM = 1


@dataclass
class HParams:
    policy: str = "mlp"  # "mlp" or "kuramoto"
    hidden_size: int = 128  # mlp actor only
    n_oscillators: int = 64  # kuramoto only
    kuramoto_steps: int = 10  # kuramoto only; see pymoto's kuramoto_inverted_double_pendulum
    kuramoto_g: float = 1.0  # kuramoto only
    kuramoto_k_scale: float = 1.0  # kuramoto only
    trainable_drive: bool = False  # kuramoto only; train W alongside K (see pymoto.layers.TrainableDrive)
    trainable_head: bool = False  # kuramoto only; train H alongside K (see pymoto.layers.TrainableHead)
    calibration_size: int = 1024  # kuramoto only
    init_log_std: float = 0.0
    critic_hidden_size: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048  # env steps collected before each PPO update
    total_steps: int = 1_000_000  # env-step budget across all rollouts
    epochs: int = 10  # PPO passes over each rollout
    minibatch_size: int = 64
    seed: int = 1
    solved_reward: float = 9100.0  # InvertedDoublePendulum-v5's reward_threshold
    log_every: int = 1  # rollouts between console prints


def parse_args() -> tuple[HParams, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = HParams()
    parser.add_argument("--policy", choices=["mlp", "kuramoto"], default=defaults.policy)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--n-oscillators", type=int, default=defaults.n_oscillators)
    parser.add_argument("--kuramoto-steps", type=int, default=defaults.kuramoto_steps)
    parser.add_argument("--kuramoto-g", type=float, default=defaults.kuramoto_g)
    parser.add_argument("--kuramoto-k-scale", type=float, default=defaults.kuramoto_k_scale)
    parser.add_argument("--trainable-drive", action="store_true", default=defaults.trainable_drive)
    parser.add_argument("--trainable-head", action="store_true", default=defaults.trainable_head)
    parser.add_argument("--calibration-size", type=int, default=defaults.calibration_size)
    parser.add_argument("--init-log-std", type=float, default=defaults.init_log_std)
    parser.add_argument("--critic-hidden-size", type=int, default=defaults.critic_hidden_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--clip-epsilon", type=float, default=defaults.clip_epsilon)
    parser.add_argument("--value-coef", type=float, default=defaults.value_coef)
    parser.add_argument("--entropy-coef", type=float, default=defaults.entropy_coef)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--rollout-steps", type=int, default=defaults.rollout_steps)
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--solved-reward", type=float, default=defaults.solved_reward)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--wandb-project", default="inverted-double-pendulum-ppo")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    hp = HParams(
        policy=args.policy,
        hidden_size=args.hidden_size,
        n_oscillators=args.n_oscillators,
        kuramoto_steps=args.kuramoto_steps,
        kuramoto_g=args.kuramoto_g,
        kuramoto_k_scale=args.kuramoto_k_scale,
        trainable_drive=args.trainable_drive,
        trainable_head=args.trainable_head,
        calibration_size=args.calibration_size,
        init_log_std=args.init_log_std,
        critic_hidden_size=args.critic_hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        rollout_steps=args.rollout_steps,
        total_steps=args.total_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        solved_reward=args.solved_reward,
        log_every=args.log_every,
    )
    return hp, args


def collect_calibration_batch(seed: int, size: int) -> torch.Tensor:
    """Random-action rollout states, for the kuramoto policy's calibrate().

    Uses its own throwaway env/seed so it doesn't disturb the training env's
    seeded reset sequence.
    """
    calib_env = gym.make(ENV_ID)
    calib_env.reset(seed=seed)
    states = []
    obs, _ = calib_env.reset()
    while len(states) < size:
        states.append(obs)
        obs, _, terminated, truncated, _ = calib_env.step(calib_env.action_space.sample())
        if terminated or truncated:
            obs, _ = calib_env.reset()
    calib_env.close()
    return torch.as_tensor(np.array(states), dtype=torch.float32)


def build_actor(hp: HParams) -> GaussianPolicy:
    if hp.policy == "mlp":
        mean_net = PolicyNet(obs_dim=OBS_DIM, hidden_size=hp.hidden_size)
    else:
        mean_net = create_model(
            "kuramoto_inverted_double_pendulum",
            n=hp.n_oscillators,
            num_steps=hp.kuramoto_steps,
            k_scale=hp.kuramoto_k_scale,
            trainable_drive=hp.trainable_drive,
            trainable_head=hp.trainable_head,
        )
        x_cal = collect_calibration_batch(seed=hp.seed + 10_000, size=hp.calibration_size)
        calibrate(mean_net, x_cal, g=hp.kuramoto_g)
    return GaussianPolicy(mean_net, init_log_std=hp.init_log_std)


class RolloutCollector:
    """Steps `env` with `actor`/`critic`, filling fixed-length rollouts across episode boundaries.

    Episode reward/length are tracked across rollouts (an episode can span
    more than one rollout); `collect()` returns the episodes that completed
    during that call alongside the rollout tensors.
    """

    def __init__(self, env: gym.Env):
        self.env = env
        self.obs, _ = env.reset()
        self.episode_reward = 0.0
        self.episode_length = 0

    @torch.no_grad()
    def collect(self, actor: GaussianPolicy, critic: Critic, rollout_steps: int) -> tuple[dict, list[tuple[float, int]]]:
        obs_buf = torch.zeros(rollout_steps, OBS_DIM)
        actions_buf = torch.zeros(rollout_steps, ACTION_DIM)
        logprobs_buf = torch.zeros(rollout_steps)
        rewards_buf = torch.zeros(rollout_steps)
        dones_buf = torch.zeros(rollout_steps)
        values_buf = torch.zeros(rollout_steps)
        completed_episodes = []

        for t in range(rollout_steps):
            obs_t = torch.as_tensor(self.obs, dtype=torch.float32).unsqueeze(0)
            mean, std = actor(obs_t)
            dist = Normal(mean.squeeze(0), std.squeeze(0))
            action = dist.sample()

            obs_buf[t] = obs_t.squeeze(0)
            actions_buf[t] = action
            logprobs_buf[t] = dist.log_prob(action).sum()
            values_buf[t] = critic(obs_t).squeeze()

            clipped = torch.clamp(action, -1.0, 1.0)
            next_obs, reward, terminated, truncated, _ = self.env.step(clipped.numpy())
            done = terminated or truncated
            rewards_buf[t] = reward
            # Terminated and truncated are both treated as episode boundaries for
            # GAE masking -- a simplification (a truncated episode's value isn't
            # actually zero beyond the horizon) that's standard in introductory
            # PPO implementations and immaterial here since episodes are capped
            # at 1000 steps of unit reward either way.
            dones_buf[t] = float(done)

            self.episode_reward += reward
            self.episode_length += 1
            if done:
                completed_episodes.append((self.episode_reward, self.episode_length))
                self.episode_reward = 0.0
                self.episode_length = 0
                next_obs, _ = self.env.reset()
            self.obs = next_obs

        bootstrap_value = critic(torch.as_tensor(self.obs, dtype=torch.float32).unsqueeze(0)).squeeze()
        batch = dict(
            obs=obs_buf, actions=actions_buf, logprobs=logprobs_buf,
            rewards=rewards_buf, dones=dones_buf, values=values_buf,
            bootstrap_value=bootstrap_value,
        )
        return batch, completed_episodes


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
    bootstrap_value: torch.Tensor, gamma: float, gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation. -> (advantages, returns), both (T,)."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = bootstrap_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def ppo_update(
    actor: GaussianPolicy, critic: Critic, optimizer: optim.Optimizer, batch: dict,
    advantages: torch.Tensor, returns: torch.Tensor, hp: HParams,
) -> dict:
    """One PPO update: `hp.epochs` passes over `batch` in minibatches. -> mean loss components."""
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    dataset_size = hp.rollout_steps
    all_params = list(itertools.chain(actor.parameters(), critic.parameters()))

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}
    for _ in range(hp.epochs):
        perm = torch.randperm(dataset_size)
        for start in range(0, dataset_size, hp.minibatch_size):
            idx = perm[start : start + hp.minibatch_size]

            mean, std = actor(batch["obs"][idx])
            dist = Normal(mean, std)
            new_logprobs = dist.log_prob(batch["actions"][idx]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = torch.exp(new_logprobs - batch["logprobs"][idx])
            surrogate1 = ratio * advantages[idx]
            surrogate2 = torch.clamp(ratio, 1 - hp.clip_epsilon, 1 + hp.clip_epsilon) * advantages[idx]
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            values_pred = critic(batch["obs"][idx]).squeeze(-1)
            value_loss = F.mse_loss(values_pred, returns[idx])

            loss = policy_loss + hp.value_coef * value_loss - hp.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, hp.max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["n_updates"] += 1

    n = stats.pop("n_updates")
    return {k: v / n for k, v in stats.items()}


def main() -> None:
    hp, args = parse_args()
    print(f"HParams: {asdict(hp)}")

    torch.manual_seed(hp.seed)
    env = gym.make(ENV_ID)
    env.reset(seed=hp.seed)
    env.action_space.seed(hp.seed)

    run = wandb.init(project=args.wandb_project, mode=args.wandb_mode, config=asdict(hp))
    print(f"run={run.name}")

    actor = build_actor(hp)
    critic = Critic(obs_dim=OBS_DIM, hidden_size=hp.critic_hidden_size)
    optimizer = optim.Adam(itertools.chain(actor.parameters(), critic.parameters()), lr=hp.lr)
    # The random-K control in eval.py needs the exact initial coupling matrix.
    K_init = actor.get_coupling().K.detach().clone() if hp.policy == "kuramoto" else None

    collector = RolloutCollector(env)
    running_reward = 0.0
    total_steps = 0
    rollout_idx = 0
    solved_at_step = None

    while total_steps < hp.total_steps:
        rollout_idx += 1
        batch, completed_episodes = collector.collect(actor, critic, hp.rollout_steps)
        total_steps += hp.rollout_steps

        advantages, returns = compute_gae(
            batch["rewards"], batch["values"], batch["dones"], batch["bootstrap_value"],
            hp.gamma, hp.gae_lambda,
        )
        losses = ppo_update(actor, critic, optimizer, batch, advantages, returns, hp)

        for episode_reward, _ in completed_episodes:
            running_reward = (
                episode_reward if running_reward == 0.0 and rollout_idx == 1
                else 0.05 * episode_reward + 0.95 * running_reward
            )

        log = {
            "total_steps": total_steps,
            "train/running_reward": running_reward,
            "train/policy_loss": losses["policy_loss"],
            "train/value_loss": losses["value_loss"],
            "train/entropy": losses["entropy"],
            "train/std": actor.log_std.exp().item(),
            "train/episodes_completed": len(completed_episodes),
        }
        if completed_episodes:
            log["train/episode_reward"] = np.mean([r for r, _ in completed_episodes])
            log["train/episode_length"] = np.mean([l for _, l in completed_episodes])
        wandb.log(log, step=total_steps)

        if rollout_idx % hp.log_every == 0:
            print(
                f"step {total_steps:8d}  rollout {rollout_idx:5d}  "
                f"running_reward {running_reward:8.1f}  "
                f"policy_loss {losses['policy_loss']:8.4f}  value_loss {losses['value_loss']:8.4f}"
            )

        if running_reward >= hp.solved_reward:
            solved_at_step = total_steps
            print(f"Solved at step {total_steps} with running reward {running_reward:.1f}")
            break

    env.close()
    torch.save(
        {"hparams": asdict(hp), "state_dict": actor.state_dict(), "K_init": K_init},
        "checkpoint.pt",
    )
    print("Saved actor weights to checkpoint.pt")

    run.summary.update(
        {
            "solved_at_step": solved_at_step,
            "final_running_reward": running_reward,
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
