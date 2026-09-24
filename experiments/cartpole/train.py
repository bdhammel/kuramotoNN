"""Train a CartPole policy with REINFORCE, logging training dynamics to wandb.

    python train.py --episodes 1000
    python train.py --policy kuramoto
    python train.py --wandb-mode disabled   # skip wandb entirely
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.distributions import Categorical

from pymoto import calibrate, create_model

from model import PolicyNet


@dataclass
class HParams:
    policy: str = "mlp"  # "mlp" or "kuramoto"
    hidden_size: int = 128  # mlp only
    n_oscillators: int = 64  # kuramoto only; half PolicyNet's hidden width
    kuramoto_steps: int = 10  # kuramoto only; see pymoto's kuramoto_cartpole
    kuramoto_g: float = 1.0  # kuramoto only
    kuramoto_k_scale: float = 1.0  # kuramoto only
    calibration_size: int = 1024  # kuramoto only
    lr: float = 1e-2
    gamma: float = 0.99
    episodes: int = 1000
    seed: int = 1
    solved_reward: float = 475.0
    log_every: int = 10


def parse_args() -> tuple[HParams, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = HParams()
    parser.add_argument("--policy", choices=["mlp", "kuramoto"], default=defaults.policy)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--n-oscillators", type=int, default=defaults.n_oscillators)
    parser.add_argument("--kuramoto-steps", type=int, default=defaults.kuramoto_steps)
    parser.add_argument("--kuramoto-g", type=float, default=defaults.kuramoto_g)
    parser.add_argument("--kuramoto-k-scale", type=float, default=defaults.kuramoto_k_scale)
    parser.add_argument("--calibration-size", type=int, default=defaults.calibration_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--solved-reward", type=float, default=defaults.solved_reward)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--wandb-project", default="cartpole-reinforce")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    hp = HParams(
        policy=args.policy,
        hidden_size=args.hidden_size,
        n_oscillators=args.n_oscillators,
        kuramoto_steps=args.kuramoto_steps,
        kuramoto_g=args.kuramoto_g,
        kuramoto_k_scale=args.kuramoto_k_scale,
        calibration_size=args.calibration_size,
        lr=args.lr,
        gamma=args.gamma,
        episodes=args.episodes,
        seed=args.seed,
        solved_reward=args.solved_reward,
        log_every=args.log_every,
    )
    return hp, args


def discount_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = torch.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return (returns - returns.mean()) / (returns.std() + 1e-8)


def collect_calibration_batch(seed: int, size: int) -> torch.Tensor:
    """Random-action rollout states, for the kuramoto policy's calibrate().

    CartPole-v1's observation_space bounds are not representative of visited
    states (velocity/angular-velocity bounds are ~3.4e38, since they're
    formally unbounded), so we can't just sample the space directly -- we have
    to actually roll the env out. Uses its own throwaway env/seed so it
    doesn't disturb the training env's seeded reset sequence.
    """
    calib_env = gym.make("CartPole-v1")
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


def build_policy(hp: HParams) -> torch.nn.Module:
    if hp.policy == "mlp":
        return PolicyNet(hidden_size=hp.hidden_size)

    policy = create_model(
        "kuramoto_cartpole",
        n=hp.n_oscillators,
        num_steps=hp.kuramoto_steps,
        k_scale=hp.kuramoto_k_scale,
    )
    x_cal = collect_calibration_batch(seed=hp.seed + 10_000, size=hp.calibration_size)
    calibrate(policy, x_cal, g=hp.kuramoto_g)
    return policy


def run_episode(env: gym.Env, policy: torch.nn.Module) -> tuple[torch.Tensor, list[float]]:
    log_probs = []
    rewards = []
    obs, _ = env.reset()
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = Categorical(logits=policy(obs_t).squeeze(0))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated
    return torch.stack(log_probs), rewards


def main() -> None:
    hp, args = parse_args()
    print(f"HParams: {asdict(hp)}")

    torch.manual_seed(hp.seed)
    env = gym.make("CartPole-v1")
    env.reset(seed=hp.seed)
    env.action_space.seed(hp.seed)

    run = wandb.init(project=args.wandb_project, mode=args.wandb_mode, config=asdict(hp))
    print(f"run={run.name}")

    policy = build_policy(hp)
    optimizer = optim.Adam(policy.parameters(), lr=hp.lr)
    # The random-K control in eval.py needs the exact initial coupling matrix.
    K_init = policy.get_coupling().K.detach().clone() if hp.policy == "kuramoto" else None

    running_reward = 0.0
    solved_at = None
    for episode in range(1, hp.episodes + 1):
        log_probs, rewards = run_episode(env, policy)
        episode_reward = sum(rewards)
        episode_length = len(rewards)
        running_reward = (
            episode_reward if episode == 1 else 0.05 * episode_reward + 0.95 * running_reward
        )

        returns = discount_returns(rewards, hp.gamma)
        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log(
            {
                "episode": episode,
                "train/episode_reward": episode_reward,
                "train/episode_length": episode_length,
                "train/running_reward": running_reward,
                "train/loss": loss.item(),
            },
            step=episode,
        )

        if episode % hp.log_every == 0:
            print(
                f"episode {episode:5d}  reward {episode_reward:6.1f}  "
                f"running_reward {running_reward:6.1f}  loss {loss.item():8.3f}"
            )

        if running_reward >= hp.solved_reward:
            solved_at = episode
            print(f"Solved at episode {episode} with running reward {running_reward:.1f}")
            break

    env.close()
    torch.save(
        {"hparams": asdict(hp), "state_dict": policy.state_dict(), "K_init": K_init},
        "checkpoint.pt",
    )
    print("Saved policy weights to checkpoint.pt")

    run.summary.update(
        {
            "solved_at_episode": solved_at,
            "final_running_reward": running_reward,
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
