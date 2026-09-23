"""Evaluate a trained CartPole policy and print summary metrics to the console.

    python eval.py --episodes 100
    python eval.py --render
"""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym
import torch

from model import KuramotoPolicy, PolicyNet

MAX_EPISODE_STEPS = 500  # CartPole-v1's per-episode cap; also the "solved" reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_policy(hp: dict) -> torch.nn.Module:
    if hp["policy"] == "mlp":
        return PolicyNet(hidden_size=hp["hidden_size"])
    return KuramotoPolicy(
        n=hp["n_oscillators"],
        num_steps=hp["kuramoto_steps"],
        k_scale=hp["kuramoto_k_scale"],
    )


def rollout(env: gym.Env, policy: torch.nn.Module) -> float:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = torch.argmax(policy(obs_t).squeeze(0)).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint)
    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    policy = build_policy(checkpoint["hparams"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    rewards = [rollout(env, policy) for _ in range(args.episodes)]
    env.close()

    for i, r in enumerate(rewards, start=1):
        print(f"episode {i:4d}  reward {r:6.1f}")

    n_solved = sum(r >= MAX_EPISODE_STEPS for r in rewards)
    print()
    print(f"episodes:        {args.episodes}")
    print(f"mean reward:     {statistics.mean(rewards):.1f}")
    print(f"median reward:   {statistics.median(rewards):.1f}")
    print(f"std reward:      {statistics.pstdev(rewards):.1f}")
    print(f"min / max:       {min(rewards):.1f} / {max(rewards):.1f}")
    print(f"success rate:    {n_solved / args.episodes:.1%}  (reward >= {MAX_EPISODE_STEPS})")


if __name__ == "__main__":
    main()
