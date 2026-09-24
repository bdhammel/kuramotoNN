"""Evaluate a trained CartPole policy and print summary metrics to the console.

    python eval.py --episodes 100
    python eval.py --render
    python eval.py --controls        # kuramoto only: the controls, scored by reward

Ends with the energy-per-inference estimate from pymoto.energy; a Kuramoto policy
is compared against the MLP policy's architecture.
"""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym
import torch

from pymoto import checkpoint_filter_fn, create_model
from pymoto.controls import with_coupling, with_num_steps, with_solver
from pymoto.energy import add_energy_args, report_from_args
from pymoto.layers import rk4_step

from model import PolicyNet

MAX_EPISODE_STEPS = 500  # CartPole-v1's per-episode cap; also the "solved" reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controls", action="store_true",
                        help="also score the num_steps=0, random-K and RK4 variants")
    parser.add_argument("--rk4-refine", type=int, default=10)
    add_energy_args(parser)
    return parser.parse_args()


def energy_baseline(hp: dict) -> tuple[int, int, int] | None:
    """The MLP policy's shape, as the energy baseline for a Kuramoto policy; None for the MLP itself."""
    if hp["policy"] != "kuramoto":
        return None
    return (4, hp.get("hidden_size", 128), 2)


def build_policy(hp: dict) -> torch.nn.Module:
    if hp["policy"] == "mlp":
        return PolicyNet(hidden_size=hp["hidden_size"])
    return create_model(
        "kuramoto_cartpole",
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


def initial_coupling(checkpoint: dict) -> torch.Tensor:
    """K at init, for the random-K control.

    Checkpoints from train.py store it. Older ones do not, but it is exactly
    recoverable: train.py seeds torch's global RNG and the policy's K is the first
    thing drawn from it (gym and wandb do not touch that RNG).
    """
    if checkpoint.get("K_init") is not None:
        return checkpoint["K_init"]
    torch.manual_seed(checkpoint["hparams"]["seed"])
    return build_policy(checkpoint["hparams"]).get_coupling().K.detach().clone()


def run_controls(env: gym.Env, policy: torch.nn.Module, K_init: torch.Tensor,
                 episodes: int, rk4_refine: int) -> None:
    """Mean greedy reward of each control variant; see pymoto.controls."""
    num_steps = policy.config.num_steps
    variants = [
        ("num_steps = 0", with_num_steps(policy, 0), "input severed: one constant action"),
        ("random K (init)", with_coupling(policy, K_init), "reservoir baseline"),
        ("trained K", policy, ""),
        (f"RK4 @ {num_steps * rk4_refine} steps", with_solver(policy, rk4_step, rk4_refine),
         f"solver transfer from Euler @ {num_steps}"),
    ]
    print(f"\nControls (mean reward over {episodes} episodes)")
    for label, variant, note in variants:
        variant.eval()
        mean = statistics.mean(rollout(env, variant) for _ in range(episodes))
        print(f"{label.ljust(18)}  {mean:6.1f}  {note}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint)
    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    policy = build_policy(checkpoint["hparams"])
    state_dict = checkpoint["state_dict"]
    if checkpoint["hparams"]["policy"] == "kuramoto":
        # Pre-pymoto checkpoints store K, W, H, ... flat; remap to the module tree.
        state_dict = checkpoint_filter_fn(state_dict)
    policy.load_state_dict(state_dict)
    policy.eval()

    rewards = [rollout(env, policy) for _ in range(args.episodes)]

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

    if args.controls:
        if checkpoint["hparams"]["policy"] != "kuramoto":
            raise SystemExit("--controls applies only to the kuramoto policy")
        run_controls(env, policy, initial_coupling(checkpoint), args.episodes, args.rk4_refine)
    env.close()

    print()
    print(report_from_args(policy, args, default_baseline=energy_baseline(checkpoint["hparams"]))[1])


if __name__ == "__main__":
    main()
