"""Evaluate a trained InvertedDoublePendulum policy and print summary metrics.

    python eval.py --episodes 20
    python eval.py --render
    python eval.py --controls        # kuramoto only: the controls, scored by reward
"""

from __future__ import annotations

import argparse
import statistics

import gymnasium as gym
import torch

from pymoto import create_model
from pymoto.controls import with_coupling, with_num_steps, with_solver
from pymoto.layers import rk4_step

from model import GaussianPolicy, PolicyNet

ENV_ID = "InvertedDoublePendulum-v5"
OBS_DIM = 9
SOLVED_REWARD = 9100.0  # env's reward_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controls", action="store_true",
                        help="also score the num_steps=0, random-K and RK4 variants")
    parser.add_argument("--rk4-refine", type=int, default=10)
    return parser.parse_args()


def build_mean_net(hp: dict) -> torch.nn.Module:
    if hp["policy"] == "mlp":
        return PolicyNet(obs_dim=OBS_DIM, hidden_size=hp["hidden_size"])
    return create_model(
        "kuramoto_inverted_double_pendulum",
        n=hp["n_oscillators"],
        num_steps=hp["kuramoto_steps"],
        k_scale=hp["kuramoto_k_scale"],
        # REINFORCE checkpoints (train.py) predate these flags; default False matches
        # their architecture (W, H frozen).
        trainable_drive=hp.get("trainable_drive", False),
        trainable_head=hp.get("trainable_head", False),
    )


def rollout(env: gym.Env, policy: GaussianPolicy) -> float:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean, _ = policy(obs_t)
            action = torch.clamp(mean.squeeze(0), -1.0, 1.0)
        obs, reward, terminated, truncated, _ = env.step(action.numpy())
        total_reward += reward
        done = terminated or truncated
    return total_reward


def initial_coupling(checkpoint: dict) -> torch.Tensor:
    """K at init, for the random-K control.

    Checkpoints from train.py store it. Older ones do not, but it is exactly
    recoverable: train.py seeds torch's global RNG and the mean-net's K is the
    first thing drawn from it (gym, the calibration rollout and wandb do not
    touch that RNG until after the policy is built).
    """
    if checkpoint.get("K_init") is not None:
        return checkpoint["K_init"]
    torch.manual_seed(checkpoint["hparams"]["seed"])
    return build_mean_net(checkpoint["hparams"]).get_coupling().K.detach().clone()


def run_controls(env: gym.Env, policy: GaussianPolicy, K_init: torch.Tensor,
                 episodes: int, rk4_refine: int) -> None:
    """Mean greedy reward of each control variant; see pymoto.controls.

    Each control rewraps the varied mean-net in a fresh GaussianPolicy sharing
    the trained log_std, since only the mean-net (not the Gaussian wrapper)
    has a coupling matrix or a num_steps to vary.
    """
    mean_net = policy.mean_net
    num_steps = mean_net.config.num_steps
    variants = [
        ("num_steps = 0", with_num_steps(mean_net, 0), "input severed: one constant action"),
        ("random K (init)", with_coupling(mean_net, K_init), "reservoir baseline"),
        ("trained K", mean_net, ""),
        (f"RK4 @ {num_steps * rk4_refine} steps", with_solver(mean_net, rk4_step, rk4_refine),
         f"solver transfer from Euler @ {num_steps}"),
    ]
    print(f"\nControls (mean reward over {episodes} episodes)")
    for label, variant_net, note in variants:
        variant = GaussianPolicy(variant_net, init_log_std=0.0)
        variant.log_std.data.copy_(policy.log_std.data)
        variant.eval()
        mean = statistics.mean(rollout(env, variant) for _ in range(episodes))
        print(f"{label.ljust(18)}  {mean:8.1f}  {note}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint)
    env = gym.make(ENV_ID, render_mode="human" if args.render else None)
    mean_net = build_mean_net(checkpoint["hparams"])
    policy = GaussianPolicy(mean_net, init_log_std=checkpoint["hparams"]["init_log_std"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    rewards = [rollout(env, policy) for _ in range(args.episodes)]

    for i, r in enumerate(rewards, start=1):
        print(f"episode {i:4d}  reward {r:8.1f}")

    n_solved = sum(r >= SOLVED_REWARD for r in rewards)
    print()
    print(f"episodes:        {args.episodes}")
    print(f"mean reward:     {statistics.mean(rewards):.1f}")
    print(f"median reward:   {statistics.median(rewards):.1f}")
    print(f"std reward:      {statistics.pstdev(rewards):.1f}")
    print(f"min / max:       {min(rewards):.1f} / {max(rewards):.1f}")
    print(f"success rate:    {n_solved / args.episodes:.1%}  (reward >= {SOLVED_REWARD})")

    if args.controls:
        if checkpoint["hparams"]["policy"] != "kuramoto":
            raise SystemExit("--controls applies only to the kuramoto policy")
        run_controls(env, policy, initial_coupling(checkpoint), args.episodes, args.rk4_refine)
    env.close()


if __name__ == "__main__":
    main()
