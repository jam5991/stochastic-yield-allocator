"""
baselines.py — Baseline allocation policies for comparison.

Implements greedy, random, and first-fit strategies to benchmark
against the trained RL policy.
"""

import numpy as np


class BaselinePolicy:
    """Base class for deterministic baseline policies."""

    def __init__(self, name: str):
        self.name = name

    def predict(self, obs: np.ndarray, env) -> int:
        raise NotImplementedError


class GreedyPolicy(BaselinePolicy):
    """Greedy: Always accept, assign to the unit that maximizes immediate revenue.

    This is the 'naive revenue management' approach that ignores fragmentation.
    """

    def __init__(self):
        super().__init__("Greedy (Rules-Based)")

    def predict(self, obs: np.ndarray, env) -> int:
        req = env._request
        best_unit = -1
        best_score = -float("inf")

        for unit in range(env.n_units):
            if env._inventory.can_place(unit, req["start_day"], req["duration"]):
                # Greedy: maximize immediate revenue, ignore fragmentation
                score = req["price_per_night"] * req["duration"]
                # Slight preference for units with higher base price (revenue alignment)
                score += env._base_prices[unit] * 0.01
                if score > best_score:
                    best_score = score
                    best_unit = unit

        if best_unit >= 0:
            return best_unit
        else:
            return env.n_units  # Reject (no availability)


class RandomPolicy(BaselinePolicy):
    """Random: Accept if possible, assign to a random available unit."""

    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self._rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, env) -> int:
        req = env._request
        available_units = [
            u for u in range(env.n_units)
            if env._inventory.can_place(u, req["start_day"], req["duration"])
        ]
        if available_units:
            return self._rng.choice(available_units)
        else:
            return env.n_units  # Reject


class FirstFitPolicy(BaselinePolicy):
    """First-Fit: Accept if possible, assign to the first available unit."""

    def __init__(self):
        super().__init__("First-Fit")

    def predict(self, obs: np.ndarray, env) -> int:
        req = env._request
        for unit in range(env.n_units):
            if env._inventory.can_place(unit, req["start_day"], req["duration"]):
                return unit
        return env.n_units  # Reject


def evaluate_baseline(
    policy: BaselinePolicy,
    env,
    n_episodes: int = 20,
    seed: int = 42,
) -> dict:
    """Evaluate a baseline policy over multiple episodes.

    Returns averaged metrics including revenue, occupancy, fragmentation,
    and rejection rate.
    """
    revenues = []
    occupancies = []
    fragmentations = []
    rejection_rates = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_steps = 0
        done = False

        while not done:
            action = policy.predict(obs, env)
            obs, _, terminated, truncated, info = env.step(action)
            total_steps += 1
            done = terminated or truncated

        revenues.append(info.get("total_revenue", 0.0))
        occupancies.append(info.get("occupancy_rate", 0.0))
        fragmentations.append(info.get("total_fragmentation", 0.0))
        rejection_rates.append(
            info.get("total_rejections", 0) / max(total_steps, 1)
        )

    return {
        "policy_name": policy.name,
        "mean_revenue": float(np.mean(revenues)),
        "std_revenue": float(np.std(revenues)),
        "mean_occupancy": float(np.mean(occupancies)),
        "mean_fragmentation": float(np.mean(fragmentations)),
        "mean_rejection_rate": float(np.mean(rejection_rates)),
        "revpar": float(np.mean(revenues)) / (env.n_units * env.horizon),
    }
