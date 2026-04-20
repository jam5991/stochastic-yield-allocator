"""
inventory_env.py — Gymnasium environment for hybrid-duration inventory allocation.

Implements the modified Bellman equation:
    max_π E[Σ γ^t (R(s_t, a_t) - λ·Ω(s_{t+1}))]

Where R is the lease revenue and Ω is the fragmentation penalty.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.core.inventory_graph import InventoryGraph


class InventoryAllocationEnv(gym.Env):
    """Hybrid-duration inventory allocation environment.

    The agent manages a portfolio of rental units over a time horizon,
    deciding which unit to assign each incoming booking request to
    (or reject it entirely). The objective is to maximize total revenue
    while minimizing inventory fragmentation.

    Observation Space:
        - Per-unit occupancy rates (n_units,)
        - Per-unit fragmentation scores (n_units,)
        - Per-unit base prices, normalized (n_units,)
        - Request features: [duration_normalized, price_normalized, day_of_week_sin, day_of_week_cos]
        Total: 3 * n_units + 4

    Action Space:
        - Discrete(n_units + 1): assign to unit 0..n-1, or reject (n)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_units: int = 20,
        horizon: int = 90,
        base_prices: np.ndarray | None = None,
        duration_distribution: dict | None = None,
        fragmentation_weight: float = 0.15,
        max_requests_per_episode: int = 200,
        seed: int | None = None,
    ):
        super().__init__()

        self.n_units = n_units
        self.horizon = horizon
        self.frag_weight = fragmentation_weight
        self.max_requests = max_requests_per_episode

        # Price configuration (from real data or defaults)
        if base_prices is not None:
            self._base_prices = np.array(base_prices[:n_units], dtype=np.float32)
        else:
            self._base_prices = np.random.default_rng(seed).uniform(80, 400, size=n_units).astype(np.float32)

        self._price_scale = max(self._base_prices.max(), 1.0)

        # Duration distribution from real data
        self._dur_dist = duration_distribution or {
            "duration_mean": 3.5,
            "duration_std": 2.5,
            "price_mean": 180.0,
            "price_std": 100.0,
        }

        # Spaces
        obs_dim = 3 * n_units + 4
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_units + 1)

        # Internal state
        self._inventory: InventoryGraph | None = None
        self._current_day = 0
        self._request = None
        self._step_count = 0
        self._total_revenue = 0.0
        self._total_rejections = 0
        self._rng = np.random.default_rng(seed)

    def _generate_request(self) -> dict:
        """Generate a booking request from real-data-fitted distributions."""
        # Duration: log-normal approximation of real distribution
        raw_dur = self._rng.lognormal(
            mean=np.log(self._dur_dist["duration_mean"]),
            sigma=0.8,
        )
        duration = int(np.clip(raw_dur, 1, min(30, self.horizon - self._current_day)))

        # Price offered: based on real ADR with noise
        base_price = self._rng.normal(
            self._dur_dist["price_mean"],
            self._dur_dist["price_std"] * 0.3,
        )
        # Longer stays get a discount
        discount = 1.0 - 0.01 * max(0, duration - 3)
        price = max(30.0, base_price * discount)

        # Day of week (cyclical)
        dow = self._current_day % 7

        return {
            "duration": duration,
            "start_day": self._current_day,
            "price_per_night": float(price),
            "day_of_week": dow,
        }

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector from current state + request."""
        # Per-unit occupancy (fraction of days booked)
        occupancy = 1.0 - self._inventory.grid.mean(axis=1).astype(np.float32)

        # Per-unit fragmentation scores (normalized)
        frag = self._inventory.get_fragmentation_vector()
        frag_max = max(frag.max(), 1.0)
        frag_norm = frag / frag_max

        # Per-unit base prices (normalized)
        prices_norm = self._base_prices / self._price_scale

        # Request features
        req = self._request
        dur_norm = req["duration"] / 30.0
        price_norm = req["price_per_night"] / self._price_scale
        dow_sin = np.sin(2 * np.pi * req["day_of_week"] / 7.0)
        dow_cos = np.cos(2 * np.pi * req["day_of_week"] / 7.0)

        obs = np.concatenate([
            occupancy,
            frag_norm,
            prices_norm,
            [dur_norm, price_norm, dow_sin, dow_cos],
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._inventory = InventoryGraph(self.n_units, self.horizon)
        self._inventory.set_base_prices(self._base_prices)
        self._current_day = 0
        self._step_count = 0
        self._total_revenue = 0.0
        self._total_rejections = 0

        self._request = self._generate_request()
        obs = self._get_obs()

        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        info = {}
        req = self._request

        if action < self.n_units:
            # Attempt to place the booking at the selected unit
            unit = action
            if self._inventory.can_place(unit, req["start_day"], req["duration"]):
                # Compute pre-placement fragmentation
                pre_frag = self._inventory.fragmentation_score(unit)

                # Place the lease
                self._inventory.place_lease(
                    unit, req["start_day"], req["duration"], req["price_per_night"]
                )
                revenue = req["price_per_night"] * req["duration"]
                self._total_revenue += revenue

                # Compute post-placement fragmentation
                post_frag = self._inventory.fragmentation_score(unit)
                frag_delta = post_frag - pre_frag

                # Modified Bellman reward: R(s,a) - λ·Ω(s')
                reward = (revenue / self._price_scale) - self.frag_weight * frag_delta

                info["placed"] = True
                info["revenue"] = revenue
                info["unit"] = unit
                info["frag_delta"] = frag_delta
            else:
                # Invalid placement → small penalty
                reward = -0.5
                info["placed"] = False
                info["reason"] = "unit_unavailable"
        else:
            # Reject the request
            # Small opportunity cost penalty scaled by potential revenue
            opportunity_cost = req["price_per_night"] * req["duration"] / self._price_scale
            reward = -0.05 * opportunity_cost
            self._total_rejections += 1
            info["placed"] = False
            info["reason"] = "rejected"

        # Advance time (requests arrive ~2-5 per day)
        self._step_count += 1
        requests_per_day = self._rng.uniform(2, 5)
        self._current_day = min(
            int(self._step_count / requests_per_day),
            self.horizon - 1,
        )

        # Check termination
        terminated = self._current_day >= self.horizon - 1
        truncated = self._step_count >= self.max_requests

        if not (terminated or truncated):
            self._request = self._generate_request()
        else:
            info["total_revenue"] = self._total_revenue
            info["occupancy_rate"] = self._inventory.occupancy_rate()
            info["total_fragmentation"] = self._inventory.total_fragmentation()
            info["total_rejections"] = self._total_rejections

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info


def make_env(
    n_units: int = 20,
    horizon: int = 90,
    listings_df=None,
    distributions: dict | None = None,
    frag_weight: float = 0.15,
    seed: int | None = None,
) -> InventoryAllocationEnv:
    """Factory function to create an environment from real data."""
    base_prices = None
    if listings_df is not None:
        # Sample real prices from listings
        rng = np.random.default_rng(seed)
        sample = listings_df.sample(n=min(n_units, len(listings_df)), random_state=seed or 42)
        base_prices = sample["price"].values

    return InventoryAllocationEnv(
        n_units=n_units,
        horizon=horizon,
        base_prices=base_prices,
        duration_distribution=distributions,
        fragmentation_weight=frag_weight,
        seed=seed,
    )
