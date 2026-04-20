"""
allocator.py — Core allocation inference wrapper.

Wraps the trained RL policy for production-style request handling,
implementing the accept/reject threshold from the DEMO.md sequence diagram.
"""

import time
import numpy as np
from dataclasses import dataclass
from src.core.inventory_graph import InventoryGraph


@dataclass
class AllocationRequest:
    """Incoming lease request."""
    duration: int
    start_day: int
    price_per_night: float
    day_of_week: int = 0


@dataclass
class AllocationResult:
    """Result of an allocation decision."""
    approved: bool
    unit_id: int | None
    revenue: float
    fragmentation_delta: float
    q_value: float
    latency_ms: float
    reason: str = ""


class Allocator:
    """Production-style allocator wrapping the RL policy.

    Implements the decision flow from DEMO.md:
    1. Receive request
    2. Fetch inventory state
    3. Pass to policy engine for Q-value estimation
    4. Compare Q-value against fragmentation threshold
    5. Approve or deny
    """

    def __init__(self, model, env, frag_threshold: float = 0.0):
        self.model = model
        self.env = env
        self.frag_threshold = frag_threshold

    def process_request(self, request: AllocationRequest) -> AllocationResult:
        """Process a single allocation request through the RL policy."""
        t0 = time.perf_counter()

        # Update the env's internal request
        self.env._request = {
            "duration": request.duration,
            "start_day": request.start_day,
            "price_per_night": request.price_per_night,
            "day_of_week": request.day_of_week,
        }

        obs = self.env._get_obs()

        # Get policy prediction
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        latency_ms = (time.perf_counter() - t0) * 1000

        if action < self.env.n_units:
            unit = action
            can_place = self.env._inventory.can_place(
                unit, request.start_day, request.duration
            )
            if can_place:
                pre_frag = self.env._inventory.fragmentation_score(unit)
                self.env._inventory.place_lease(
                    unit, request.start_day, request.duration, request.price_per_night
                )
                post_frag = self.env._inventory.fragmentation_score(unit)
                frag_delta = post_frag - pre_frag
                revenue = request.price_per_night * request.duration

                return AllocationResult(
                    approved=True,
                    unit_id=unit,
                    revenue=revenue,
                    fragmentation_delta=frag_delta,
                    q_value=revenue - self.env.frag_weight * frag_delta,
                    latency_ms=latency_ms,
                )
            else:
                return AllocationResult(
                    approved=False,
                    unit_id=None,
                    revenue=0.0,
                    fragmentation_delta=0.0,
                    q_value=0.0,
                    latency_ms=latency_ms,
                    reason="unit_unavailable",
                )
        else:
            return AllocationResult(
                approved=False,
                unit_id=None,
                revenue=0.0,
                fragmentation_delta=0.0,
                q_value=0.0,
                latency_ms=latency_ms,
                reason="policy_rejected",
            )
