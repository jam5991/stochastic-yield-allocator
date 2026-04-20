"""
inventory_graph.py — Temporal inventory graph for unit × day allocation.

Models inventory as a 2D availability grid (units × days) and provides
efficient contiguous availability queries and fragmentation scoring.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class InventoryGraph:
    """Temporal inventory graph tracking availability for N units over T days.

    Attributes:
        n_units: Number of rental units in the portfolio.
        horizon: Number of days in the planning horizon.
        grid: Boolean array [n_units, horizon] — True = available.
        revenue_grid: Float array [n_units, horizon] — daily revenue if booked.
    """

    n_units: int
    horizon: int
    grid: np.ndarray = field(init=False)
    revenue_grid: np.ndarray = field(init=False)
    base_prices: np.ndarray = field(init=False)

    def __post_init__(self):
        # All units start fully available
        self.grid = np.ones((self.n_units, self.horizon), dtype=bool)
        self.revenue_grid = np.zeros((self.n_units, self.horizon), dtype=np.float32)
        self.base_prices = np.zeros(self.n_units, dtype=np.float32)

    def set_base_prices(self, prices: np.ndarray):
        """Set the base nightly price for each unit (from real data)."""
        self.base_prices = prices.astype(np.float32)

    def can_place(self, unit: int, start_day: int, duration: int) -> bool:
        """Check if a lease can be placed at unit/start/duration."""
        end = start_day + duration
        if end > self.horizon or start_day < 0 or unit < 0 or unit >= self.n_units:
            return False
        return np.all(self.grid[unit, start_day:end])

    def place_lease(self, unit: int, start_day: int, duration: int, price_per_night: float):
        """Place a lease, marking days as occupied and recording revenue."""
        end = start_day + duration
        self.grid[unit, start_day:end] = False
        self.revenue_grid[unit, start_day:end] = price_per_night

    def get_contiguous_blocks(self, unit: int) -> list[int]:
        """Get lengths of all contiguous available blocks for a unit."""
        avail = self.grid[unit]
        blocks = []
        i = 0
        while i < len(avail):
            if avail[i]:
                start = i
                while i < len(avail) and avail[i]:
                    i += 1
                blocks.append(i - start)
            else:
                i += 1
        return blocks

    def fragmentation_score(self, unit: int, k_values: list[int] = None) -> float:
        """Compute the non-linear fragmentation penalty Ω for a unit.

        Ω = Σ (1 / block_length) for all available gaps
              + count of blocks that cannot serve any k-length lease

        Small gaps are penalized heavily (1/1 = 1.0 vs 1/30 = 0.03).
        """
        if k_values is None:
            k_values = [7, 14, 30]

        blocks = self.get_contiguous_blocks(unit)
        if not blocks:
            return 0.0  # Fully booked, no fragmentation

        # Base penalty: sum of inverse block lengths
        penalty = sum(1.0 / b for b in blocks)

        # Additional penalty for blocks too small for desired durations
        for k in k_values:
            unusable = sum(1 for b in blocks if b < k)
            penalty += 0.1 * unusable

        return penalty

    def total_fragmentation(self) -> float:
        """Average fragmentation across all units."""
        scores = [self.fragmentation_score(u) for u in range(self.n_units)]
        return float(np.mean(scores))

    def occupancy_rate(self) -> float:
        """Overall occupancy rate across all units and days."""
        return 1.0 - self.grid.mean()

    def total_revenue(self) -> float:
        """Total revenue from all placed leases."""
        return float(self.revenue_grid.sum())

    def get_unit_availability_vector(self, unit: int) -> np.ndarray:
        """Get the availability bitmap for a single unit."""
        return self.grid[unit].astype(np.float32)

    def get_state_tensor(self) -> np.ndarray:
        """Get the full state as a flat tensor for the RL observation."""
        return self.grid.astype(np.float32).flatten()

    def get_fragmentation_vector(self) -> np.ndarray:
        """Get per-unit fragmentation scores as a vector."""
        return np.array(
            [self.fragmentation_score(u) for u in range(self.n_units)],
            dtype=np.float32,
        )

    def reset(self):
        """Reset all units to fully available."""
        self.grid[:] = True
        self.revenue_grid[:] = 0.0

    def copy(self) -> "InventoryGraph":
        """Create a deep copy of the inventory state."""
        new = InventoryGraph(self.n_units, self.horizon)
        new.grid = self.grid.copy()
        new.revenue_grid = self.revenue_grid.copy()
        new.base_prices = self.base_prices.copy()
        return new
