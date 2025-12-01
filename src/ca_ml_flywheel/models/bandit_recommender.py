from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class ArmStats:
    """Simple stats for a single bandit arm."""
    arm_id: str
    successes: int = 0  # e.g., clicks
    failures: int = 0   # e.g., non-click impressions

    @property
    def alpha(self) -> float:
        # Beta prior parameters
        return 1.0 + self.successes

    @property
    def beta(self) -> float:
        return 1.0 + self.failures


class DealerBandit:
    """
    Very simple contextual bandit demo.

    Context: dealer_id (and optional metadata)
    Arms: layout or recommendation strategies like "A", "B", "C".
    Learning: Thompson Sampling on per-dealer arm stats.
    """

    def __init__(self, arms: List[str] | None = None):
        self.arms = arms or ["layout_a", "layout_b", "layout_c"]
        # state: dealer_id -> arm_id -> ArmStats
        self.state: Dict[str, Dict[str, ArmStats]] = {}

    def _init_dealer(self, dealer_id: str):
        if dealer_id not in self.state:
            self.state[dealer_id] = {
                arm_id: ArmStats(arm_id=arm_id) for arm_id in self.arms
            }

    def select_arm(self, dealer_id: str) -> str:
        """
        Thompson Sampling:
        Sample from Beta(alpha, beta) for each arm and select the max.
        """
        self._init_dealer(dealer_id)
        arm_stats = self.state[dealer_id]

        samples: List[Tuple[str, float]] = []
        rng = np.random.default_rng()

        for arm_id, stats in arm_stats.items():
            sample = rng.beta(stats.alpha, stats.beta)
            samples.append((arm_id, sample))

        # pick arm with max sampled value
        best_arm, _ = max(samples, key=lambda x: x[1])
        return best_arm

    def update(
        self,
        dealer_id: str,
        arm_id: str,
        reward: int,
    ) -> None:
        """
        Update stats based on observed reward.

        reward = 1 for success (e.g., click)
        reward = 0 for no click
        """
        self._init_dealer(dealer_id)
        if arm_id not in self.state[dealer_id]:
            self.state[dealer_id][arm_id] = ArmStats(arm_id=arm_id)

        stats = self.state[dealer_id][arm_id]
        if reward == 1:
            stats.successes += 1
        else:
            stats.failures += 1

    @classmethod
    def from_historical_data(
        cls,
        path: Path | None = None,
        arms: List[str] | None = None,
    ) -> "DealerBandit":
        """
        Construct a bandit instance seeded from historical data.

        Expected CSV schema (example):

        dealer_id,arm_id,clicks,impressions
        D001,layout_a,120,300
        D001,layout_b,50,180
        D002,layout_a,10,40
        ...
        """
        if path is None:
            path = DATA_PROCESSED_DIR / "dealer_bandit_context.csv"

        df = pd.read_csv(path)

        if arms is None:
            arms = sorted(df["arm_id"].unique().tolist())

        bandit = cls(arms=arms)

        for (dealer_id, arm_id), group in df.groupby(["dealer_id", "arm_id"]):
            clicks = int(group["clicks"].sum())
            impressions = int(group["impressions"].sum())
            failures = max(impressions - clicks, 0)

            bandit._init_dealer(dealer_id)
            bandit.state[dealer_id][arm_id] = ArmStats(
                arm_id=arm_id,
                successes=clicks,
                failures=failures,
            )

        return bandit
