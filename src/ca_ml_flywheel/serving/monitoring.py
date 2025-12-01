from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import logging
import statistics

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic logging config (safe if app has no global config yet)
    logging.basicConfig(level=logging.INFO)


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class PredictionEvent:
    """Lightweight record of a prediction for audit / debugging."""

    timestamp: datetime
    dealer_id: str
    region: str
    model_name: str
    model_version: Optional[str]
    approval_risk: float
    decision: str


@dataclass
class MonitoringState:
    """
    In-memory monitoring state.

    In a real deployment this would be replaced or complemented by:
    - Prometheus metrics
    - Datadog / New Relic / Azure Monitor
    - Structured logging to a data lake for analysis
    """

    total_requests: int = 0
    total_errors: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    approvals: int = 0
    reviews: int = 0

    last_events: List[PredictionEvent] = field(default_factory=list)
    max_events_kept: int = 200

    def record_prediction(
        self,
        dealer_id: str,
        region: str,
        model_name: str,
        model_version: Optional[str],
        approval_risk: float,
        decision: str,
        latency_ms: Optional[float] = None,
    ) -> None:
        self.total_requests += 1

        if decision.upper() == "APPROVE":
            self.approvals += 1
        elif decision.upper() == "REVIEW":
            self.reviews += 1

        if latency_ms is not None:
            self.latencies_ms.append(latency_ms)

        event = PredictionEvent(
            timestamp=datetime.utcnow(),
            dealer_id=dealer_id,
            region=region,
            model_name=model_name,
            model_version=model_version,
            approval_risk=approval_risk,
            decision=decision,
        )

        self.last_events.append(event)
        if len(self.last_events) > self.max_events_kept:
            # Keep only the most recent N events
            self.last_events = self.last_events[-self.max_events_kept :]

        logger.info(
            "prediction_event dealer_id=%s region=%s risk=%.4f decision=%s model=%s version=%s",
            dealer_id,
            region,
            approval_risk,
            decision,
            model_name,
            model_version,
        )

    def record_error(self) -> None:
        self.total_errors += 1

    def snapshot_metrics(self) -> Dict[str, Any]:
        """
        Return a simple snapshot of metrics suitable for a `/metrics` or `/health` endpoint.
        """
        latency_p50 = (
            statistics.median(self.latencies_ms) if self.latencies_ms else None
        )
        latency_p95 = (
            _percentile(self.latencies_ms, 95) if self.latencies_ms else None
        )

        total = self.total_requests or 1  # avoid div by zero
        approval_rate = self.approvals / total
        review_rate = self.reviews / total
        error_rate = self.total_errors / total

        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": error_rate,
            "approval_rate": approval_rate,
            "review_rate": review_rate,
            "latency_ms_p50": latency_p50,
            "latency_ms_p95": latency_p95,
        }

    def recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Return the last N events as plain dicts (safe to JSON-serialize).
        """
        events = self.last_events[-limit:]
        return [
            {
                "timestamp": e.timestamp.isoformat() + "Z",
                "dealer_id": e.dealer_id,
                "region": e.region,
                "model_name": e.model_name,
                "model_version": e.model_version,
                "approval_risk": e.approval_risk,
                "decision": e.decision,
            }
            for e in events
        ]


# Global in-memory state for the process
monitoring_state = MonitoringState()


# -----------------------------
# Drift / distribution helpers
# -----------------------------


def calculate_psi(expected: List[float], actual: List[float], buckets: int = 10) -> float:
    """
    Population Stability Index (PSI) for a single continuous feature.

    This is a very simplified implementation intended for demo purposes only.

    expected: baseline scores (e.g., from training)
    actual: current scores (e.g., from recent production window)
    """
    if not expected or not actual:
        return 0.0

    import numpy as np

    expected_arr = np.array(expected)
    actual_arr = np.array(actual)

    # Define bucket edges from expected distribution
    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = np.quantile(expected_arr, quantiles)

    # Avoid identical cut points
    cuts = np.unique(cuts)
    if len(cuts) <= 2:
        # Not enough variation for meaningful PSI
        return 0.0

    expected_hist, _ = np.histogram(expected_arr, bins=cuts)
    actual_hist, _ = np.histogram(actual_arr, bins=cuts)

    expected_frac = expected_hist / max(len(expected_arr), 1)
    actual_frac = actual_hist / max(len(actual_arr), 1)

    psi = 0.0
    for e, a in zip(expected_frac, actual_frac):
        if e == 0 or a == 0:
            continue
        psi += (a - e) * np.log(a / e)

    return float(psi)


# -----------------------------
# Internal helpers
# -----------------------------


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)
