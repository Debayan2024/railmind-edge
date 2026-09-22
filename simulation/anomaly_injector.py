"""
RailMind Edge — Anomaly Injector

Injects realistic anomalies into the running simulation at scheduled
or random times. Each anomaly is logged with ground-truth labels
for training the Signal Anomaly and Track Health agents.

Anomaly types:
  1. Track degradation   — sudden health-score drop on a block
  2. Signal stuck on RED — a signal stops responding (fails safe)
  3. Signal wrong aspect — a signal shows GREEN when it should be RED (dangerous)
  4. Interlocking failure — (logged only, not physically simulated, for training data)
"""

import simpy
import numpy as np
from typing import List, Optional

from simulation.config import (
    BLOCKS, IncidentType, Severity,
    MIN_ANOMALIES, MAX_ANOMALIES, SIM_DURATION_MIN,
    SignalAspect,
)
from simulation.network import RailwayNetwork, Signal
from simulation.data_logger import DataLogger


class AnomalyInjector:
    """
    Schedules and injects anomalies throughout the simulation.

    Anomalies are pre-scheduled at init time (random times within the
    simulation duration) and executed as SimPy processes.
    """

    def __init__(
        self,
        env: simpy.Environment,
        network: RailwayNetwork,
        logger: DataLogger,
        seed: int = 123,
    ):
        self.env = env
        self.network = network
        self.logger = logger
        self.rng = np.random.default_rng(seed)
        self._schedule: List[dict] = []

    def start(self):
        """Generate anomaly schedule and start the injection process."""
        self._generate_schedule()
        self.env.process(self._run())
        print(f"  Anomaly Injector: {len(self._schedule)} anomalies scheduled")

    def _generate_schedule(self):
        """Create a randomized anomaly schedule."""
        num_anomalies = self.rng.integers(MIN_ANOMALIES, MAX_ANOMALIES + 1)
        block_ids = list(BLOCKS.keys())

        # Distribute anomalies across the simulation timeline
        times = sorted(self.rng.uniform(60, SIM_DURATION_MIN - 60, size=num_anomalies))

        anomaly_types = [
            IncidentType.TRACK_DEGRADATION,
            IncidentType.SIGNAL_STUCK_RED,
            IncidentType.SIGNAL_WRONG_ASPECT,
            IncidentType.INTERLOCKING_FAILURE,
        ]
        # Weight towards track degradation (most common real-world fault)
        weights = [0.40, 0.25, 0.20, 0.15]

        for t in times:
            anomaly_type = self.rng.choice(anomaly_types, p=weights)
            target_block = self.rng.choice(block_ids)
            self._schedule.append({
                "time": float(t),
                "type": anomaly_type,
                "block_id": target_block,
            })

    def _run(self):
        """Execute anomalies at their scheduled times."""
        for entry in self._schedule:
            # Wait until the scheduled time
            wait_time = entry["time"] - self.env.now
            if wait_time > 0:
                yield self.env.timeout(wait_time)

            anomaly_type = entry["type"]
            block_id = entry["block_id"]

            if anomaly_type == IncidentType.TRACK_DEGRADATION:
                self._inject_track_degradation(block_id)
            elif anomaly_type == IncidentType.SIGNAL_STUCK_RED:
                self._inject_signal_stuck_red(block_id)
            elif anomaly_type == IncidentType.SIGNAL_WRONG_ASPECT:
                self._inject_signal_wrong_aspect(block_id)
            elif anomaly_type == IncidentType.INTERLOCKING_FAILURE:
                self._inject_interlocking_failure(block_id)

    # ─────────────── Anomaly Implementations ───────────────

    def _inject_track_degradation(self, block_id: str):
        """Sudden track health drop — simulates rail fracture, waterlogging, etc."""
        block = self.network.blocks[block_id]
        drop = float(self.rng.uniform(15, 40))
        old_health = block.health_score
        block.health_score = max(0.0, block.health_score - drop)

        severity = Severity.CRITICAL if block.health_score < 30 else (
            Severity.HIGH if block.health_score < 50 else Severity.MEDIUM
        )

        self.logger.log_incident(
            self.env.now,
            IncidentType.TRACK_DEGRADATION.value,
            block_id,
            severity.value,
            f"Track health dropped from {old_health:.1f} to {block.health_score:.1f} "
            f"(delta={drop:.1f}). Possible rail fracture or subsurface defect.",
        )

    def _inject_signal_stuck_red(self, block_id: str):
        """Signal fails safe — stuck on RED even when block is clear."""
        # Find a signal that guards this block
        signal = self._find_signal_for_block(block_id)
        if signal is None:
            return

        signal.force_aspect(SignalAspect.RED)

        self.logger.log_incident(
            self.env.now,
            IncidentType.SIGNAL_STUCK_RED.value,
            block_id,
            Severity.MEDIUM.value,
            f"Signal {signal.signal_id} stuck on RED. "
            f"Fail-safe mode - trains will be delayed.",
        )

        # Auto-clear after 15-45 minutes (simulates maintenance response)
        clear_time = float(self.rng.uniform(15, 45))
        self.env.process(self._clear_signal_fault(signal, clear_time))

    def _inject_signal_wrong_aspect(self, block_id: str):
        """DANGEROUS: Signal shows GREEN when it should show RED."""
        signal = self._find_signal_for_block(block_id)
        if signal is None:
            return

        signal.force_aspect(SignalAspect.GREEN)

        self.logger.log_incident(
            self.env.now,
            IncidentType.SIGNAL_WRONG_ASPECT.value,
            block_id,
            Severity.CRITICAL.value,
            f"Signal {signal.signal_id} showing GREEN regardless of block state. "
            f"DANGEROUS - potential for collision if block is occupied.",
        )

        # Auto-clear after 5-20 minutes (critical, faster response)
        clear_time = float(self.rng.uniform(5, 20))
        self.env.process(self._clear_signal_fault(signal, clear_time))

    def _inject_interlocking_failure(self, block_id: str):
        """Log an interlocking failure event (documentation-only for training data)."""
        self.logger.log_incident(
            self.env.now,
            IncidentType.INTERLOCKING_FAILURE.value,
            "BHN",  # Always at junction
            Severity.CRITICAL.value,
            f"Interlocking failure detected at Bhilai Nagar Junction. "
            f"Conflicting routes may be set simultaneously. "
            f"Triggered near block {block_id}.",
        )

    # ─────────────── Helpers ───────────────

    def _find_signal_for_block(self, block_id: str) -> Optional[Signal]:
        """Find any signal that guards entry to the given block."""
        for sig_id, signal in self.network.signals.items():
            if signal.guarded_block.block_id == block_id:
                return signal
        return None

    def _clear_signal_fault(self, signal: Signal, delay_min: float):
        """SimPy process to clear a signal fault after a delay."""
        yield self.env.timeout(delay_min)
        signal.force_aspect(None)  # Clear override, return to normal
