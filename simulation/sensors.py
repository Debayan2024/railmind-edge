"""
RailMind Edge — Sensor Simulation

Generates realistic sensor readings for each block section as a
parallel SimPy process. Sensor values correlate with track health
and train presence, providing ground-truth-labeled training data
for the Track Health Agent.

Sensors modeled:
  - Vibration (g)      — spikes when train passes, elevated on degraded track
  - Temperature (°C)   — rises with friction, anomalous on degraded track
  - Acoustic (dB)      — loud when train passes, unusual patterns on cracks
"""

import simpy
import numpy as np
from typing import Dict

from simulation.config import (
    BLOCKS, SENSOR_POLL_INTERVAL_MIN, HEALTH_DEGRADATION_RATE,
    SENSOR_NOISE_STD,
)
from simulation.network import RailwayNetwork, BlockSection
from simulation.data_logger import DataLogger


# ─────────────── Sensor Baselines ───────────────

VIBRATION_BASELINE = 0.8       # g (gravitational acceleration)
VIBRATION_TRAIN_BOOST = 3.5    # additional g when train is passing
VIBRATION_DEGRADE_FACTOR = 2.0 # additional g at health=0

TEMP_AMBIENT = 30.0            # °C baseline ambient
TEMP_TRAIN_BOOST = 10.0        # °C rise from friction when train passes
TEMP_DEGRADE_FACTOR = 15.0     # °C additional at health=0

ACOUSTIC_BASELINE = 45.0       # dB ambient noise
ACOUSTIC_TRAIN_BOOST = 40.0    # dB when train passes
ACOUSTIC_DEGRADE_FACTOR = 20.0 # dB from cracking/unusual noise at health=0


def _add_noise(value: float, rng: np.random.Generator) -> float:
    """Add Gaussian noise proportional to the value."""
    noise = rng.normal(0, max(abs(value) * SENSOR_NOISE_STD, 0.01))
    return value + noise


class SensorSystem:
    """
    Runs as a SimPy process, periodically sampling sensors for every block.
    Each block's health degrades linearly over time (with the option for
    sudden drops from the anomaly injector).
    """

    def __init__(
        self,
        env: simpy.Environment,
        network: RailwayNetwork,
        logger: DataLogger,
        seed: int = 42,
    ):
        self.env = env
        self.network = network
        self.logger = logger
        self.rng = np.random.default_rng(seed)

        # Per-block baseline variations (so blocks have slightly different normals)
        self._block_offsets: Dict[str, Dict[str, float]] = {}
        for block_id in BLOCKS:
            self._block_offsets[block_id] = {
                "vibration": self.rng.uniform(-0.2, 0.2),
                "temperature": self.rng.uniform(-3.0, 3.0),
                "acoustic": self.rng.uniform(-5.0, 5.0),
            }

    def start(self):
        """Start the sensor polling process."""
        self.env.process(self._poll_loop())
        print(f"  Sensors: polling every {SENSOR_POLL_INTERVAL_MIN} min "
              f"across {len(BLOCKS)} blocks")

    def _poll_loop(self):
        """Main sensor polling loop — runs for the entire simulation."""
        while True:
            self._sample_all_blocks()
            yield self.env.timeout(SENSOR_POLL_INTERVAL_MIN)

    def _sample_all_blocks(self):
        """Take sensor readings from every block section."""
        for block_id, block in self.network.blocks.items():
            # Degrade health over time
            block.health_score = max(
                0.0,
                block.health_score - HEALTH_DEGRADATION_RATE * SENSOR_POLL_INTERVAL_MIN
            )

            health = block.health_score
            degradation = max(0.0, 1.0 - health / 100.0)  # 0.0 (healthy) → 1.0 (failed)
            train_present = block.is_occupied
            offsets = self._block_offsets[block_id]

            # ── Vibration ──
            vibration = VIBRATION_BASELINE + offsets["vibration"]
            if train_present:
                vibration += VIBRATION_TRAIN_BOOST * self.rng.uniform(0.8, 1.2)
            vibration += degradation * VIBRATION_DEGRADE_FACTOR
            vibration = _add_noise(vibration, self.rng)
            vibration = max(0.0, vibration)

            # ── Temperature ──
            temperature = TEMP_AMBIENT + offsets["temperature"]
            if train_present:
                temperature += TEMP_TRAIN_BOOST * self.rng.uniform(0.7, 1.3)
            temperature += degradation * TEMP_DEGRADE_FACTOR
            temperature = _add_noise(temperature, self.rng)

            # ── Acoustic ──
            acoustic = ACOUSTIC_BASELINE + offsets["acoustic"]
            if train_present:
                acoustic += ACOUSTIC_TRAIN_BOOST * self.rng.uniform(0.8, 1.2)
            acoustic += degradation * ACOUSTIC_DEGRADE_FACTOR
            acoustic = _add_noise(acoustic, self.rng)
            acoustic = max(0.0, acoustic)

            self.logger.log_sensor_reading(
                timestamp=self.env.now,
                block_id=block_id,
                vibration=vibration,
                temperature=temperature,
                acoustic=acoustic,
                track_health_score=health,
                train_present=train_present,
            )
