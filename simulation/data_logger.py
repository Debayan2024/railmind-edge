"""
RailMind Edge — Data Logger

Centralized logging for all simulation events.
Collects signal changes, sensor readings, train movements, and incidents
into DataFrames, then writes CSVs to data/synthetic/.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pandas as pd


class DataLogger:
    """Collects all simulation events and writes them to CSV files."""

    def __init__(self, output_dir: str = "data/synthetic"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Event buffers
        self._signal_events: List[Dict[str, Any]] = []
        self._sensor_events: List[Dict[str, Any]] = []
        self._train_events: List[Dict[str, Any]] = []
        self._incident_events: List[Dict[str, Any]] = []

    # ─────────────── Signal Logging ───────────────

    def log_signal_event(
        self,
        timestamp: float,
        signal_id: str,
        block_id: str,
        aspect: str,
        previous_aspect: str,
        reason: str,
    ):
        """Log a signal aspect change."""
        self._signal_events.append({
            "timestamp": round(timestamp, 2),
            "signal_id": signal_id,
            "block_id": block_id,
            "aspect": aspect,
            "previous_aspect": previous_aspect,
            "reason": reason,
        })

    # ─────────────── Sensor Logging ───────────────

    def log_sensor_reading(
        self,
        timestamp: float,
        block_id: str,
        vibration: float,
        temperature: float,
        acoustic: float,
        track_health_score: float,
        train_present: bool,
    ):
        """Log a sensor reading for a block section."""
        self._sensor_events.append({
            "timestamp": round(timestamp, 2),
            "block_id": block_id,
            "vibration_g": round(vibration, 4),
            "temperature_c": round(temperature, 2),
            "acoustic_db": round(acoustic, 2),
            "track_health_score": round(track_health_score, 2),
            "train_present": train_present,
        })

    # ─────────────── Train Logging ────────────────

    def log_train_event(
        self,
        timestamp: float,
        train_id: str,
        train_type: str,
        event: str,
        location: str,
        speed_kmh: float = 0.0,
        delay_min: float = 0.0,
    ):
        """Log a train movement event."""
        self._train_events.append({
            "timestamp": round(timestamp, 2),
            "train_id": train_id,
            "train_type": train_type,
            "event": event,
            "location": location,
            "speed_kmh": round(speed_kmh, 1),
            "delay_min": round(delay_min, 2),
        })

    # ──────────────── Incident Logging ────────────────

    def log_incident(
        self,
        timestamp: float,
        incident_type: str,
        location: str,
        severity: str,
        description: str,
    ):
        """Log an injected anomaly / incident."""
        self._incident_events.append({
            "timestamp": round(timestamp, 2),
            "incident_type": incident_type,
            "location": location,
            "severity": severity,
            "description": description,
        })

    # ──────────────── CSV Export ────────────────

    def save_all(self):
        """Write all collected events to CSV files."""
        self._save_csv(self._signal_events, "signal_log.csv")
        self._save_csv(self._sensor_events, "sensor_log.csv")
        self._save_csv(self._train_events, "train_log.csv")
        self._save_csv(self._incident_events, "incident_log.csv")

    def _save_csv(self, events: List[Dict], filename: str):
        """Save a list of event dicts to a CSV file."""
        filepath = os.path.join(self.output_dir, filename)
        if events:
            df = pd.DataFrame(events)
            df.to_csv(filepath, index=False)
            print(f"  [OK] {filepath} -- {len(df)} rows")
        else:
            # Write empty file with headers
            pd.DataFrame().to_csv(filepath, index=False)
            print(f"  [OK] {filepath} -- 0 rows (empty)")

    # ──────────────── Summary ────────────────

    def print_summary(self):
        """Print a summary of collected events."""
        print("\n" + "=" * 50)
        print("DATA GENERATION SUMMARY")
        print("=" * 50)
        print(f"  Signal state changes : {len(self._signal_events)}")
        print(f"  Sensor readings      : {len(self._sensor_events)}")
        print(f"  Train events         : {len(self._train_events)}")
        print(f"  Incidents injected   : {len(self._incident_events)}")
        print("=" * 50)
