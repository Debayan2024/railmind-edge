"""
RailMind Edge — Simulation Runner

CLI entry point that assembles and runs the full railway simulation.
Creates the SimPy environment, builds the network, starts all
subsystems, and saves generated data to CSV files.
"""

import sys
import time
import argparse

import simpy
import numpy as np

from simulation.config import SIM_DURATION_MIN, RANDOM_SEED, TIMETABLE
from simulation.network import RailwayNetwork
from simulation.train import TrainDispatcher
from simulation.sensors import SensorSystem
from simulation.anomaly_injector import AnomalyInjector
from simulation.data_logger import DataLogger


def run_simulation(
    duration_min: int = SIM_DURATION_MIN,
    seed: int = RANDOM_SEED,
    output_dir: str = "data/synthetic",
    verbose: bool = True,
):
    """
    Run the full railway corridor simulation.

    Args:
        duration_min: Simulation duration in minutes.
        seed: Random seed for reproducibility.
        output_dir: Directory to write CSV log files.
        verbose: If True, print progress to stdout.
    """
    if verbose:
        print("=" * 60)
        print("  RAILMIND EDGE - SIMULATION ENGINE")
        print("=" * 60)
        print(f"\n  Duration     : {duration_min} min ({duration_min / 60:.0f} hours)")
        print(f"  Random seed  : {seed}")
        print(f"  Output dir   : {output_dir}")
        print(f"  Trains       : {len(TIMETABLE)}")
        print()

    # Set global seed
    np.random.seed(seed)

    # Create SimPy environment
    env = simpy.Environment()

    # Initialize subsystems
    if verbose:
        print("Initializing subsystems...")

    logger = DataLogger(output_dir=output_dir)
    network = RailwayNetwork(env, logger)

    if verbose:
        print(f"  Network: {len(network.blocks)} blocks, "
              f"{len(network.signals)} signals, "
              f"{len(network.stations)} stations")

    dispatcher = TrainDispatcher(env, network, logger)
    sensors = SensorSystem(env, network, logger, seed=seed)
    anomaly_injector = AnomalyInjector(env, network, logger, seed=seed + 1)

    # Start all subsystems
    dispatcher.start()
    sensors.start()
    anomaly_injector.start()

    # Run simulation
    if verbose:
        print(f"\nRunning simulation...")

    start_wall = time.time()
    env.run(until=duration_min)
    elapsed = time.time() - start_wall

    if verbose:
        print(f"  Simulation completed in {elapsed:.2f}s wall-clock time")

    # Save data
    if verbose:
        print(f"\nSaving data to {output_dir}/...")

    logger.save_all()
    logger.print_summary()

    if verbose:
        print(f"\n[OK] Phase 1 simulation complete. Data ready for agent training.")
        print("=" * 60)

    return logger


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RailMind Edge — Railway Corridor Simulation"
    )
    parser.add_argument(
        "--duration", type=int, default=SIM_DURATION_MIN,
        help=f"Simulation duration in minutes (default: {SIM_DURATION_MIN})"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--output", type=str, default="data/synthetic",
        help="Output directory for CSV files (default: data/synthetic)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()
    run_simulation(
        duration_min=args.duration,
        seed=args.seed,
        output_dir=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
