"""
RailMind Edge — Simulation Package

Discrete-event simulation of an Indian railway corridor using SimPy.
Generates synthetic data for training the Track Health and Signal Anomaly agents.
"""

from simulation.network import RailwayNetwork, BlockSection, Signal
from simulation.train import TrainDispatcher, train_process
from simulation.sensors import SensorSystem
from simulation.anomaly_injector import AnomalyInjector
from simulation.data_logger import DataLogger
from simulation.run_simulation import run_simulation

__all__ = [
    "RailwayNetwork",
    "BlockSection",
    "Signal",
    "TrainDispatcher",
    "train_process",
    "SensorSystem",
    "AnomalyInjector",
    "DataLogger",
    "run_simulation",
]
