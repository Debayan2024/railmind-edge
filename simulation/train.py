"""
RailMind Edge — Train Process

SimPy processes that model individual trains traversing the railway network.
Each train follows its route, respects signals, acquires block sections,
and logs all events.
"""

import simpy
from typing import List, Optional

from simulation.config import (
    TrainSchedule, TrainType, SignalAspect,
    ROUTES, SEGMENTS, TIMETABLE,
)
from simulation.network import RailwayNetwork
from simulation.data_logger import DataLogger


# Speed restrictions
YELLOW_SPEED_LIMIT_KMH = 40.0   # Max speed when signal shows YELLOW
FREIGHT_SPEED_PENALTY = 0.9     # Freight trains run at 90% of their max


def _compute_expected_travel_time(schedule: TrainSchedule, network: RailwayNetwork) -> float:
    """Compute the ideal (no-delay) travel time for a train's full route, in minutes."""
    total_time = 0.0
    for i in range(len(schedule.route_stations) - 1):
        from_stn = schedule.route_stations[i]
        to_stn = schedule.route_stations[i + 1]
        block_ids = network.get_route_blocks(from_stn, to_stn)

        for bid in block_ids:
            block = network.blocks[bid]
            speed = min(schedule.max_speed_kmh, block.config.speed_limit_kmh)
            total_time += (block.config.length_km / speed) * 60  # km / (km/h) * 60 = min

        # Add dwell time at intermediate stations
        if i < len(schedule.route_stations) - 2:
            total_time += schedule.dwell_time_min

    return total_time


def train_process(
    env: simpy.Environment,
    schedule: TrainSchedule,
    network: RailwayNetwork,
    logger: DataLogger,
):
    """
    SimPy generator process for a single train.

    Flow for each segment (station-to-station):
      1. Acquire section lock (if single track)
      2. Acquire junction lock (if route passes through BHN junction)
      3. For each block in the segment:
         a. Check signal aspect
         b. Request block resource (waits if occupied → RED signal behaviour)
         c. Travel through block (timeout based on speed)
         d. Release block
      4. Release section lock
      5. Dwell at station
    """
    # Wait for scheduled departure
    yield env.timeout(schedule.departure_time_min)

    train_type_str = schedule.train_type.value
    expected_time = _compute_expected_travel_time(schedule, network)
    expected_arrival = schedule.departure_time_min + expected_time
    depart_time = env.now

    logger.log_train_event(
        env.now, schedule.train_id, train_type_str,
        "DEPARTED", schedule.route_stations[0],
        speed_kmh=schedule.max_speed_kmh,
    )

    # Traverse each station-to-station segment
    for seg_idx in range(len(schedule.route_stations) - 1):
        from_stn = schedule.route_stations[seg_idx]
        to_stn = schedule.route_stations[seg_idx + 1]

        block_ids = network.get_route_blocks(from_stn, to_stn)
        signal_chain = network.get_route_signal_chain(from_stn, to_stn)
        section_lock = network.get_section_lock(from_stn, to_stn)
        junction_needed = network.needs_junction(from_stn, to_stn)

        # ── Acquire section lock for single-track segments ──
        section_lock_req = None
        if section_lock:
            section_lock_req = section_lock.request()
            yield section_lock_req
            logger.log_train_event(
                env.now, schedule.train_id, train_type_str,
                "SECTION_LOCK_ACQUIRED", f"{from_stn}->{to_stn}",
            )

        # ── Acquire junction interlocking ──
        junction_req = None
        if junction_needed:
            junction_req = network.junction_lock.request()
            yield junction_req
            logger.log_train_event(
                env.now, schedule.train_id, train_type_str,
                "JUNCTION_LOCK_ACQUIRED", "BHN",
            )

        # ── Traverse blocks one by one ──
        for blk_idx, block_id in enumerate(block_ids):
            block = network.blocks[block_id]
            signal = signal_chain[blk_idx]

            # Check signal before requesting block
            signal_aspect = signal.aspect

            # Log signal check
            logger.log_train_event(
                env.now, schedule.train_id, train_type_str,
                f"SIGNAL_{signal_aspect.value}", signal.signal_id,
            )

            # If signal is RED, we'll wait on resource request
            if signal_aspect == SignalAspect.RED:
                logger.log_train_event(
                    env.now, schedule.train_id, train_type_str,
                    "WAITING_AT_RED", block_id,
                )

            # Request the block (SimPy queues us if occupied)
            block_req = block.resource.request()
            yield block_req

            # We now occupy the block
            block.set_occupied(schedule.train_id)

            # Release junction lock after entering first block of the segment
            # (points are set and locked, train is past the junction throat)
            if junction_needed and blk_idx == 0 and junction_req is not None:
                network.junction_lock.release(junction_req)
                junction_req = None

            # Compute actual speed
            speed = min(schedule.max_speed_kmh, block.config.speed_limit_kmh)
            if signal_aspect == SignalAspect.YELLOW:
                speed = min(speed, YELLOW_SPEED_LIMIT_KMH)

            # Travel through block
            travel_time_min = (block.config.length_km / speed) * 60

            logger.log_train_event(
                env.now, schedule.train_id, train_type_str,
                "ENTER_BLOCK", block_id,
                speed_kmh=speed,
            )

            yield env.timeout(travel_time_min)

            # Exit block
            block.set_clear()
            block.resource.release(block_req)

            logger.log_train_event(
                env.now, schedule.train_id, train_type_str,
                "EXIT_BLOCK", block_id,
                speed_kmh=speed,
            )

        # ── Release section lock ──
        if section_lock and section_lock_req is not None:
            section_lock.release(section_lock_req)

        # ── Release junction lock if still held ──
        if junction_req is not None:
            network.junction_lock.release(junction_req)

        # ── Arrival / Dwell at station ──
        is_final = seg_idx == len(schedule.route_stations) - 2

        if is_final:
            actual_arrival = env.now
            delay = actual_arrival - expected_arrival
            logger.log_train_event(
                env.now, schedule.train_id, train_type_str,
                "ARRIVED_FINAL", to_stn,
                delay_min=max(0, delay),
            )
        else:
            # Intermediate station — dwell
            platform_req = network.stations[to_stn].request()
            yield platform_req

            logger.log_train_event(
                env.now, schedule.train_id, train_type_str,
                "ARRIVED_INTERMEDIATE", to_stn,
            )

            yield env.timeout(schedule.dwell_time_min)

            network.stations[to_stn].release(platform_req)

            logger.log_train_event(
                env.now, schedule.train_id, train_type_str,
                "DEPARTED_INTERMEDIATE", to_stn,
            )


class TrainDispatcher:
    """Spawns train processes according to the timetable."""

    def __init__(
        self,
        env: simpy.Environment,
        network: RailwayNetwork,
        logger: DataLogger,
        timetable: Optional[List[TrainSchedule]] = None,
    ):
        self.env = env
        self.network = network
        self.logger = logger
        self.timetable = timetable or TIMETABLE
        self._processes = []

    def start(self):
        """Register all train processes with the SimPy environment."""
        for schedule in self.timetable:
            proc = self.env.process(
                train_process(self.env, schedule, self.network, self.logger)
            )
            self._processes.append(proc)

        print(f"  Dispatcher: {len(self._processes)} trains scheduled")

    @property
    def num_trains(self) -> int:
        return len(self._processes)
