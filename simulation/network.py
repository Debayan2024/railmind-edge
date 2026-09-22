"""
RailMind Edge — Railway Network Model

SimPy-based simulation of a railway corridor with:
- Block sections (simpy.Resource, capacity=1)
- 3-aspect colour light signals (computed from block occupancy)
- Single-track section locks
- Junction interlocking at BHN
"""

import simpy
from typing import Dict, List, Optional, Tuple

from simulation.config import (
    BLOCKS, SEGMENTS, STATIONS, ROUTES,
    BlockConfig, SignalAspect, INITIAL_TRACK_HEALTH,
)
from simulation.data_logger import DataLogger


class Signal:
    """
    A 3-aspect colour light signal guarding entry to a block section.

    Aspect is computed reactively from block occupancy:
      - RED:    the guarded block is occupied
      - YELLOW: guarded block clear, but lookahead block is occupied
      - GREEN:  both blocks clear

    Can be overridden via force_aspect() for anomaly injection.
    """

    def __init__(
        self,
        signal_id: str,
        guarded_block: "BlockSection",
        lookahead_block: Optional["BlockSection"],
        logger: DataLogger,
        env: simpy.Environment,
    ):
        self.signal_id = signal_id
        self.guarded_block = guarded_block
        self.lookahead_block = lookahead_block
        self.logger = logger
        self.env = env
        self._current_aspect = SignalAspect.GREEN
        self._forced_aspect: Optional[SignalAspect] = None

    @property
    def aspect(self) -> SignalAspect:
        return self._current_aspect

    def force_aspect(self, aspect: Optional[SignalAspect]):
        """Force signal to show a specific aspect (for anomaly injection).
        Pass None to clear the override."""
        self._forced_aspect = aspect
        self.update("anomaly_injection")

    def update(self, reason: str = "block_state_change") -> SignalAspect:
        """Recompute aspect from block state and log if changed."""
        if self._forced_aspect is not None:
            new_aspect = self._forced_aspect
        elif self.guarded_block.is_occupied:
            new_aspect = SignalAspect.RED
        elif self.lookahead_block and self.lookahead_block.is_occupied:
            new_aspect = SignalAspect.YELLOW
        else:
            new_aspect = SignalAspect.GREEN

        if new_aspect != self._current_aspect:
            old_aspect = self._current_aspect
            self._current_aspect = new_aspect
            self.logger.log_signal_event(
                self.env.now,
                self.signal_id,
                self.guarded_block.block_id,
                new_aspect.value,
                old_aspect.value,
                reason,
            )
        return self._current_aspect


class BlockSection:
    """
    A block section of track, modeled as a simpy.Resource(capacity=1).

    Tracks occupancy state and health score. When occupancy changes,
    notifies all dependent signals to recompute their aspects.
    """

    def __init__(self, env: simpy.Environment, config: BlockConfig):
        self.env = env
        self.config = config
        self.block_id = config.block_id
        self.resource = simpy.Resource(env, capacity=1)
        self.is_occupied = False
        self.occupying_train: Optional[str] = None
        self.health_score = INITIAL_TRACK_HEALTH
        self._dependent_signals: List[Signal] = []

    def register_signal(self, signal: Signal):
        """Register a signal that depends on this block's occupancy."""
        if signal not in self._dependent_signals:
            self._dependent_signals.append(signal)

    def set_occupied(self, train_id: str):
        """Mark block as occupied and notify dependent signals."""
        self.is_occupied = True
        self.occupying_train = train_id
        self._notify_signals("train_entered")

    def set_clear(self):
        """Mark block as clear and notify dependent signals."""
        self.is_occupied = False
        self.occupying_train = None
        self._notify_signals("train_exited")

    def _notify_signals(self, reason: str):
        """Update all signals that depend on this block."""
        for signal in self._dependent_signals:
            signal.update(reason)


class RailwayNetwork:
    """
    Assembles the full railway network from config.

    Creates block sections, signals (per route direction), section locks
    for single-track segments, junction interlocking, and station platforms.
    """

    def __init__(self, env: simpy.Environment, logger: DataLogger):
        self.env = env
        self.logger = logger

        # Core network components
        self.blocks: Dict[str, BlockSection] = {}
        self.signals: Dict[str, Signal] = {}
        self.stations: Dict[str, simpy.Resource] = {}
        self.section_locks: Dict[str, simpy.Resource] = {}
        self.junction_lock: Optional[simpy.Resource] = None

        # Route → ordered signal chain
        self.route_signals: Dict[Tuple[str, str], List[Signal]] = {}

        self._build()

    def _build(self):
        """Construct the entire network from config."""
        # 1. Create block sections
        for block_id, config in BLOCKS.items():
            self.blocks[block_id] = BlockSection(self.env, config)

        # 2. Create station platform resources
        for code, stn in STATIONS.items():
            self.stations[code] = simpy.Resource(self.env, capacity=stn.platforms)

        # 3. Section locks for single-track segments
        for seg_id, seg in SEGMENTS.items():
            if seg.is_single_track:
                self.section_locks[seg_id] = simpy.Resource(self.env, capacity=1)

        # 4. Junction interlocking lock
        self.junction_lock = simpy.Resource(self.env, capacity=1)

        # 5. Create signals per route direction
        for (from_stn, to_stn), route_cfg in ROUTES.items():
            segment = SEGMENTS[route_cfg.segment_id]
            block_ids = list(segment.block_ids)
            if route_cfg.reverse:
                block_ids = list(reversed(block_ids))

            direction = "REV" if route_cfg.reverse else "FWD"
            signal_chain: List[Signal] = []

            for i, bid in enumerate(block_ids):
                next_bid = block_ids[i + 1] if i + 1 < len(block_ids) else None
                sig_id = f"SIG_{segment.segment_id}_{direction}_{i}"

                sig = Signal(
                    signal_id=sig_id,
                    guarded_block=self.blocks[bid],
                    lookahead_block=self.blocks[next_bid] if next_bid else None,
                    logger=self.logger,
                    env=self.env,
                )
                signal_chain.append(sig)
                self.signals[sig_id] = sig

                # Register signal dependencies:
                # This signal depends on its guarded block and lookahead block
                self.blocks[bid].register_signal(sig)
                if next_bid:
                    self.blocks[next_bid].register_signal(sig)

            self.route_signals[(from_stn, to_stn)] = signal_chain

    # ─────────────── Route Helpers ───────────────

    def get_route_blocks(self, from_station: str, to_station: str) -> List[str]:
        """Get ordered list of block IDs for a station-to-station segment."""
        route_cfg = ROUTES[(from_station, to_station)]
        segment = SEGMENTS[route_cfg.segment_id]
        block_ids = list(segment.block_ids)
        if route_cfg.reverse:
            block_ids = list(reversed(block_ids))
        return block_ids

    def get_route_signal_chain(
        self, from_station: str, to_station: str
    ) -> List[Signal]:
        """Get the signal chain for a route."""
        return self.route_signals[(from_station, to_station)]

    def get_section_lock(
        self, from_station: str, to_station: str
    ) -> Optional[simpy.Resource]:
        """Get the section lock for single-track segments, or None."""
        route_cfg = ROUTES[(from_station, to_station)]
        return self.section_locks.get(route_cfg.segment_id)

    def needs_junction(self, from_station: str, to_station: str) -> bool:
        """Check if this route passes through the BHN junction interlocking."""
        route_cfg = ROUTES[(from_station, to_station)]
        return route_cfg.segment_id in ("BC", "BD")

    def get_full_route_blocks(self, route_stations: List[str]) -> List[str]:
        """Get all block IDs for a multi-station route."""
        all_blocks = []
        for i in range(len(route_stations) - 1):
            all_blocks.extend(
                self.get_route_blocks(route_stations[i], route_stations[i + 1])
            )
        return all_blocks
