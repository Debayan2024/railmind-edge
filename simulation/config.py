"""
RailMind Edge — Simulation Configuration

All network topology, train timetable, and simulation parameters.
Models a 4-station Indian railway corridor with 3-aspect signalling.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────── Enums ───────────────────────────

class TrainType(Enum):
    """Train categories with different priorities and characteristics."""
    RAJDHANI = "Rajdhani"       # Premium express, highest priority
    EXPRESS = "Express"         # Regular express, medium priority
    FREIGHT = "Freight"         # Freight train, lowest priority


class SignalAspect(Enum):
    """3-aspect colour light signal states per Indian Railways rules."""
    RED = "RED"         # Stop — block ahead is occupied
    YELLOW = "YELLOW"   # Caution — proceed, but next block is occupied
    GREEN = "GREEN"     # Clear — proceed at maximum permitted speed


class IncidentType(Enum):
    """Types of anomalies that can be injected into the simulation."""
    TRACK_DEGRADATION = "TRACK_DEGRADATION"
    SIGNAL_STUCK_RED = "SIGNAL_STUCK_RED"
    SIGNAL_WRONG_ASPECT = "SIGNAL_WRONG_ASPECT"
    INTERLOCKING_FAILURE = "INTERLOCKING_FAILURE"


class Severity(Enum):
    """Incident severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ─────────────────────── Data Classes ────────────────────────

@dataclass
class StationConfig:
    """Configuration for a railway station."""
    code: str
    name: str
    position_km: float
    platforms: int
    is_junction: bool = False


@dataclass
class BlockConfig:
    """Configuration for a block section of track."""
    block_id: str
    segment_id: str
    length_km: float
    speed_limit_kmh: float
    position_start_km: float


@dataclass
class SegmentConfig:
    """Configuration for a track segment between two stations."""
    segment_id: str
    from_station: str
    to_station: str
    is_single_track: bool
    block_ids: List[str]


@dataclass
class RouteConfig:
    """Maps a station pair to a segment and direction."""
    segment_id: str
    reverse: bool  # True if blocks are traversed in reverse order


@dataclass
class TrainSchedule:
    """A single train's schedule entry."""
    train_id: str
    train_type: TrainType
    route_stations: List[str]
    departure_time_min: float
    max_speed_kmh: float
    dwell_time_min: float = 2.0


# ═════════════════════════════════════════════════════════════
#                     NETWORK TOPOLOGY
# ═════════════════════════════════════════════════════════════
#
#  Station A (ANP) ══════[Double Track, 40km]══════ Junction B (BHN)
#                                                      │
#                                       ┌──────────────┼──────────────┐
#                                       │              │              │
#                                [Single Track]        │       [Single Track]
#                                   25 km              │          20 km
#                                       │              │              │
#                                 Station C (CNP)      │      Station D (DHG)
#                                  (terminus)          │       (branch)
#
# ═════════════════════════════════════════════════════════════

STATIONS = {
    "ANP": StationConfig("ANP", "Anandpur", 0.0, 3),
    "BHN": StationConfig("BHN", "Bhilai Nagar Junction", 40.0, 2, is_junction=True),
    "CNP": StationConfig("CNP", "Chandanpur", 65.0, 2),
    "DHG": StationConfig("DHG", "Dharamgarh", 60.0, 1),
}

BLOCKS = {
    # ── A → B Up Line (double track, each block ~10 km) ──
    "AB_UP_1": BlockConfig("AB_UP_1", "AB_UP", 10.0, 110.0, 0.0),
    "AB_UP_2": BlockConfig("AB_UP_2", "AB_UP", 10.0, 110.0, 10.0),
    "AB_UP_3": BlockConfig("AB_UP_3", "AB_UP", 10.0, 110.0, 20.0),
    "AB_UP_4": BlockConfig("AB_UP_4", "AB_UP", 10.0, 110.0, 30.0),
    # ── B → A Down Line (double track, reverse direction) ──
    "AB_DN_1": BlockConfig("AB_DN_1", "AB_DN", 10.0, 110.0, 40.0),
    "AB_DN_2": BlockConfig("AB_DN_2", "AB_DN", 10.0, 110.0, 30.0),
    "AB_DN_3": BlockConfig("AB_DN_3", "AB_DN", 10.0, 110.0, 20.0),
    "AB_DN_4": BlockConfig("AB_DN_4", "AB_DN", 10.0, 110.0, 10.0),
    # ── B ↔ C Single Track (3 blocks, ~8.3 km each) ──
    "BC_1": BlockConfig("BC_1", "BC", 8.33, 80.0, 40.0),
    "BC_2": BlockConfig("BC_2", "BC", 8.33, 80.0, 48.33),
    "BC_3": BlockConfig("BC_3", "BC", 8.34, 80.0, 56.67),
    # ── B ↔ D Single Track (2 blocks, 10 km each) ──
    "BD_1": BlockConfig("BD_1", "BD", 10.0, 60.0, 40.0),
    "BD_2": BlockConfig("BD_2", "BD", 10.0, 60.0, 50.0),
}

SEGMENTS = {
    "AB_UP": SegmentConfig("AB_UP", "ANP", "BHN", False,
                           ["AB_UP_1", "AB_UP_2", "AB_UP_3", "AB_UP_4"]),
    "AB_DN": SegmentConfig("AB_DN", "BHN", "ANP", False,
                           ["AB_DN_1", "AB_DN_2", "AB_DN_3", "AB_DN_4"]),
    "BC":    SegmentConfig("BC", "BHN", "CNP", True,
                           ["BC_1", "BC_2", "BC_3"]),
    "BD":    SegmentConfig("BD", "BHN", "DHG", True,
                           ["BD_1", "BD_2"]),
}

# Maps (from_station, to_station) → segment + direction
ROUTES = {
    ("ANP", "BHN"): RouteConfig("AB_UP", reverse=False),
    ("BHN", "ANP"): RouteConfig("AB_DN", reverse=False),
    ("BHN", "CNP"): RouteConfig("BC", reverse=False),
    ("CNP", "BHN"): RouteConfig("BC", reverse=True),
    ("BHN", "DHG"): RouteConfig("BD", reverse=False),
    ("DHG", "BHN"): RouteConfig("BD", reverse=True),
}


# ═════════════════════════════════════════════════════════════
#                     TRAIN TIMETABLE
# ═════════════════════════════════════════════════════════════
# 18 trains over 24 hours across 3 routes.
# Times in minutes from simulation start (00:00).

TIMETABLE = [
    # ── Morning Peak (05:00 – 09:00) ──
    TrainSchedule("12301", TrainType.RAJDHANI, ["ANP", "BHN", "CNP"],
                  300, 100, 2),
    TrainSchedule("12302", TrainType.RAJDHANI, ["CNP", "BHN", "ANP"],
                  315, 100, 2),
    TrainSchedule("14501", TrainType.EXPRESS,  ["ANP", "BHN", "DHG"],
                  360, 75, 3),
    TrainSchedule("14502", TrainType.EXPRESS,  ["DHG", "BHN", "ANP"],
                  390, 75, 3),
    TrainSchedule("12601", TrainType.EXPRESS,  ["ANP", "BHN", "CNP"],
                  420, 80, 3),

    # ── Late Morning (09:00 – 12:00) ──
    TrainSchedule("FGHT01", TrainType.FREIGHT, ["ANP", "BHN", "CNP"],
                  540, 45, 5),
    TrainSchedule("12303", TrainType.RAJDHANI, ["CNP", "BHN", "ANP"],
                  600, 100, 2),

    # ── Afternoon (12:00 – 17:00) ──
    TrainSchedule("12602", TrainType.EXPRESS,  ["CNP", "BHN", "ANP"],
                  720, 80, 3),
    TrainSchedule("14503", TrainType.EXPRESS,  ["ANP", "BHN", "DHG"],
                  780, 75, 3),
    TrainSchedule("FGHT02", TrainType.FREIGHT, ["ANP", "BHN", "DHG"],
                  840, 40, 5),

    # ── Evening Peak (17:00 – 21:00) ──
    TrainSchedule("12304", TrainType.RAJDHANI, ["ANP", "BHN", "CNP"],
                  1020, 100, 2),
    TrainSchedule("14504", TrainType.EXPRESS,  ["DHG", "BHN", "ANP"],
                  1080, 75, 3),
    TrainSchedule("12603", TrainType.EXPRESS,  ["ANP", "BHN", "CNP"],
                  1140, 80, 3),
    TrainSchedule("12604", TrainType.EXPRESS,  ["ANP", "BHN", "DHG"],
                  1200, 75, 3),

    # ── Night (21:00 – 05:00) ──
    TrainSchedule("FGHT03", TrainType.FREIGHT, ["CNP", "BHN", "ANP"],
                  1260, 45, 5),
    TrainSchedule("12305", TrainType.RAJDHANI, ["CNP", "BHN", "ANP"],
                  1320, 100, 2),
    TrainSchedule("FGHT04", TrainType.FREIGHT, ["DHG", "BHN", "ANP"],
                  1380, 40, 5),
    TrainSchedule("14505", TrainType.EXPRESS,  ["ANP", "BHN", "CNP"],
                  1410, 80, 3),
]


# ═════════════════════════════════════════════════════════════
#                   SIMULATION PARAMETERS
# ═════════════════════════════════════════════════════════════

SIM_DURATION_MIN = 24 * 60          # 24 hours
SENSOR_POLL_INTERVAL_MIN = 5        # sensor readings every 5 minutes
RANDOM_SEED = 42

# ── Track Health & Sensor Parameters ──
INITIAL_TRACK_HEALTH = 95.0         # starting health for all blocks (0–100)
HEALTH_DEGRADATION_RATE = 0.005     # health points lost per minute (~7.2/day)
SENSOR_NOISE_STD = 0.05             # relative noise (fraction of reading)

# ── Anomaly Injection Parameters ──
MIN_ANOMALIES = 8                   # min anomalies in a 24h run
MAX_ANOMALIES = 15                  # max anomalies in a 24h run
