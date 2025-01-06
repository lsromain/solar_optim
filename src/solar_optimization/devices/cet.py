from dataclasses import dataclass
from datetime import timedelta

@dataclass
class CETProperties:
    power: float
    min_duration: timedelta
    max_duration: timedelta