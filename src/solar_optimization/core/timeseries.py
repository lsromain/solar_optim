from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

@dataclass
class TimeSeriesConfig:
    start_time: datetime
    end_time: datetime
    time_delta: timedelta

    def create_timestamps(self) -> List[datetime]:
        timestamps = []
        current_time = self.start_time
        while current_time <= self.end_time:
            timestamps.append(current_time)
            current_time += self.time_delta
        return timestamps