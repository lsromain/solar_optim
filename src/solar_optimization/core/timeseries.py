from datetime import datetime, timedelta
from typing import List


class TimeSeriesConfig:
    @classmethod
    def create_timestamps(self, start_time: datetime=datetime(2024, 1, 1, 0, 0) , end_time: datetime=datetime(2024, 1, 2, 0, 0),time_delta:timedelta=timedelta(minutes=5) ) -> List[datetime]:
        timestamps = []
        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += time_delta
        return timestamps