"""Append task evidence; callers provide structured non-secret fields."""
from datetime import datetime, timezone
from pathlib import Path
import json
from threading import Lock


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()

    def __call__(self, event, payload):
        record = {'wall_time_utc': datetime.now(timezone.utc).isoformat(), 'event': event, 'payload': payload}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.lock, self.path.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
