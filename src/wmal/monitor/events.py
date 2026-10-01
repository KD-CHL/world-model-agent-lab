"""Incremental, read-only JSONL event tailing with bounded line handling."""
import json
from pathlib import Path
import os
from hashlib import sha256
from threading import Lock


def _reject_constant(value):
    raise ValueError('Nonfinite JSON value')


class JsonlEventTail:
    def __init__(self, path, max_line_bytes=1_048_576):
        if type(max_line_bytes) is not int or max_line_bytes < 1:
            raise ValueError('max_line_bytes must be a positive integer')
        self.path = Path(path)
        self.max_line_bytes = max_line_bytes
        self._lock = Lock()
        self._identity = None
        self._offset = 0
        self._pending = b''
        self._discarding = False
        self._cursor = 0
        self._line = 0
        self._source_generation = 0
        self._head_line = None
        self._error_count = 0
        self._last_error = None
        self._exists = False
        self._prefix_hash = sha256()
        self._mtime_ns = None

    @property
    def status(self):
        with self._lock:
            return {'path': str(self.path), 'exists': self._exists,
                    'offset': self._offset, 'source_generation': self._source_generation,
                    'cursor': self._cursor, 'line': self._line,
                    'error_count': self._error_count, 'last_error': self._last_error,
                    'partial_line': bool(self._pending) or self._discarding}

    def reset(self):
        with self._lock:
            self._identity = None
            self._offset = 0
            self._pending = b''
            self._discarding = False
            self._line = 0
            self._head_line = None
            self._exists = False
            self._prefix_hash = sha256()
            self._mtime_ns = None

    def read_new(self):
        with self._lock:
            try:
                stream = self.path.open('rb')
            except FileNotFoundError:
                self._exists = False
                return []
            with stream:
                info = os.fstat(stream.fileno())
                identity = (info.st_dev, info.st_ino)
                if self._identity is not None and identity != self._identity:
                    self._restart_source()
                elif self._identity is not None and info.st_size < self._offset:
                    self._restart_source()
                elif (self._identity is not None and self._offset and info.st_size == self._offset
                      and self._mtime_ns != info.st_mtime_ns):
                    # Appends remain incremental; rare same-length rewrites require a content check.
                    stream.seek(0)
                    digest = sha256()
                    for block in iter(lambda: stream.read(65_536), b''):
                        digest.update(block)
                    if digest.digest() != self._prefix_hash.digest():
                        self._restart_source()
                stream.seek(0)
                first_line = stream.readline(self.max_line_bytes + 1)
                if (self._identity is not None and self._head_line is not None
                        and first_line.rstrip(b'\r\n') != self._head_line
                        and self._offset > 0):
                    self._restart_source()
                if self._identity is None:
                    self._identity = identity
                if self._head_line is None and first_line.endswith(b'\n'):
                    self._head_line = first_line.rstrip(b'\r\n')
                self._exists = True
                stream.seek(self._offset)
                events = self._consume(b'')
                while len(events) < 1000:
                    chunk = stream.read(65_536)
                    if not chunk:
                        break
                    self._offset = stream.tell()
                    self._prefix_hash.update(chunk)
                    events.extend(self._consume(chunk, remaining=1000 - len(events)))
                self._mtime_ns = info.st_mtime_ns
                return events

    def _restart_source(self):
        self._source_generation += 1
        self._identity = None
        self._offset = 0
        self._pending = b''
        self._discarding = False
        self._line = 0
        self._head_line = None
        self._prefix_hash = sha256()
        self._mtime_ns = None

    def _record_error(self, kind, message, line=None):
        self._error_count += 1
        self._last_error = {'kind': kind, 'message': message,
                            'line': self._line + 1 if line is None else line,
                            'source_generation': self._source_generation}

    def _normalize(self, value):
        if not isinstance(value, dict) or not isinstance(value.get('event'), str):
            self._record_error('invalid_event', 'Each event row must be an object with a string event',
                               line=self._line)
            return None
        if isinstance(value.get('payload'), dict):
            payload = value['payload']
        else:
            payload = {key: item for key, item in value.items()
                       if key not in ('event', 'wall_time_utc', 'timestamp', '_source_cursor')}
        self._cursor += 1
        return {'event': value['event'], 'wall_time_utc': value.get('wall_time_utc', value.get('timestamp')),
                'payload': payload, 'raw': value, 'cursor': self._cursor,
                'source_line': self._line, 'source_generation': self._source_generation}

    def _consume(self, incoming, remaining=1000):
        data = self._pending + incoming
        self._pending = b''
        emitted = []
        while b'\n' in data and len(emitted) < remaining:
            line, data = data.split(b'\n', 1)
            self._line += 1
            if self._discarding:
                self._discarding = False
                continue
            if len(line) > self.max_line_bytes:
                self._record_error('line_too_large', 'JSONL row exceeded the configured byte limit',
                                   line=self._line)
                continue
            if not line.strip():
                continue
            try:
                value = json.loads(line.decode('utf-8'), parse_constant=_reject_constant)
            except (UnicodeDecodeError, ValueError, RecursionError):
                self._record_error('malformed_json', 'JSONL row is not valid UTF-8 JSON', line=self._line)
                continue
            row = self._normalize(value)
            if row is not None:
                emitted.append(row)
        self._pending = data
        if self._discarding and b'\n' not in self._pending:
            self._pending = b''
        elif b'\n' not in self._pending and len(self._pending) > self.max_line_bytes:
            self._pending = b''
            self._discarding = True
            self._record_error('line_too_large', 'JSONL row exceeded the configured byte limit')
        return emitted
