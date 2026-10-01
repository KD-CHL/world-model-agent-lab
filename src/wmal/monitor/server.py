"""Loopback-only read-only HTTP/SSE monitor, usable standalone or inside an experiment."""
import argparse
from collections import OrderedDict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from threading import Event, Lock, Thread, BoundedSemaphore
from urllib.parse import parse_qs, unquote, urlsplit

from wmal.monitor.catalog import RunCatalog, _encode_png
from wmal.monitor.events import JsonlEventTail
from wmal.monitor.frame_buffer import LatestFrameBuffer
from wmal.monitor.state import sanitize, summarize


class EventStore:
    def __init__(self, catalog, source_stopped=lambda _: False):
        self.catalog = catalog
        self.source_stopped = source_stopped
        self.sessions = OrderedDict()
        self.lock = Lock()

    def snapshot(self, run_id, after=0, limit=500):
        path = self.catalog.event_log_path(run_id)
        if path.is_symlink():
            raise ValueError('Symbolic-link logs are not served')
        with self.lock:
            if run_id not in self.sessions:
                if len(self.sessions) >= 32:
                    self.sessions.popitem(last=False)
                self.sessions[run_id] = (JsonlEventTail(path), deque(maxlen=5000), None)
            self.sessions.move_to_end(run_id)
            tail, history, reset_through = self.sessions[run_id]
            previous_generation = tail.status['source_generation']
            previous_cursor = tail.status['cursor']
            added = tail.read_new()
            source_changed = tail.status['source_generation'] != previous_generation
            if source_changed:
                history.clear()
                reset_through = previous_cursor
                self.sessions[run_id] = (tail, history, reset_through)
            history.extend(added)
            retained = list(history)
            reset = bool(source_changed or (reset_through is not None and after <= reset_through)
                         or after > tail.status['cursor'] or
                         (retained and after and after < retained[0]['cursor'] - 1))
            effective = 0 if reset else after
            selected = [row for row in retained if row['cursor'] > effective][:limit]
            cursor = selected[-1]['cursor'] if selected else tail.status['cursor']
            return sanitize({'events': selected, 'cursor': cursor, 'reset': reset,
                             'source': tail.status,
                             'summary': summarize(retained, self.source_stopped(run_id))})


class MonitorServer:
    def __init__(self, catalog, frame_buffer=None, host='127.0.0.1', port=8765, poll_interval_s=.25):
        if host != '127.0.0.1':
            raise ValueError('Monitor binds only to 127.0.0.1')
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError('Invalid monitor port')
        if not math.isfinite(poll_interval_s) or not .05 <= poll_interval_s <= 10:
            raise ValueError('Invalid monitor polling interval')
        self.catalog = catalog
        self.frames = frame_buffer if frame_buffer is not None else LatestFrameBuffer()
        self.sources = {}
        self.events = EventStore(catalog, lambda run_id: bool(
            self.sources.get(run_id) and self.sources[run_id].status().get('closed')))
        self.poll_interval_s = poll_interval_s
        self.stopping = Event()
        self._streams = BoundedSemaphore(16)
        self._thread = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.send_error(405, 'Read-only monitor')

            do_PUT = do_POST
            do_DELETE = do_POST
            do_PATCH = do_POST

            def do_GET(self):
                try:
                    host_header = self.headers.get('Host', '')
                    if host_header not in (f'127.0.0.1:{owner.address[1]}', f'localhost:{owner.address[1]}'):
                        self.send_error(403, 'Unsupported Host')
                        return
                    origin = self.headers.get('Origin')
                    if origin and origin not in (f'http://127.0.0.1:{owner.address[1]}',
                                                 f'http://localhost:{owner.address[1]}'):
                        self.send_error(403, 'Local origin required')
                        return
                    if len(self.path) > 4096:
                        raise ValueError('Request path too long')
                    owner._get(self)
                except KeyError:
                    self.send_error(404, 'Unknown run or artifact')
                except FileNotFoundError:
                    self.send_error(404, 'Artifact unavailable')
                except (ValueError, TypeError, UnicodeError):
                    self.send_error(400, 'Invalid monitor request or unsupported artifact')
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
                except OSError:
                    self.send_error(503, 'Experiment data unavailable')

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True

    @property
    def address(self):
        return self.httpd.server_address

    def start(self):
        self._thread = Thread(target=self.serve_forever, name='wmal-monitor', daemon=True)
        self._thread.start()
        return self

    def serve_forever(self):
        self.httpd.serve_forever(poll_interval=.1)

    def shutdown(self):
        self.stopping.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def register_source(self, run_id, source):
        self.sources[run_id] = source

    def _send(self, handler, data, content_type='application/json; charset=utf-8', headers=None):
        if not isinstance(data, bytes):
            data = json.dumps(sanitize(data), ensure_ascii=False, allow_nan=False).encode('utf-8')
        handler.send_response(200)
        handler.send_header('Content-Type', content_type)
        handler.send_header('Content-Length', str(len(data)))
        handler.send_header('Cache-Control', 'no-store')
        handler.send_header('X-Content-Type-Options', 'nosniff')
        handler.send_header('Content-Security-Policy',
                            "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        for key, value in (headers or {}).items():
            handler.send_header(key, str(value))
        handler.end_headers()
        handler.wfile.write(data)

    def _number(self, query, key, default, maximum=2 ** 63 - 1):
        raw = query.get(key, [str(default)])[0]
        if not raw.isdigit() or len(raw) > 20:
            raise ValueError('Invalid integer parameter')
        result = int(raw)
        if result > maximum:
            raise ValueError('Integer parameter out of range')
        return result

    def _live_status(self, run_id):
        frame = self.frames.latest(run_id)
        result = {'run_id': run_id, 'state': 'unavailable', 'reason': 'No live frame publisher',
                  'dropped_frames': self.frames.dropped_frames}
        source = self.sources.get(run_id)
        if source is not None:
            result.update(source.status())
        if frame is not None:
            try:
                timestamp = datetime.fromisoformat(frame.captured_at_utc.replace('Z', '+00:00'))
                age = max(0, (datetime.now(timezone.utc) - timestamp).total_seconds())
            except (ValueError, TypeError):
                age = None
            result.update(episode_id=frame.episode_id, step_id=frame.step_id, sim_time_s=frame.sim_time_s,
                          camera=frame.camera, captured_at_utc=frame.captured_at_utc, age_s=age,
                          width=frame.rgb.shape[1], height=frame.rgb.shape[0])
            result['state'] = ('stopped' if source is not None and source.status().get('closed') else
                               'stale' if age is None or age > 4 else 'live')
        return result

    def _get(self, handler):
        url = urlsplit(handler.path)
        path = unquote(url.path)
        query = parse_qs(url.query, max_num_fields=12)
        if path == '/api/health':
            return self._send(handler, {'status': 'ok', 'read_only': True})
        if path == '/api/runs':
            return self._send(handler, {'runs': self.catalog.list_runs()})
        if path == '/' or path.startswith('/static/'):
            name = 'index.html' if path == '/' else path.removeprefix('/static/')
            if name not in ('index.html', 'monitor.css', 'monitor.js'):
                raise KeyError(name)
            asset = Path(__file__).parent / 'static' / name
            if not asset.exists():
                return self._send(handler, b'Monitor service ready', 'text/plain; charset=utf-8')
            types = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
                     '.js': 'text/javascript; charset=utf-8'}
            return self._send(handler, asset.read_bytes(), types[asset.suffix])
        parts = path.strip('/').split('/')
        if len(parts) >= 3 and parts[:2] == ['api', 'runs']:
            run_id = parts[2]
            self.catalog.resolve_run(run_id)
            if len(parts) == 3:
                snapshot = self.events.snapshot(run_id)
                metadata = next(row for row in self.catalog.list_runs() if row['run_id'] == run_id)
                return self._send(handler, {'run': metadata, 'artifacts': self.catalog.list_artifacts(run_id),
                                            'summary': snapshot['summary'], 'source': snapshot['source']})
            if len(parts) == 4 and parts[3] == 'events':
                return self._send(handler, self.events.snapshot(run_id, self._number(query, 'after', 0),
                                                               self._number(query, 'limit', 500, 1000)))
            if len(parts) == 4 and parts[3] == 'stream':
                event_id = handler.headers.get('Last-Event-ID')
                if event_id is not None:
                    query['after'] = [event_id]
                return self._stream(handler, run_id, self._number(query, 'after', 0))
            if len(parts) == 4 and parts[3] == 'artifacts':
                return self._send(handler, {'artifacts': self.catalog.list_artifacts(run_id)})
            if len(parts) >= 5 and parts[3] == 'artifacts':
                relative = '/'.join(parts[4:])
                target = self.catalog._safe_file(run_id, relative)
                if target.suffix == '.json':
                    return self._send(handler, self.catalog.read_json_artifact(run_id, relative))
                if target.suffix == '.npz':
                    if query.get('info') == ['1']:
                        return self._send(handler, self.catalog.read_prediction_info(run_id, relative))
                    return self._send(handler, self.catalog.read_prediction_frame(run_id, relative,
                         self._number(query, 'index', 0, 255), query.get('key', ['predicted_rgb'])[0]), 'image/png')
                if target.suffix.lower() in ('.png', '.jpg', '.jpeg') and target.stat().st_size <= 8 * 1024 * 1024:
                    return self._send(handler, target.read_bytes(), 'image/png' if target.suffix.lower() == '.png' else 'image/jpeg')
                raise ValueError('Unsupported preview type')
        if len(parts) == 4 and parts[:2] == ['api', 'live']:
            run_id = parts[2]
            self.catalog.resolve_run(run_id)
            if parts[3] == 'status':
                return self._send(handler, self._live_status(run_id))
            if parts[3] in ('frame.png', 'frame.jpg'):
                frame = self.frames.latest(run_id)
                if frame is None:
                    raise FileNotFoundError('No live frame')
                return self._send(handler, _encode_png(frame.rgb), 'image/png',
                                  {'X-Episode-ID': frame.episode_id, 'X-Step-ID': frame.step_id,
                                   'X-Sim-Time-S': frame.sim_time_s, 'X-Captured-At-UTC': frame.captured_at_utc})
        raise KeyError(path)

    def _stream(self, handler, run_id, after):
        if not self._streams.acquire(blocking=False):
            handler.send_error(503, 'Too many event subscribers')
            return
        try:
            handler.connection.settimeout(3)
            handler.send_response(200)
            handler.send_header('Content-Type', 'text/event-stream')
            handler.send_header('Cache-Control', 'no-cache')
            handler.send_header('X-Content-Type-Options', 'nosniff')
            handler.end_headers()
            first = True
            while not self.stopping.is_set():
                snapshot = self.events.snapshot(run_id, after)
                if first or snapshot['events'] or snapshot['reset']:
                    message = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
                    handler.wfile.write(f'id: {snapshot["cursor"]}\nevent: snapshot\ndata: {message}\n\n'.encode('utf-8'))
                    after = snapshot['cursor']
                    first = False
                else:
                    handler.wfile.write(b': heartbeat\n\n')
                handler.wfile.flush()
                self.stopping.wait(self.poll_interval_s)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            self._streams.release()


def main():
    parser = argparse.ArgumentParser(description='本地只读 Agent 实验监控客户端')
    parser.add_argument('--runs-root', default='runs')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    server = MonitorServer(RunCatalog(args.runs_root), host=args.host, port=args.port).start()
    url = f'http://127.0.0.1:{server.address[1]}'
    print(f'Agent monitor: {url}  (Ctrl+C to stop)', flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        while not server.stopping.wait(1):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == '__main__':
    main()
