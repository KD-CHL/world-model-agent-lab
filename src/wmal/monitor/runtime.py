"""Experiment-owned lifetime for the read-only service and frame publisher."""
from pathlib import Path

from wmal.monitor.catalog import RunCatalog
from wmal.monitor.frame_buffer import LatestFrameBuffer
from wmal.monitor.publisher import FramePublisher
from wmal.monitor.server import MonitorServer


class MonitorRuntime:
    def __init__(self, run_path, runs_root='runs', port=8765, camera='workcell_overview', max_fps=5):
        catalog = RunCatalog(runs_root)
        run = Path(run_path).resolve()
        if not run.is_relative_to(catalog.root) or run.is_symlink() or not run.is_dir():
            raise ValueError('Monitor run directory must be inside --monitor-runs-root')
        self.run_id = catalog._run_id(catalog._relative(run))
        buffer = LatestFrameBuffer(max_history=2)
        self.publisher = FramePublisher(buffer, self.run_id, camera, max_fps=max_fps)
        self.server = MonitorServer(catalog, buffer, port=port)
        self.server.register_source(self.run_id, self.publisher)

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.address[1]}/?run={self.run_id}'

    def __enter__(self):
        self.server.start()
        return self

    def __exit__(self, *_):
        self.publisher.close()
        self.server.shutdown()
