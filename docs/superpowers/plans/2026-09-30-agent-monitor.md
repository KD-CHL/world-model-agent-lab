# Agent Experiment Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local read-only browser dashboard for live MuJoCo experiment telemetry and historical Agent/world-model run inspection, with safe degradation when live rendering is unavailable.

**Architecture:** A versioned telemetry contract and bounded in-process frame publisher keep MuJoCo access on the simulator-owning thread. A loopback-only Python service tails JSONL and serves validated run artifacts/events plus latest frames; a small browser UI displays live status, prediction-versus-observation evidence, event timeline, and run history. Initial integration targets the existing visual workcell Agent path, then G1 frame source if its renderer can be safely exercised without changing action behavior.

**Tech Stack:** Python 3.10+, standard-library HTTP/SSE service, existing MuJoCo and NumPy extras, static HTML/CSS/JavaScript, unittest; avoid mandatory new server/UI dependencies.

**Spec:** `docs/superpowers/specs/2026-09-30-agent-monitor-design.md`

## Execution status (2026-10-01)

Tasks 1–6 are complete in the current main checkout, as explicitly requested. Final suite: 171 tests, 1 ROS2 integration skipped; scaffold and whitespace checks pass. A real trained visual model completed two consecutive targets in one monitored session, and browser checks exercised live RGB, aligned prediction images, training history, and run comparison. Independent final review produced 8 Important findings, addressed in one regression-fix pass. Evidence, limits and implementation decisions are recorded in [the verification report](../../architecture/agent-monitor-verification.md).

## Global Constraints

- The monitor is read-only and has no robot command, reset, pause, resume, or experiment configuration routes.
- HTTP binds only to `127.0.0.1`; all artifact access is confined to an explicitly configured runs root.
- Web threads never read or mutate MuJoCo model/data/viewer objects; frame publication copies RGB on the simulator owner thread into bounded memory.
- Closing/disconnecting the monitor must not change simulation actions, physics stepping, or task completion.
- Unknown, missing, or uncalibrated values are labeled as such; ensemble spread is never represented as failure probability.
- JSONL source files remain authoritative and unchanged; malformed/incomplete rows are reported without crashing.
- Do not persist continuous video or introduce a database in the first release.

## Review Focus

- A slow/disconnected browser or full frame buffer must not block simulation; test overwrite-oldest behavior and bounded memory.
- A partially written, malformed, truncated, or rotated event log must neither crash the monitor nor replay already consumed events; test each transition.
- Symlinks, encoded paths, and `..` traversal must not escape the configured runs directory; test every file-serving endpoint.
- Old event formats and incomplete task logs must preserve raw evidence and show unknown/incomplete instead of inferred success; test both wrapped and flat schemas.
- A stale or cross-episode frame must never be labeled as current; test run/episode/step alignment and source-stop handling.

---

### Task 1: Telemetry contracts and bounded frame buffer

**Files:**
- Create: `src/wmal/monitor/__init__.py`
- Create: `src/wmal/monitor/contracts.py`
- Create: `src/wmal/monitor/frame_buffer.py`
- Test: `tests/test_monitor_contracts.py`
- Test: `tests/test_monitor_frame_buffer.py`

**Interfaces:**
- Produces `RunDescriptor(run_id: str, run_path: str, run_kind: str | None, started_at_utc: str | None)`.
- Produces `SimFrame(run_id: str, episode_id: str, step_id: int, sim_time_s: float, captured_at_utc: str, camera: str, rgb: numpy.ndarray)`; validates dimensions, uint8 RGB, finite time, nonempty identifiers.
- Produces `LatestFrameBuffer(max_history: int = 1)` with `publish(frame) -> None`, `latest(run_id: str | None = None) -> SimFrame | None`, `history(run_id: str, episode_id: str | None = None) -> list[SimFrame]`, and `dropped_frames -> int`; publication copies arrays and is thread safe.

- [ ] **Step 1: Write failing contract and buffer tests**

Test valid/invalid frame metadata and image shapes, immutable copied data, concurrent publish/read, bounded history and drop accounting, and run/episode filtering.

- [ ] **Step 2: Run `python -m unittest tests.test_monitor_contracts tests.test_monitor_frame_buffer -v` and confirm the tests fail because the module does not exist**

- [ ] **Step 3: Implement the telemetry dataclasses and bounded lock-protected buffer**

Keep frame ownership local; no thread, socket, or MuJoCo logic belongs in the buffer.

- [ ] **Step 4: Run the focused tests and confirm they pass**

- [ ] **Step 5: Commit the telemetry contract and tests**

### Task 2: Safe event tailing and historical run catalog

**Files:**
- Create: `src/wmal/monitor/events.py`
- Create: `src/wmal/monitor/catalog.py`
- Test: `tests/test_monitor_events.py`
- Test: `tests/test_monitor_catalog.py`

**Interfaces:**
- Produces `JsonlEventTail(path, max_line_bytes=1_048_576)` with `read_new() -> list[dict]`, `status -> dict`, `reset() -> None`; supports flat visual-agent rows and wrapped EventLog rows, partial final lines, append, truncate, and file replacement.
- Produces `RunCatalog(runs_root)` with `list_runs() -> list[dict]`, `resolve_run(run_id) -> Path`, `list_artifacts(run_id) -> list[dict]`, `read_json_artifact(run_id, relative_path) -> dict`, and `read_prediction_frame(run_id, relative_path, index) -> bytes | None`.
- Catalog uses opaque run ids derived from validated root-relative paths; resolves and verifies every candidate path after symlink resolution.

- [ ] **Step 1: Add failing tests for partial/malformed lines, rotation, flat/wrapped schemas and cursor de-duplication**
- [ ] **Step 2: Add failing catalog tests for discovered manifests/results/events/NPZ, missing files, symlink escape and traversal attempts**
- [ ] **Step 3: Run focused tests and confirm expected failures**
- [ ] **Step 4: Implement incremental bounded JSONL reader preserving original event payloads and source line/order metadata**
- [ ] **Step 5: Implement read-only run/artifact catalog and safe NPZ image extraction**

NPZ loading must use `allow_pickle=False`, allow only documented prediction keys, enforce shape/byte limits, and convert selected uint8/float RGB frames to PNG without extracting arbitrary files. PNG uses the Python standard library so monitoring does not add an image-codec dependency.

- [ ] **Step 6: Run focused event/catalog tests and confirm they pass**
- [ ] **Step 7: Commit event tailing, catalog and tests**

### Task 3: Read-only loopback monitor service

**Files:**
- Create: `src/wmal/monitor/server.py`
- Create: `scripts/agent_monitor.py`
- Modify: `pyproject.toml`
- Test: `tests/test_monitor_server.py`

**Interfaces:**
- Produces `MonitorServer(catalog, frame_buffer, host='127.0.0.1', port=8765, poll_interval_s=.25)` with `serve_forever()`, `shutdown()`, and `address`.
- Read-only routes: `GET /`, `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/events?after=<cursor>`, `/api/runs/{id}/stream` (SSE), `/api/runs/{id}/artifacts`, `/api/runs/{id}/artifacts/{path}`, `/api/live/{id}/frame.jpg`, and `/api/live/{id}/status`.
- Reject all non-loopback host values at constructor/CLI boundaries; no mutation/control routes.
- Produces CLI `python -m wmal.monitor.server --runs-root runs --host 127.0.0.1 --port 8765 --open-browser` (browser opening is optional and must be opt-in).

- [ ] **Step 1: Add route tests for health/catalog/events/SSE/frame/JSON/artifact handling and 404/400 cases**
- [ ] **Step 2: Add tests asserting host validation, request-size limits, correct content types and absence of control endpoints**
- [ ] **Step 3: Run tests and confirm failures**
- [ ] **Step 4: Implement standard-library threaded HTTP server with bounded request parsing and loopback-only binding**
- [ ] **Step 5: Implement JSON event polling/SSE keepalives, frame freshness metadata, and strict catalog-backed artifact routes**
- [ ] **Step 6: Add console entry point and CLI, defaulting to local runs directory and no automatic browser opening**
- [ ] **Step 7: Run server tests and smoke-test start/stop from the repository environment**
- [ ] **Step 8: Commit the local service and tests**

### Task 4: Dashboard UI for live and historical experiments

**Files:**
- Create: `src/wmal/monitor/static/index.html`
- Create: `src/wmal/monitor/static/monitor.css`
- Create: `src/wmal/monitor/static/monitor.js`
- Modify: `src/wmal/monitor/server.py`
- Test: `tests/test_monitor_ui.py`

**Interfaces:**
- UI consumes only the documented `/api` routes and never reads local files directly.
- Main views: run selector; live frame/status; run/episode/step/sim-time; current task and subgoal state; model/planner/prediction/execution evidence; event timeline and filters; historical artifacts/metrics and run comparison.
- A shared renderer maps known event types/fields and labels absent values as “未记录” / “不适用”.

- [ ] **Step 1: Add static asset and API contract tests (all assets served locally; no external script/style/font hosts; no control actions)**
- [ ] **Step 2: Implement responsive Chinese-language monitor layout, explicit live/stale/stopped/no-frame states, and accessible event list**
- [ ] **Step 3: Implement event normalization/rendering, SSE reconnect with cursor, frame refresh, and run/episode/step alignment indicators**
- [ ] **Step 4: Implement prediction-vs-observation panel and safe history artifact/metric browsing**

Only show image/NPZ data from validated catalog API. Clearly distinguish observed RGB, predicted RGB and stale frames.

- [ ] **Step 5: Add UI tests for unknown fields, XSS-like strings, reconnect, no data, stale frame, and incomplete task states**
- [ ] **Step 6: Run tests and perform a browser smoke check if an available browser tool is configured; otherwise record the limitation**
- [ ] **Step 7: Commit static dashboard and tests**

### Task 5: Publish frames from MuJoCo-owning experiment thread

**Files:**
- Create: `src/wmal/monitor/publisher.py`
- Modify: `src/wmal/envs/visual_workcell.py`
- Modify: `scripts/visual_skill_agent.py`
- Modify: `src/wmal/envs/g1_session.py` only if its render integration can be kept inside the owner thread without impacting physics/actions
- Test: `tests/test_monitor_publisher.py`
- Test: `tests/test_visual_workcell.py`
- Test: `tests/test_g1_locomotion_runtime.py` if G1 hook is added

**Interfaces:**
- Produces `FramePublisher(buffer, run_id, camera, max_fps=5, width=640, height=480)` with `publish(rgb, *, episode_id, step_id, sim_time_s) -> bool`, `status() -> dict`, `close() -> None`.
- Publisher accepts already-rendered RGB only; it does not own or access a MuJoCo object.
- The simulation owner may pass a callback receiving its existing observation RGB, or call a renderer on that owner thread; disabled publisher performs no extra render.

- [ ] **Step 1: Add fake-source tests proving throttling, metadata, close behavior and nonblocking latest-frame publication**
- [ ] **Step 2: Add a MuJoCo integration test showing published frames match the same observation episode/step and RGB content**
- [ ] **Step 3: Run focused tests and confirm failures**
- [ ] **Step 4: Implement bounded publisher with max-FPS throttling and status/error counters**
- [ ] **Step 5: Add opt-in publisher injection to visual workcell session and visual Agent entry point; no publisher keeps the current code path unchanged**
- [ ] **Step 6: Evaluate G1 session owner-thread rendering. If it changes action/physics timing, creates unsafe renderer context behavior, or cannot be tested headlessly, do not add the G1 hook and surface “live frame source unavailable” while retaining event monitoring**
- [ ] **Step 7: Compare deterministic action/state results with monitoring enabled and disabled; assert equality within exact existing deterministic tolerances**
- [ ] **Step 8: Run relevant MuJoCo and monitor tests and commit the integration**

### Task 6: End-to-end verification and user documentation

**Files:**
- Modify: `README.md`
- Create: `docs/24_agent_monitor.md`
- Test: `tests/test_monitor_integration.py`

- [ ] **Step 1: Add end-to-end test that starts a fake or small MuJoCo run, observes live event/frame updates, disconnects client, and verifies the run completes unchanged**
- [ ] **Step 2: Add historical run test using representative existing event/manifest/task/prediction fixture data**
- [ ] **Step 3: Document environment setup, launch commands, UI sections, supported run formats, safe limitations, and live rendering fallback**
- [ ] **Step 4: Run the full unittest suite under the project conda environment, scaffold checks, and `git diff --check`**
- [ ] **Step 5: Review for security (loopback/path traversal/no control routes), research validity (no invented confidence), and unchanged no-monitor behavior**
- [ ] **Step 6: Commit docs and integration tests; report launch command, verified capabilities, and any G1 rendering limitation**
