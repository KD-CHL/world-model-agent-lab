"""Cancellable prediction only. This module never owns a robot or executor."""
from collections import deque
from dataclasses import dataclass, field
import multiprocessing as mp
import json
import threading
import time
from uuid import uuid4

import numpy as np

from wmal.agents.prediction_contracts import (PredictionRequest, array_hash,
    array_snapshot, content_hash, validate_response)
from wmal.models.embodied_world import EmbodiedWorldModel
from wmal.models.embodied_calibration import EmbodiedCalibration


def predict_request(model, request):
    model.validate_profile(request.profile)
    if (request.model_version != model.version or request.protocol_hash !=
            model.protocol_hash(samples=request.samples, mode=request.mode)):
        raise ValueError('Requested model/protocol differs from frozen worker')
    return request.bind_response(model.predict_context(request.rgb, request.physical,
        request.commands, request.past_actions, request.actions,
        view_mask=request.view_mask, seed=request.seed, samples=request.samples, mode=request.mode))


def _prediction_worker(connection, checkpoint, device):
    try:
        import torch
        torch.set_num_threads(1)
        model = EmbodiedWorldModel.load(checkpoint, device=device)
        connection.send({'ready': model.version})
        while True:
            request = connection.recv()
            try:
                connection.send({'result': predict_request(model, request)})
            except (ValueError, RuntimeError, TypeError, KeyError):
                # No checkpoint paths, request payloads or credentials in diagnostics.
                connection.send({'error': 'Worker rejected prediction contract'})
    except (EOFError, OSError):
        pass
    finally:
        connection.close()


class BoundedInference:
    """Owned spawn worker; all IPC, including sending, is deadline bounded."""
    def __init__(self, checkpoint, *, device='cpu', startup_timeout_s=30):
        self.checkpoint, self.device = str(checkpoint), device
        self.startup_timeout_s = float(startup_timeout_s)
        if not 0 < self.startup_timeout_s <= 60:
            raise ValueError('Invalid startup timeout')
        self._guard, self._calls = threading.RLock(), threading.Lock()
        self.closed, self.faulted, self.generation = False, False, 0
        self._seen = deque(maxlen=2048)
        self.process = self.connection = None
        self._start()

    def _start(self):
        parent, child = mp.get_context('spawn').Pipe()
        process = mp.get_context('spawn').Process(target=_prediction_worker,
            args=(child, self.checkpoint, self.device), daemon=True)
        self.connection, self.process = parent, process
        try:
            process.start(); child.close()
            if not parent.poll(self.startup_timeout_s):
                raise TimeoutError('Prediction worker startup timeout')
            ready = parent.recv()
            if not isinstance(ready, dict) or not ready.get('ready'):
                raise RuntimeError('Prediction worker failed startup')
            self.version = ready['ready']
        except BaseException:
            child.close(); self._stop(); self.faulted = True
            raise

    def _stop(self):
        process, connection = self.process, self.connection
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill(); process.join(1)
        if connection is not None:
            connection.close()

    def infer(self, request):
        if not isinstance(request, PredictionRequest):
            raise ValueError('PredictionRequest required')
        budget = min(request.timeout_s, request.deadline-time.monotonic())
        if budget <= 0:
            raise TimeoutError('Expired client prediction deadline')
        if not self._calls.acquire(timeout=budget):
            raise TimeoutError('Prediction worker busy')
        box, done = {}, threading.Event()
        try:
            with self._guard:
                if self.closed or self.faulted:
                    raise RuntimeError('Prediction worker locked; explicit reset required')
                if request.request_id in self._seen:
                    raise ValueError('Duplicate prediction request')
                generation, connection = self.generation, self.connection
            def exchange():
                try:
                    connection.send(request)
                    box['reply'] = connection.recv()
                except (OSError, EOFError, ValueError) as exc:
                    box['error'] = type(exc).__name__
                finally:
                    done.set()
            thread = threading.Thread(target=exchange, daemon=True)
            thread.start()
            finished = done.wait(max(0, min(budget, request.deadline-time.monotonic())))
            with self._guard:
                if generation != self.generation or self.closed:
                    raise RuntimeError('Prediction invalidated by reset/close')
                if not finished:
                    self.faulted = True; self._stop(); thread.join(1)
                    raise TimeoutError('Prediction worker timeout; locked until reset')
                reply = box.get('reply', {})
                if box.get('error') or not isinstance(reply, dict) or 'result' not in reply:
                    self.faulted = True; self._stop()
                    raise RuntimeError('Prediction worker failed; locked until reset')
                try:
                    result = validate_response(request, reply['result'])
                except (ValueError, TypeError, KeyError):
                    self.faulted = True; self._stop()
                    raise
                self._seen.append(request.request_id)
                return result
        finally:
            self._calls.release()

    def reset(self):
        with self._guard:
            if self.closed:
                raise RuntimeError('Closed prediction worker cannot reset')
            self.generation += 1
            self._stop(); self._seen.clear(); self.faulted = False
            self._start()

    def close(self):
        with self._guard:
            if not self.closed:
                self.closed = True; self.generation += 1; self._stop()


@dataclass(frozen=True, kw_only=True)
class DecisionContext:
    run_id: str
    task_id: str
    node_id: str
    episode_id: str
    context_step_ids: tuple
    observation_times: tuple
    controller: dict
    deadline: float
    rgb: np.ndarray
    state: np.ndarray
    past_actions: np.ndarray
    decision_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        for key in ('rgb', 'state', 'past_actions'):
            object.__setattr__(self, key, array_snapshot(getattr(self, key)))
        object.__setattr__(self, 'controller', json.dumps(self.controller,sort_keys=True,allow_nan=False))
        object.__setattr__(self, 'context_step_ids', tuple(self.context_step_ids))
        object.__setattr__(self, 'observation_times', tuple(self.observation_times))

    def __getattribute__(self, name):
        value=object.__getattribute__(self,name)
        return json.loads(value) if name=='controller' and isinstance(value,str) else value

    def digest(self):
        return content_hash({key: array_hash(getattr(self, key)) for key in
                             ('rgb', 'state', 'past_actions')})


class EmbodiedAgentProvider:
    family = 'wmal.embodied_world.v1'
    def __init__(self, model, calibration, transport, *, samples=8):
        self.model, self.transport, self.samples = model, transport, samples
        self.adapter = model.as_visual_predictor(calibration, samples=samples)
        self.calibration = self.adapter.calibration
        self._context = None
        self._lock = threading.RLock()

    @property
    def version(self):
        return self.model.version

    @property
    def config(self):
        return self.adapter.config

    @property
    def metadata(self):
        return self.model.metadata

    @property
    def semantics(self):
        return self.adapter.semantics

    @property
    def normalization(self):
        return self.adapter.normalization

    def prepare_decision(self, context):
        if not isinstance(context, DecisionContext):
            raise ValueError('Explicit immutable task decision context required')
        with self._lock:
            self._context = context

    def predict_candidate(self, rgb, state, past_actions, actions, *, candidate_id, skill):
        with self._lock:
            context = self._context
        if context is None:
            raise RuntimeError('No current task context; prepare after explicit reset')
        if skill not in ('joint_reach', 'joint_hold'):
            raise ValueError('Unversioned or unknown fixed joint skill')
        digest = content_hash({key: array_hash(value) for key, value in
            zip(('rgb', 'state', 'past_actions'), (rgb, state, past_actions))})
        if digest != context.digest():
            raise ValueError('Prediction history changed after decision binding')
        p = self.model.profile
        rgb, state = np.asarray(rgb), np.asarray(state)
        protocol = self.model.protocol(samples=self.samples)
        request = PredictionRequest(request_id=str(uuid4()), decision_id=context.decision_id,
            run_id=context.run_id, task_id=context.task_id, node_id=context.node_id,
            episode_id=context.episode_id, step_id=context.context_step_ids[-1],
            context_step_ids=context.context_step_ids, observation_times=context.observation_times,
            candidate_id=candidate_id, skill_version='wmal.g1.joint_skills.v1', profile=p,
            controller=context.controller, model_version=self.version,
            protocol_hash=self.model.protocol_hash(samples=self.samples),
            rgb=rgb[:, None] if rgb.ndim == 4 else rgb, physical=state[:, :2], commands=state[:, 2:],
            past_actions=past_actions, actions=actions, deadline=context.deadline,
            timeout_s=max(.000001, context.deadline-time.monotonic()), context_steps=self.config.context_steps,
            max_horizon=self.metadata.get('max_horizon', self.config.horizon),
            member_count=len(self.model.members), samples=self.samples, seed=0,
            event_supervised_channels=tuple(protocol['event_supervised_channels']))
        result = self.transport.infer(request)
        with self._lock:
            if self._context is not context:
                raise RuntimeError('Late prediction from invalidated task')
        validate_response(request, result)
        physical = result['physical_states'].mean(1)
        commands = np.broadcast_to(result['known_commands'], (*physical.shape[:-1], 2))
        result = dict(result)
        result.update(states=np.concatenate([physical, commands], -1),
            frames=result['frames'].mean(1)[:, :, 0], diagnostics={**result.get('diagnostics', {}),
            'prediction_binding': request.binding(), 'prediction_scope': self.metadata.get('prediction_scope')})
        return result

    def reset(self):
        with self._lock:
            self._context = None
        self.transport.reset()

    def close(self):
        with self._lock:
            self._context = None
        self.transport.close()


def load_embodied_agent_provider(checkpoint, calibration, *, device='cpu', samples=8, remote_url=None):
    # Client owns pinned model/profile and calibration; never trusts server bounds.
    model = EmbodiedWorldModel.load(checkpoint, device='cpu')
    calibration = EmbodiedCalibration.load(calibration) if calibration is not None else None
    if remote_url is None:
        transport = BoundedInference(checkpoint, device=device)
    else:
        from wmal.models.embodied_service import RemoteInference
        transport = RemoteInference(remote_url)
    try:
        provider = EmbodiedAgentProvider(model, calibration, transport, samples=samples)
        return provider, provider.calibration
    except BaseException:
        transport.close()
        raise
