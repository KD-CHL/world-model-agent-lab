"""Real loopback requests and allocation/identity/auth boundaries."""
import importlib
import json
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np

from test_embodied_provider import inputs, make_model
from wmal.agents.prediction_contracts import PredictionRequest


def request_for(model, timeout=5):
    rgb,physical,commands,past,actions=inputs()
    protocol=model.protocol(samples=8)
    return PredictionRequest(request_id=f'req-{time.monotonic_ns()}',decision_id='decision1',
        run_id='run1',task_id='task1',node_id='node1',episode_id='episode1',step_id=0,
        context_step_ids=(0,),observation_times=(0.,),candidate_id='candidate1',
        skill_version='wmal.g1.joint_skills.v1',profile=model.profile,controller=model.profile.controller,
        model_version=model.version,protocol_hash=model.protocol_hash(samples=8),
        rgb=rgb,physical=physical,commands=commands,past_actions=past,actions=actions,
        deadline=time.monotonic()+timeout,timeout_s=timeout,context_steps=2,max_horizon=3,
        member_count=1,event_supervised_channels=tuple(protocol['event_supervised_channels']))


class EmbodiedServiceTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('wmal.models.embodied_service')
        except ModuleNotFoundError:
            self.fail('Task5 inference-only service is missing')

    def test_no_auth_nonloopback_refused(self):
        api=self.api()
        with self.assertRaises(ValueError):
            api.InferenceServer(('0.0.0.0',0), 'missing.pt', token=None)

    def test_remote_clock_translation_keeps_original_binding(self):
        api=self.api()
        from wmal.agents.prediction_contracts import content_hash
        model=make_model(importlib.import_module('wmal.models.embodied_world'))
        original=request_for(model)
        payload=api.request_payload(original)
        binding=payload['binding']
        binding['deadline']=.0001  # another host's unrelated monotonic epoch
        binding['request_hash']=content_hash({key:val for key,val in binding.items() if key!='request_hash'})
        local,echo=api.receive_request(payload)
        self.assertGreater(local.deadline,time.monotonic())
        self.assertEqual(echo['deadline'],.0001)
        self.assertNotEqual(local.request_hash,echo['request_hash'])

    def test_wire_rejects_declared_allocation_bomb(self):
        api=self.api()
        payload=json.dumps({'schema':'wmal.embodied_wire.v1','payload':{'__array__':True,
            'dtype':'<f4','shape':[2**63,2**63],'bytes':''}}).encode()
        with self.assertRaises(ValueError):
            api.decode_packet(payload,4096)

    def test_slow_header_drip_obeys_absolute_client_deadline(self):
        api=self.api()
        class SlowHeaders(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                try:
                    self.connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\nX-Slow: ')
                    for _ in range(25):
                        self.connection.sendall(b'x');time.sleep(.03)
                    self.connection.sendall(b'\r\n\r\n{}')
                except OSError:
                    pass
        with HTTPServer(('127.0.0.1',0),SlowHeaders) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                transport=api.RemoteInference(f'http://127.0.0.1:{server.server_address[1]}/predict')
                model=make_model(importlib.import_module('wmal.models.embodied_world'))
                started=time.monotonic()
                with self.assertRaises(TimeoutError):
                    transport.infer(request_for(model,timeout=.1))
                self.assertLess(time.monotonic()-started,.45)
                with self.assertRaises(RuntimeError):
                    transport.infer(request_for(model))
            finally:
                server.shutdown();thread.join(2)

    def test_concurrent_duplicate_http_responses_are_rejected(self):
        api=self.api()
        from wmal.agents.embodied_provider import predict_request
        model=make_model(importlib.import_module('wmal.models.embodied_world'))
        request=request_for(model)
        body=api.encode_packet(predict_request(model,request),api.MAX_RESPONSE_BYTES)
        class Replay(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200);self.send_header('Content-Length',str(len(body)))
                self.end_headers();self.wfile.write(body)
        with HTTPServer(('127.0.0.1',0),Replay) as server:
            owner=threading.Thread(target=server.serve_forever,daemon=True);owner.start()
            try:
                transport=api.RemoteInference(f'http://127.0.0.1:{server.server_address[1]}/predict')
                exchange=transport._exchange
                barrier=threading.Barrier(2)
                def paired(req,conn):
                    result=exchange(req,conn);barrier.wait(2);return result
                transport._exchange=paired  # synchronize actual real HTTP reply arrival
                results=[]
                def call():
                    try:
                        results.append(transport.infer(request))
                    except Exception as exc:
                        results.append(exc)
                threads=[threading.Thread(target=call) for _ in range(2)]
                for thread in threads:thread.start()
                for thread in threads:thread.join(3)
                self.assertEqual(sum(isinstance(value,dict) for value in results),1)
                self.assertTrue(any(isinstance(value,ValueError) for value in results))
            finally:
                server.shutdown();owner.join(2)

    def test_real_loopback_transport_binding_and_size_limit(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            model=make_model(importlib.import_module('wmal.models.embodied_world'))
            path=Path(directory)/'model.pt'; model.save(path)
            with api.InferenceServer(('127.0.0.1',0),path,max_body_bytes=200000) as server:
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                try:
                    url=f'http://127.0.0.1:{server.server_address[1]}/predict'
                    transport=api.RemoteInference(url)
                    request=request_for(model)
                    result=transport.infer(request)
                    self.assertEqual(result['request_id'],request.request_id)
                    self.assertEqual(result['physical_states'].shape,(1,8,2,2))
                    self.assertEqual(result['known_commands'].tolist(),[[111,218],[114,222]])
                    # Oversized declared body rejected without waiting/reading it.
                    bad=Request(url,data=b'{}',headers={'Content-Length':'200001'},method='POST')
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(bad,timeout=2)
                    self.assertEqual(caught.exception.code,413)
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(Request(url.replace('/predict','/execute'),data=b'{}'),timeout=2)
                    self.assertEqual(caught.exception.code,404)
                finally:
                    server.shutdown(); thread.join(2)
