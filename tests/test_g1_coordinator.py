from dataclasses import replace
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Thread
import unittest

from wmal.agents.api_client import ApiModelClient
from wmal.agents.g1_coordinator import LanguageMissionPlanner,G1Coordinator,ExecutionGate
from wmal.envs.indoor_scene import IndoorScene
from wmal.locomotion.contracts import G1State,G1VelocityAction
from wmal.locomotion.planner import G1Plan,G1RolloutPlanner
from wmal.locomotion.navigation import NavigationPlanner


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.state=G1State('test',0,0.,0.,0.,.8,0.,0.,0.,0.,0.,0.,0.,.8)

    def test_schema_and_unsupported(self):
        class Client:
            model='test'
            reply={'status':'planned','goals':[{'x':1.,'y':0.,'yaw':None}]}
            def request_json(self,body): return self.reply
        client=Client()
        language=LanguageMissionPlanner(client)
        self.assertEqual(len(language.propose('go',self.state,IndoorScene(),'test')),1)
        client.reply={'status':'unsupported','goals':[]}
        self.assertIsNone(language.propose('grasp',self.state,IndoorScene(),'test'))
        for reply in ({'status':'planned','goals':[{'x':2.,'y':0.,'yaw':None}]},
                      {'status':'planned','goals':[{'x':True,'y':0.,'yaw':None}]},
                      {'status':'planned','goals':[]}, {'status':'planned','goals':[],'code':'evil'}):
            client.reply=reply
            with self.assertRaises(ValueError):
                language.propose('go',self.state,IndoorScene(),'test')

    def test_single_use_and_uncertain_execution(self):
        parent=self
        class Session:
            is_running=True
            fail=False
            calls=0
            def observe(self): return parent.state
            def step(self,a,d):
                self.calls+=1
                if self.fail: raise RuntimeError('connection lost')
                parent.state=replace(parent.state,step_id=parent.state.step_id+1,sim_time_s=parent.state.sim_time_s+d)
                return parent.state
        session=Session()
        gate=ExecutionGate(session,lambda *args:None)
        action=G1VelocityAction(0.,0.,0.,.5)
        predicted=replace(self.state,step_id=1,sim_time_s=.5)
        plan=G1Plan('test',0,action,predicted,predicted,0.)
        gate.approve(self.state,plan)
        gate.step(action,.5)
        with self.assertRaises(RuntimeError): gate.step(action,.5)
        self.assertEqual(session.calls,1)
        predicted=replace(self.state,step_id=2,sim_time_s=1.)
        gate.approve(self.state,G1Plan('test',1,action,predicted,predicted,0.))
        session.fail=True
        with self.assertRaises(RuntimeError): gate.step(action,.5)
        self.assertTrue(gate.faulted)
        with self.assertRaises(RuntimeError): gate.step(action,.5)
        self.assertEqual(session.calls,2)

    def test_http_retry_and_mujoco_end_to_end(self):
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append(body)
                self.send_response(503 if len(calls)==1 else 200)
                self.end_headers()
                self.wfile.write(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':
                    json.dumps({'status':'planned','goals':[{'x':.8,'y':0.,'yaw':None}]})}}]}).encode())
            def log_message(self,*args): pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join,2)
        self.addCleanup(server.shutdown)
        client=ApiModelClient(f'http://127.0.0.1:{server.server_port}/v1','fixture-secret','fixture',timeout_s=5)
        # Use a synthetic model here; the test checks real HTTP and real simulation
        # integration, not model accuracy. No checkpoint or paid API is required.
        from wmal.locomotion.contracts import G1_STATE_SCHEMA,G1_ACTION_SCHEMA,G1Prediction
        class Model:
            version='test-kinematic'
            state_schema,action_schema=G1_STATE_SCHEMA,G1_ACTION_SCHEMA
            def predict(self,s,a,d):
                import math
                vx=math.cos(s.yaw)*a.vx-math.sin(s.yaw)*a.vy
                vy=math.sin(s.yaw)*a.vx+math.cos(s.yaw)*a.vy
                return G1Prediction(replace(s,x=s.x+vx*d,y=s.y+vy*d,
                    yaw=s.yaw+a.yaw_rate*d,step_id=s.step_id+1,sim_time_s=s.sim_time_s+d))
        scene=IndoorScene()
        planner=NavigationPlanner(G1RolloutPlanner(Model(),samples=12,max_linear_velocity=.25),scene)
        records=[]
        coordinator=G1Coordinator(LanguageMissionPlanner(client),planner,planner.scene,Model.version,
                                  log=lambda e,p:records.append((e,p)),max_cycles=40)
        from wmal.envs.g1_session import G1MuJoCoSession
        with G1MuJoCoSession(viewer=False,realtime=False,scene=scene) as session:
            result=coordinator.run('go forward',session)
        self.assertEqual(result['status'],'succeeded')
        self.assertEqual(len(calls),2)
        sent=[p['command_id'] for e,p in records if e=='command_sent']
        ack=[p['command_id'] for e,p in records if e=='command_ack']
        self.assertEqual(sent,ack)
        self.assertNotIn('fixture-secret',json.dumps(records))
