"""Inference-only HTTP transport. Default loopback; no robot/execution route."""
import argparse
import base64
from dataclasses import fields, replace
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import math
import os
import socket
import threading
import time
from urllib.parse import urlsplit

import numpy as np

from wmal.agents.prediction_contracts import PredictionRequest, content_hash, validate_response
from wmal.agents.embodied_provider import BoundedInference

WIRE_SCHEMA = 'wmal.embodied_wire.v1'
MAX_RESPONSE_BYTES = 64*1024*1024
MAX_BODY_BYTES = 4*1024*1024


def loopback(host):
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def encode_packet(value, limit):
    def encode(item):
        if isinstance(item, np.ndarray):
            if item.dtype.kind not in 'bifu' or not np.isfinite(item).all():
                raise ValueError('Invalid wire array')
            return {'__array__':True,'dtype':item.dtype.str,'shape':list(item.shape),
                'bytes':base64.b64encode(np.ascontiguousarray(item).tobytes()).decode('ascii')}
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {key:encode(val) for key,val in item.items()}
        if isinstance(item, (list,tuple)):
            return [encode(val) for val in item]
        return item
    payload = json.dumps({'schema':WIRE_SCHEMA,'payload':encode(value)},allow_nan=False,
        separators=(',',':')).encode()
    if len(payload) > limit:
        raise ValueError('Wire size limit exceeded')
    return payload


def decode_packet(payload, limit):
    if len(payload) > limit:
        raise ValueError('Wire size limit exceeded')
    packet = json.loads(payload)
    if not isinstance(packet,dict) or set(packet) != {'schema','payload'} or packet['schema'] != WIRE_SCHEMA:
        raise ValueError('Invalid wire schema')
    def decode(item):
        if isinstance(item, dict) and '__array__' in item:
            if set(item) != {'__array__','dtype','shape','bytes'} or item['__array__'] is not True:
                raise ValueError('Invalid array envelope')
            dtype = np.dtype(item['dtype'])
            shape = item['shape']
            if (dtype.str not in ('<f4','<f8','<i4','<i8','|b1','|u1') or
                    not isinstance(shape,list) or len(shape)>8 or
                    any(type(x) is not int or x<0 for x in shape)):
                raise ValueError('Unsupported wire shape/dtype')
            size = math.prod(shape)*dtype.itemsize
            if size > limit:
                raise ValueError('Array allocation limit exceeded')
            raw = base64.b64decode(item['bytes'],validate=True)
            if len(raw) != size:
                raise ValueError('Array shape/bytes mismatch')
            return np.frombuffer(raw,dtype=dtype).reshape(shape)
        if isinstance(item,dict):
            return {key:decode(val) for key,val in item.items()}
        if isinstance(item,list):
            return [decode(val) for val in item]
        return item
    return decode(packet['payload'])


def request_payload(request):
    return {'binding':request.binding(),'remaining_s':min(request.timeout_s,
        request.deadline-time.monotonic()),'arrays':{key:getattr(request,key) for key in
        ('rgb','physical','commands','past_actions','actions','view_mask')}}


def receive_request(payload, max_timeout_s=30):
    """Translate remote monotonic clock locally; keep original binding for echo."""
    if not isinstance(payload,dict) or set(payload) != {'binding','remaining_s','arrays'}:
        raise ValueError('Invalid request envelope')
    binding, remaining = payload['binding'],payload['remaining_s']
    if not isinstance(binding,dict):
        raise ValueError('Invalid binding')
    original = dict(binding)
    claimed_hash = original.pop('request_hash',None)
    if claimed_hash != content_hash(original):
        raise ValueError('Invalid canonical request hash')
    if (type(remaining) not in (int,float) or not math.isfinite(remaining) or
            not 0 < remaining <= max_timeout_s or remaining > binding['timeout_s']):
        raise ValueError('Invalid remaining transport budget')
    names = {field.name for field in fields(PredictionRequest) if field.init}
    values = {key:val for key,val in binding.items() if key in names}
    arrays = payload['arrays']
    if not isinstance(arrays,dict) or set(arrays) != {'rgb','physical','commands','past_actions','actions','view_mask'}:
        raise ValueError('Invalid request arrays')
    values.update(arrays,deadline=time.monotonic()+remaining,timeout_s=remaining)
    local = PredictionRequest(**values)
    # Only client clock fields/hash can differ after local translation.
    for key,val in local.binding().items():
        if key not in ('deadline','timeout_s','request_hash') and content_hash(val) != content_hash(binding.get(key)):
            raise ValueError('Request content/identity differs from binding')
    return local,binding


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *args):
        pass  # never log token, body, attacker-controlled headers or paths

    def handle(self):
        # An absolute connection boundary includes request-line/header parsing,
        # unauthenticated idle clients and slow drip, not merely the body reads.
        self.connection_deadline=time.monotonic()+self.server.max_timeout_s
        def expire():
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        timer=threading.Timer(self.server.max_timeout_s,expire)
        timer.daemon=True; timer.start()
        try:
            self.connection.settimeout(self.server.max_timeout_s)
            super().handle()
        except OSError:
            self.close_connection=True
        finally:
            timer.cancel()

    def do_POST(self):
        if self.path != '/predict':
            self.send_error(404); return
        server=self.server
        if server.token is not None and not hmac.compare_digest(
                self.headers.get('Authorization',''), 'Bearer '+server.token):
            self.send_error(401); return
        try:
            raw_length=self.headers.get('Content-Length','')
            if not raw_length.isdecimal() or self.headers.get('Transfer-Encoding'):
                self.send_error(400); return
            length=int(raw_length)
            if length>server.max_body_bytes:
                self.send_error(413); return
            deadline=self.connection_deadline
            parts=[]; remaining=length
            while remaining:
                budget=deadline-time.monotonic()
                if budget<=0:
                    raise TimeoutError('Body deadline')
                self.connection.settimeout(budget)
                part=self.rfile.read1(min(65536,remaining))
                if not part:
                    raise ValueError('Incomplete body')
                parts.append(part); remaining-=len(part)
            local,binding=receive_request(decode_packet(b''.join(parts),server.max_body_bytes),server.max_timeout_s)
            # Decode/header/body time cannot replenish the inference deadline.
            budget=min(local.timeout_s,deadline-time.monotonic())
            if budget<=0:
                raise TimeoutError('Connection prediction deadline')
            local=replace(local,deadline=min(local.deadline,deadline),timeout_s=budget)
            result=server.worker.infer(local)
            result=dict(result); result.update(binding)
            body=encode_packet(result,server.max_response_bytes)
            self.connection.settimeout(max(.001,local.deadline-time.monotonic()))
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)
        except (TimeoutError, OSError):
            self.send_error(504, 'Prediction transport timeout')
        except (ValueError, TypeError, KeyError, RuntimeError, OverflowError, RecursionError):
            self.send_error(400, 'Prediction contract rejected')


class InferenceServer(HTTPServer):
    def __init__(self,address,checkpoint,*,device='cpu',token=None,
                 max_body_bytes=MAX_BODY_BYTES,max_response_bytes=MAX_RESPONSE_BYTES,max_timeout_s=30):
        token=os.environ.get('WMAL_INFERENCE_TOKEN') if token is None else token
        if not loopback(address[0]) and not token:
            raise ValueError('Non-loopback inference requires environment authentication')
        if token is not None and (not isinstance(token,str) or not token or '\n' in token or '\r' in token):
            raise ValueError('Invalid authentication credential')
        if (type(max_body_bytes) is not int or not 0<max_body_bytes<=MAX_BODY_BYTES or
                type(max_response_bytes) is not int or not 0<max_response_bytes<=MAX_RESPONSE_BYTES or
                type(max_timeout_s) not in (int,float) or not 0<max_timeout_s<=30):
            raise ValueError('Invalid service limits')
        self.token,self.max_body_bytes,self.max_response_bytes,self.max_timeout_s=token,max_body_bytes,max_response_bytes,max_timeout_s
        self.worker=BoundedInference(checkpoint,device=device)
        try:
            super().__init__(address,_Handler)
        except BaseException:
            self.worker.close(); raise

    def server_close(self):
        self.worker.close(); super().server_close()


class RemoteInference:
    def __init__(self,url,*,max_response_bytes=MAX_RESPONSE_BYTES):
        parsed=urlsplit(url)
        if (parsed.scheme not in ('http','https') or parsed.path!='/predict' or
                parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment):
            raise ValueError('Explicit inference-only URL required')
        if parsed.hostname != 'localhost':
            try:
                ipaddress.ip_address(parsed.hostname)
            except ValueError as exc:
                raise ValueError('Literal IP required for bounded transport (no unbounded DNS)') from exc
        if not loopback(parsed.hostname) and parsed.scheme != 'https':
            raise ValueError('Remote inference requires verified HTTPS')
        if type(max_response_bytes) is not int or not 0<max_response_bytes<=MAX_RESPONSE_BYTES:
            raise ValueError('Invalid response limit')
        self.url,self.parsed,self.limit=url,parsed,max_response_bytes
        self.token=os.environ.get('WMAL_INFERENCE_TOKEN')
        self.closed,self.faulted,self.generation=False,False,0
        self._seen=set()
        self._guard=threading.RLock()
        self._active={}

    def infer(self,request):
        if self.closed or self.faulted:
            raise RuntimeError('Remote provider locked; explicit reset required')
        if request.request_id in self._seen:
            raise ValueError('Duplicate prediction response/request')
        with self._guard:
            generation=self.generation
        budget=min(request.timeout_s,request.deadline-time.monotonic())
        if budget<=0:
            raise TimeoutError('Expired client deadline')
        host='127.0.0.1' if self.parsed.hostname=='localhost' else self.parsed.hostname
        cls=http.client.HTTPSConnection if self.parsed.scheme=='https' else http.client.HTTPConnection
        connection=cls(host,self.parsed.port,timeout=budget)
        with self._guard:
            self._active[connection]=None
        box,done={},threading.Event()
        def exchange():
            try:
                box['result']=self._exchange(request,connection)
            except Exception as exc:
                box['error']=exc
            finally:
                connection.close()
                done.set()
        thread=threading.Thread(target=exchange,daemon=True)
        thread.start()
        try:
            if not done.wait(max(0,min(budget,request.deadline-time.monotonic()))):
                self._abort(connection);thread.join(1)
                raise TimeoutError('Absolute client transport deadline exceeded')
            if 'error' in box:
                raise box['error']
            result=box['result']
            with self._guard:
                if self.closed or generation!=self.generation:
                    raise RuntimeError('Late remote response after reset/close')
                validate_response(request,result)
                if request.request_id in self._seen:
                    raise ValueError('Duplicate prediction response')
                if len(self._seen)>=2048:
                    raise RuntimeError('Remote request session limit; explicit reset required')
                self._seen.add(request.request_id)
                return result
        except BaseException:
            with self._guard:
                if generation==self.generation:
                    self.faulted=True
            raise
        finally:
            self._abort(connection)
            with self._guard:
                self._active.pop(connection,None)

    def _exchange(self,request,connection):
        try:
            body=encode_packet(request_payload(request),MAX_BODY_BYTES)
            headers={'Content-Type':'application/json'}
            if self.token:
                headers['Authorization']='Bearer '+self.token
            connection.request('POST','/predict',body=body,headers=headers)
            transport_socket=connection.sock
            with self._guard:
                if connection not in self._active or self.closed:
                    raise RuntimeError('Transport cancelled')
                self._active[connection]=transport_socket
            response=connection.getresponse()
            length=response.getheader('Content-Length')
            if response.status!=200 or length is None or not length.isdecimal() or int(length)>self.limit:
                raise ValueError('Rejected or oversized inference response')
            parts=[];remaining=int(length)
            while remaining:
                budget=request.deadline-time.monotonic()
                if budget<=0:
                    raise TimeoutError('Client response deadline')
                transport_socket.settimeout(budget)
                part=response.read1(min(65536,remaining))
                if not part:
                    raise ValueError('Incomplete response')
                parts.append(part);remaining-=len(part)
            result=decode_packet(b''.join(parts),self.limit)
            return result
        finally:
            connection.close()

    def _abort(self,connection):
        with self._guard:
            sock=self._active.get(connection) or connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()

    def reset(self):
        with self._guard:
            if self.closed:
                raise RuntimeError('Closed provider cannot reset')
            self.generation+=1;self.faulted=False;self._seen.clear()
            active=list(self._active)
        for connection in active:
            self._abort(connection)

    def close(self):
        with self._guard:
            self.closed=True;self.generation+=1
            active=list(self._active)
        for connection in active:
            self._abort(connection)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args(argv)
    with InferenceServer((args.host,args.port),args.checkpoint,device=args.device) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__=='__main__':
    main()
