"""Import real V2.1 parquet/video without changing held-out episode assignments."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('pyarrow') and importlib.util.find_spec('av'),
                     'Optional visual-data dependencies not installed')
class VisualLeRobotTests(unittest.TestCase):
    def test_v21_alignment_provenance_and_original_test_are_preserved(self):
        import av
        import pyarrow as pa
        import pyarrow.parquet as pq
        from wmal.datasets.visual_lerobot import convert_lerobot_v21
        from wmal.datasets.visual_sequences import VisualDataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            for split,count in [('train',12),('validation',6),('test',2)]:
                for i in range(count):
                    index=len(rows)
                    parquet=root/f'{index}.parquet'
                    video=root/f'{index}.mp4'
                    values=np.arange(8,dtype='float32').reshape(4,2)+index
                    pq.write_table(pa.table({'observation.state':values.tolist(),
                        'action':(values+.1).tolist(),'timestamp':[0.,.1,.2,.3],
                        'frame_index':[0,1,2,3]}),parquet)
                    with av.open(str(video),mode='w') as writer:
                        stream=writer.add_stream('mpeg4',rate=10)
                        stream.width=stream.height=32
                        stream.pix_fmt='yuv420p'
                        for t in range(4):
                            rgb=np.full((32,32,3),(index*9+t)%255,dtype='uint8')
                            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):
                                writer.mux(packet)
                        for packet in stream.encode():
                            writer.mux(packet)
                    rows.append({'episode_index':index,'split':split,
                                 'parquet':{'path':str(parquet)},'video':{'path':str(video)}})
            source={'format':'lerobot_v2.1_wma_source','dataset':{'repo_id':'test','revision':'fixed','license':'test'},
                    'fps':10.,'camera':{'key':'front'},'state':{'order':['j1','j2']},
                    'action':{'order':['j1','j2']},'episodes':rows}
            manifest=root/'source.json'
            manifest.write_text(json.dumps(source))
            result=convert_lerobot_v21(manifest,root/'output',action_schema='test.absolute.v1',
                                      image_size=32,max_frames=4)
            test=VisualDataset(result,'test',horizon=2)
            self.assertEqual({r['episode_id'] for r in test.rows},{'lerobot_18','lerobot_19'})
            sample=test[0]
            np.testing.assert_allclose(sample['states'][0],[18.,19.])
            np.testing.assert_allclose(sample['actions'][0],[18.1,19.1])
            self.assertEqual(test.semantics['action_schema'],'test.absolute.v1')
            self.assertIn('source_manifest_sha256',test.manifest['source'])
            source['episodes'][0]['split']='unsupported'
            manifest.write_text(json.dumps(source))
            with self.assertRaises(ValueError):
                convert_lerobot_v21(manifest,root/'bad',action_schema='test.absolute.v1')


if __name__=='__main__':
    unittest.main()
