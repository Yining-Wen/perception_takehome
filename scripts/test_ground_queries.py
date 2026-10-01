"""Regression checks for original/filtered indexing and abstention."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from ground_queries import Grounder,parse_query

class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);s=self.root/'synthetic'
        (s/'depth').mkdir(parents=True)
        cam=dict(width=32,height=32,fx=30.,fy=30.,cx=15.5,cy=15.5)
        poses=[]
        for i,x in enumerate([0.,10.]):
            T=np.eye(4);T[0,3]=x;poses.append(dict(id=i,t=float(i),T_world_camera=T.tolist()))
            cv2.imwrite(str(s/'depth'/f'{i:06d}.png'),np.full((32,32),2000,dtype=np.uint16))
        payloads={'calib.json':dict(color=cam,depth=dict(cam,scale_m=.001,T_color_depth=np.eye(4).tolist()),frames=poses),
                  'original_to_new_rgb.json':[0,None,None,None,1],
                  'index_map.json':[dict(original_rgb_index=0,original_depth_index=0),dict(original_rgb_index=4,original_depth_index=3)],
                  'sync.json':dict(pairs=[[0,0],[1,1]])}
        for f,obj in payloads.items():(s/f).write_text(json.dumps(obj))
        self.args=SimpleNamespace(data_root=self.root,min_detection=.12,min_confidence=.12)
        self.g=Grounder('synthetic',{'4':[dict(label='chair',score=.8,box=[4,4,28,28])]},self.args)
    def tearDown(self):self.tmp.cleanup()
    def test_original_detection_filtered_pose(self):
        d=self.g.observations(4)[0]
        self.assertEqual((d['frame'],d['filtered_frame'],d['depth_frame'],d['index']),(4,1,3,0))
        self.assertAlmostEqual(d['point'][2],2.,places=5)
        self.assertLess(abs(d['point'][0]-10),.1)
    def test_flagged_frame_is_null(self):
        answer,trace=self.g.answer(1,'chair',None)
        self.assertIsNone(answer['goal_world']);self.assertEqual(trace['decision'],'flagged_in_part1')
    def test_materialized_depth_used_despite_time_offset(self):
        self.g.pairs[1,1]+=.1
        detection=self.g.observations(4)[0]
        self.assertEqual(detection['depth_frame'],3)
        self.assertAlmostEqual(detection['offset_ms'],100.)
        self.assertAlmostEqual(detection['point'][2],2.)
    def test_missing_reference_declines(self):
        answer,trace=self.g.answer(4,'chair','sink')
        self.assertIsNone(answer['goal_world']);self.assertEqual(trace['decision'],'reference_object_not_grounded_in_current_frame')
    def test_central_region_median_not_near_surface(self):
        # 6/15 columns at 1m, the majority at 3m: median must be 3m.
        depth=np.full((32,32),3000,dtype=np.uint16)
        depth[:,9:15]=1000
        cv2.imwrite(str(self.root/'synthetic/depth/000001.png'),depth)
        d=self.g.observations(4)[0]
        self.assertAlmostEqual(d['point'][2],3.,places=5)
        self.assertEqual(d['depth_points'],225)
    def test_only_current_frame_is_accessed(self):
        original=self.g.observations
        def current_only(frame):
            self.assertEqual(frame,4)
            return original(frame)
        self.g.observations=current_only
        answer,_=self.g.answer(4,'chair',None)
        self.assertIsNotNone(answer['goal_world'])
        self.assertEqual(set(self.g.cache),{4})
    def test_grammar(self):
        self.assertEqual(parse_query('the chair nearest the sink'),('chair','sink'))
        self.assertEqual(parse_query('the tv'),('tv_monitor',None))
        with self.assertRaises(ValueError):parse_query('something unhandled')

if __name__=='__main__':unittest.main()
