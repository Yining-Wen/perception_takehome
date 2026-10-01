import unittest
import numpy as np
from build_instances import associate,merge_eligible

def observation(f,x=0):return dict(frame=f,index=0,label='chair',point=[x,0,2],box=[0,0,20,20],appearance=(np.ones(96)/96,None))
def track(f,x=0):return dict(id=f+1,label='chair',obs=[observation(f,x)],frames={f},last=f)
class MappingTests(unittest.TestCase):
    def test_one_to_one(self):
        matches=associate([track(0)],[observation(1),observation(1,.1)],1)
        self.assertEqual(len(matches),1)
    def test_simultaneous_instances_do_not_merge(self):self.assertIsNone(merge_eligible(track(0),track(0,.1)))
    def test_distant_tracks_do_not_merge(self):self.assertIsNone(merge_eligible(track(0),track(20,2)))
    def test_appearance_supported_revisit(self):self.assertIsNotNone(merge_eligible(track(0),track(20,.1)))
    def test_class_conflict(self):
        b=track(20);b['label']='table';self.assertIsNone(merge_eligible(track(0),b))
if __name__=='__main__':unittest.main()
