"""Verify filtered/flagged partition, source correspondence, image hashes and Twc determinants."""
import argparse
import json
from pathlib import Path
import numpy as np
from materialize_pairs import ROOT,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);args=p.parse_args()
    source=ROOT/'data'/args.scene;root=ROOT/'data/repaired_filtered'/args.scene
    original=json.loads((source/'calib.json').read_text());sync=json.loads((ROOT/'variant'/args.scene/'sync.json').read_text())['pairs']
    working=json.loads((ROOT/'deliverables/ordered_audit'/args.scene/'working_calib.json').read_text())
    seen=[];summaries={}
    for folder,expected in [(root,working),(root/'flagged',original)]:
        calib=json.loads((folder/'calib.json').read_text());m=json.loads((folder/'index_map.json').read_text());pairs=json.loads((folder/'sync.json').read_text())['pairs']
        assert len(calib['frames'])==len(m)==len(pairs)
        assert calib['color']==expected['color'] and calib['depth']==expected['depth']
        for kind in ['rgb','depth']:assert len(list((folder/kind).glob('*.png')))==len(m)
        det=[]
        for k,r in enumerate(m):
            i,j=r['original_rgb_index'],r['original_depth_index'];seen.append(i)
            f=dict(original['frames'][i]);f['id']=k;assert calib['frames'][k]==f
            assert pairs[k]==[sync[i][0],sync[j][1]]
            for kind,index in [('rgb',i),('depth',j)]:assert digest(folder/kind/f'{k:06d}.png')==digest(source/kind/f'{index:06d}.png')
            det.append(float(np.linalg.det(np.array(f['T_world_camera'])[:3,:3])))
        summaries['retained' if folder==root else 'flagged']=dict(frames=len(m),min_det=min(det) if det else None,max_det=max(det) if det else None,max_absolute_error=max(abs(d-1) for d in det) if det else None,failed_local_indices=[i for i,d in enumerate(det) if not np.isfinite(d) or abs(d-1)>1e-3])
    assert sorted(seen)==list(range(len(original['frames']))) and len(set(seen))==len(seen)
    retained=json.loads((root/'index_map.json').read_text());rev=json.loads((root/'original_to_new_rgb.json').read_text())
    assert len(rev)==len(original['frames'])
    expected=[None]*len(rev)
    for k,r in enumerate(retained):expected[r['original_rgb_index']]=k
    assert rev==expected
    pairs=np.array(json.loads((root/'sync.json').read_text())['pairs'],float)
    assert len(pairs)<2 or np.all(np.diff(pairs[:,1])>0)
    report=dict(scene=args.scene,verification='All image hashes, frame IDs, source timestamps/poses, calibration, sync, reverse mapping, partition and retained depth order verified.',threshold=1e-3,twc=summaries,limitation='det only; no geometric trajectory validation or correction.')
    (root/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
