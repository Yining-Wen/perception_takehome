"""Schema, provenance and unique-ownership checks; no ground-truth labels."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    instances=json.loads((ROOT/'instances.json').read_text());dets=json.loads((ROOT/'variant/detections.json').read_text())['scenes'];result={}
    assert set(instances)==set(dets)
    for scene,items in instances.items():
        mapping=json.loads((ROOT/'data/repaired_filtered'/scene/'original_to_new_rgb.json').read_text())
        ids=set();claims=set()
        for inst in items:
            assert set(inst)=={'id','label','center_world','observations'}
            assert type(inst['id']) is int and inst['id'] not in ids;ids.add(inst['id'])
            xyz=np.asarray(inst['center_world']);assert xyz.shape==(3,) and np.isfinite(xyz).all()
            frames=set()
            for f,k in inst['observations']:
                assert type(f) is int and type(k) is int
                assert 0<=f<len(mapping) and mapping[f] is not None
                assert 0<=k<len(dets[scene][str(f)])
                assert dets[scene][str(f)][k]['label']==inst['label']
                assert (f,k) not in claims and f not in frames
                claims.add((f,k));frames.add(f)
            assert len(frames)>=3
        result[scene]=dict(instances=len(items),unique_observations=len(claims),status='passed')
    out=ROOT/'deliverables/mapping_audit/validation.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
