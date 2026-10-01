"""Validate all query-frame outputs and evaluate dev labels AFTER inference; never writes answers."""
import csv
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from score import in_box,GOAL_MARGIN_M
from perception.scene import Scene


def main():
    answers=json.loads((ROOT/'answers.json').read_text());queries=json.loads((ROOT/'variant/queries.json').read_text())['scenes']
    out=ROOT/'deliverables/grounding_filtered';traces=json.loads((out/'decisions.json').read_text());rows=[];counts={}
    assert set(answers)==set(queries)
    for name,qs in queries.items():
        assert set(answers[name])=={q['id'] for q in qs}
        targets_path=ROOT/'variant'/name/'query_targets.json'
        targets=json.loads(targets_path.read_text()) if targets_path.exists() else None
        scene=Scene(ROOT/'data'/name) if targets is not None else None
        counts[name]=Counter()
        for q in qs:
            assert set(answers[name][q['id']])==set(map(str,q['frames']))
            for f in q['frames']:
                a=answers[name][q['id']][str(f)];t=traces[name][q['id']][str(f)]
                assert set(a)=={'goal_world','confidence'}
                assert np.isfinite(a['confidence']) and 0<=a['confidence']<=1
                point=a['goal_world']
                assert point is None or np.asarray(point).shape==(3,) and np.isfinite(point).all()
                counts[name][t['decision']]+=1
                if targets is None:continue
                ids=targets[q['id']]['frames'][str(f)]['visible_targets']
                hit=point is not None and any(in_box(point,b,GOAL_MARGIN_M) for b in scene.boxes if b['uid'] in ids)
                rows.append(dict(scene=name,query_id=q['id'],text=q['text'],frame=f,answered=point is not None,confidence=a['confidence'],target_visible=bool(ids),hit=hit,decision=t['decision']))
    with (out/'dev_evaluation.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(query_frames=sum(sum(c.values()) for c in counts.values()),per_scene=counts,validation='Exact scene/query/original-frame coverage, finite xyz, confidence range verified. Dev labels used for evaluation only.')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    print('Dev errors:',json.dumps([r for r in rows if r['answered'] and not r['hit']],indent=2))

if __name__=='__main__':main()
