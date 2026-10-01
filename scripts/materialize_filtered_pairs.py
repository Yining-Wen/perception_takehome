"""Materialize accepted pairs; archive rejected original pairs separately with aligned metadata."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from materialize_pairs import ROOT,digest


def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',default='scene_c')
    p.add_argument('--audit-root',type=Path,default=ROOT/'deliverables/depth_pair_search_global')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();source=ROOT/'data'/args.scene;out=args.output.resolve()
    if out.exists():raise SystemExit(f'Refusing existing output: {out}')
    audit=args.audit_root/args.scene;mapping=json.loads((audit/'pairing.json').read_text())
    decisions={r['rgb_frame']:r for r in json.loads((audit/'decisions.json').read_text())}
    original=json.loads((source/'calib.json').read_text())
    calibpath=ROOT/'deliverables/ordered_audit'/args.scene/'working_calib.json'
    working=json.loads(calibpath.read_text());sync=json.loads((ROOT/'variant'/args.scene/'sync.json').read_text())
    n=len(original['frames'])
    assert len(mapping)==len(sync['pairs'])==n and working['frames']==original['frames']
    assert all(j is None or type(j) is int and 0<=j<n for j in mapping)
    temp=out.with_name(out.name+'.building')
    if temp.exists():raise SystemExit(f'Staging directory exists: {temp}')
    temp.mkdir(parents=True)
    kept=[i for i,j in enumerate(mapping) if j is not None];flagged=[i for i,j in enumerate(mapping) if j is None]
    for directory,indices,calib,is_flagged in [(temp,kept,working,False),(temp/'flagged',flagged,original,True)]:
        (directory/'rgb').mkdir(parents=True);(directory/'depth').mkdir()
        result=deepcopy(calib);result['frames']=[];log=deepcopy(sync);log['pairs']=[];provenance=[]
        for new_id,i in enumerate(indices):
            j=i if is_flagged else mapping[i]
            frame=deepcopy(calib['frames'][i]);frame['id']=new_id;result['frames'].append(frame)
            log['pairs'].append([sync['pairs'][i][0],sync['pairs'][j][1]])
            item=dict(new_index=new_id,original_rgb_index=i,original_depth_index=j,original_frame_id=original['frames'][i]['id'])
            if is_flagged:item['decision']=decisions[i]
            provenance.append(item)
            for kind,old_id in [('rgb',i),('depth',j)]:
                src=source/kind/f'{old_id:06d}.png';dst=directory/kind/f'{new_id:06d}.png'
                shutil.copy2(src,dst);assert digest(src)==digest(dst)
            assert frame['t']==original['frames'][i]['t'] and frame['T_world_camera']==original['frames'][i]['T_world_camera']
        dump(directory/'calib.json',result);dump(directory/'sync.json',log)
        dump(directory/'index_map.json',provenance)
        dump(directory/'original_rgb_indices.json',indices)
        dump(directory/'original_depth_indices.json',[i if is_flagged else mapping[i] for i in indices])
        if (source/'boxes.json').exists():shutil.copy2(source/'boxes.json',directory/'boxes.json')
    reverse=[None]*n
    for k,i in enumerate(kept):reverse[i]=k
    dump(temp/'original_to_new_rgb.json',reverse)
    summary=json.loads((audit/'summary.json').read_text())
    dump(temp/'materialization.json',dict(source=str(source),audit=str(audit),original_frames=n,retained_frames=len(kept),flagged_frames=len(flagged),repaired_pairs=sum(mapping[i]!=i for i in kept),working_calibration_sha256=digest(calibpath),pairing_sha256=digest(audit/'pairing.json'),source_sync_sha256=digest(ROOT/'variant'/args.scene/'sync.json'),verification='Every image SHA-256 verified; retained timestamps/poses unchanged; calib frame IDs renumbered; sync and provenance aligned.',search=summary))
    (temp/'README.md').write_text('''# Filtered repaired dataset

Only accepted rows are in rgb/, depth/, calib.json and sync.json. IDs and filenames are compacted to 0..N-1; original timestamps and camera poses are preserved, so time gaps remain. Spatial calibration comes from the frozen working calibration. Read with ordinary perception.scene.Scene.

flagged/ archives rejected ORIGINAL RGB/depth pairs, without applying candidate repairs. It has its own original spatial calib, compact frame IDs, sync and index_map.json with rejection evidence. It is excluded from the parent Scene.

Use original_to_new_rgb.json to translate original query/detection frame indices; null means excluded. Do not apply original-depth mappings again. Existing variant queries/detections/answers were NOT rewritten and must not be indexed directly with compacted frame IDs.

Search thresholds are heuristic. Global appearance similarity is not synchronization truth. Original data and prior repaired output are preserved.
''')
    (temp/'flagged/README.md').write_text('Rejected original pairs, indexed locally. See index_map.json for original indices and rejection decisions. Images, original exposure timestamps and poses preserved; only frame IDs renumbered. Not part of the parent filtered dataset.\n')
    temp.rename(out);print(json.dumps({'output':str(out),'retained':len(kept),'flagged':len(flagged)},indent=2))

if __name__=='__main__':main()
