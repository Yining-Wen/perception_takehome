"""Window-first search, full-depth fallback, conservative exclusion of unresolved pairs."""
import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from check_sync import ROOT, Scene, projected_edges
from search_depth_pairs import write_csv


def quality(base, ranked, max_score):
    if not ranked:
        return ['no_scorable_candidate']
    best=ranked[0]; reasons=[]
    if best['depth_frame']==base['depth_frame']: reasons.append('original_is_best')
    if best['score_px']>max_score: reasons.append('high_absolute_error')
    gain=None if base['score_px'] is None else base['score_px']-best['score_px']
    if gain is None or gain<.5 or gain<.2*base['score_px']: reasons.append('weak_improvement')
    if len(ranked)<2: reasons.append('no_runner_up')
    elif ranked[1]['score_px']-best['score_px']<max(.15,.1*ranked[1]['score_px']): reasons.append('ambiguous_top_two')
    if best['edge_count']<.8*base['edge_count']: reasons.append('edge_support_drop')
    return reasons


def sequence_reasons(mapping, times, suspect):
    """Only accepted rows participate; mark suspect endpoints of conflicts, never anchors."""
    reasons={}; prev=None
    for i,j in enumerate(mapping):
        if j is None: continue
        if prev is not None:
            a,b=prev
            if times[j]<=times[b]+1e-6:
                why='sequence_reused' if abs(times[j]-times[b])<1e-6 else 'sequence_backwards'
                for k in (a,i):
                    if k in suspect: reasons.setdefault(k,[]).append(why)
        prev=(i,j)
    return reasons


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',default='scene_c');p.add_argument('--window-ms',type=float,default=1000)
    p.add_argument('--max-score-px',type=float,default=3.,help='Heuristic absolute edge error ceiling, not calibrated probability')
    p.add_argument('--output',type=Path,default=ROOT/'deliverables/depth_pair_search_global')
    args=p.parse_args()
    if args.window_ms<=0 or args.max_score_px<=0:p.error('Thresholds must be positive')
    calib=ROOT/'deliverables/ordered_audit'/args.scene/'working_calib.json'
    scene=Scene(ROOT/'data'/args.scene,calib_path=calib)
    pairs=np.asarray(json.loads((ROOT/'variant'/args.scene/'sync.json').read_text())['pairs'],float)
    out=args.output/args.scene
    if out.exists():raise SystemExit(f'Refusing existing output: {out}')
    out.mkdir(parents=True)
    suspect=set(map(int,np.flatnonzero(abs(pairs[:,1]-pairs[:,0])>1e-6)))
    edges=lru_cache(maxsize=None)(lambda j:projected_edges(scene,j,.05,.03))
    rows={}; details=[];mapping=list(range(len(scene))); caches={}
    def evaluate(i,global_search=False):
        if i not in caches:
            gray=cv2.cvtColor(scene.rgb(i),cv2.COLOR_RGB2GRAY)
            e=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),50,120)
            caches[i]=(cv2.distanceTransform((e==0).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE),int(np.count_nonzero(e)),{})
        dist,count,measured=caches[i]
        unique={pairs[i,1]:i}
        candidates=range(len(scene)) if global_search else np.flatnonzero(abs(pairs[:,1]-pairs[i,0])<=args.window_ms/1000)
        for j in candidates:unique.setdefault(pairs[j,1],int(j))
        for j in unique.values():
            if j in measured:continue
            uv=edges(j);v=dist[uv[:,1],uv[:,0]]
            r=dict(rgb_frame=i,depth_frame=j,offset_ms=float(1000*(pairs[j,1]-pairs[i,0])),edge_count=len(v),score_px=float(np.minimum(v,10).mean()) if len(v)>=30 and count>=30 else None,first_scored_in='global' if global_search else 'window')
            measured[j]=r;details.append(r)
        ranked=sorted((measured[j] for j in unique.values() if measured[j]['score_px'] is not None),key=lambda r:(r['score_px'],abs(r['offset_ms']),r['depth_frame']))
        reasons=quality(measured[i],ranked,args.max_score_px)
        best=ranked[0] if ranked else None
        record=dict(stage='global' if global_search else 'window',candidate_count=len(unique),best_depth=best['depth_frame'] if best else None,best_score_px=best['score_px'] if best else None,best_offset_ms=best['offset_ms'] if best else None,original_score_px=measured[i]['score_px'],reason=';'.join(reasons))
        return record,None if reasons else best['depth_frame']
    for i in sorted(suspect):
        record,mapping[i]=evaluate(i);rows[i]={'rgb_frame':i,'window':record}
    # Local sequence failures also trigger global fallback.
    for i,reasons in sequence_reasons(mapping,pairs[:,1],suspect).items():
        rows[i]['window']['reason']=';'.join(filter(None,[rows[i]['window']['reason'],*reasons]));mapping[i]=None
    fallback=[i for i in sorted(suspect) if mapping[i] is None]
    for k,i in enumerate(fallback):
        record,mapping[i]=evaluate(i,True);rows[i]['global']=record
        if k%10==0:print(f'{args.scene} global {k+1}/{len(fallback)}',flush=True)
    # Iterate after exclusions: a removed row can expose a new conflict.
    while True:
        conflicts=sequence_reasons(mapping,pairs[:,1],suspect)
        if not conflicts:break
        for i,reasons in conflicts.items():
            final=rows[i].get('global',rows[i]['window'])
            final['reason']=';'.join(filter(None,[final['reason'],*reasons]));mapping[i]=None
            # A global candidate can expose a new conflict with a previously
            # accepted local row. That row must also receive its global attempt.
            if 'global' not in rows[i]:
                record,mapping[i]=evaluate(i,True)
                rows[i]['global']=record
                fallback.append(i)
    flat=[]
    for i in sorted(suspect):
        r=rows[i];r['accepted_depth']=mapping[i];r['status']='flagged' if mapping[i] is None else 'accepted';final=r.get('global',r['window'])
        flat.append(dict(rgb_frame=i,status=r['status'],accepted_depth=mapping[i],**final,window_reason=r['window']['reason']))
    def dump(name,obj):(out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    dump('pairing.json',mapping);dump('decisions.json',list(rows.values()));write_csv(out/'frames.csv',flat);write_csv(out/'candidates.csv',details)
    accepted=[r for r in flat if r['status']=='accepted']
    summary=dict(scene=args.scene,source_frames=len(scene),suspect_frames=len(suspect),window_ms=args.window_ms,max_score_px=args.max_score_px,global_attempted=len(fallback),accepted_window=sum(r['stage']=='window' for r in accepted),accepted_global=sum(r['stage']=='global' for r in accepted),flagged=sum(j is None for j in mapping),retained=sum(j is not None for j in mapping),candidate_comparisons=len(details),retained_repaired_indices=[r['rgb_frame'] for r in accepted],flagged_indices=[i for i,j in enumerate(mapping) if j is None],accepted_before_mean_px=float(np.mean([r['original_score_px'] for r in accepted])) if accepted else None,accepted_after_mean_px=float(np.mean([r['best_score_px'] for r in accepted])) if accepted else None,calibration=str(calib),policy='Window then full-depth minimum; reject ambiguous/weak/high-error/support-loss matches and exposure reuse/backwards among retained rows. No fabricated timestamps.',limitations=['Thresholds are heuristic, not truth labels.','Global image similarity cannot recover missing exposure or prove synchronization.','Timestamp-matched original rows retained as anchors, not newly image-verified.','After sequence rejection no alternate-candidate optimization is performed.'])
    dump('summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
