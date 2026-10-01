"""Kc -> Kd -> timestamps -> Tcd rotation -> trajectory. No scale search."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from check_sync import projected_edges
from parameter_audit import ROOT,Scene,step1,step2,step3,dump,table,rotation_error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reuse-intrinsics',action='store_true',help='Reuse existing RGB/depth audit outputs; rerun downstream steps')
    p.add_argument('--scene',default='scene_a')
    p.add_argument('--det-only',action='store_true',help='Only check pose rotation determinants; skip extended trajectory diagnostics')
    p.add_argument('--accept-principal',action='store_true',help='Allow a winning principal-point candidate that passes held-out screens as a conditional working correction')
    p.add_argument('--models',nargs='+',choices=['principal','focal','rotation','translation','rigid'],default=['principal','focal','rotation'])
    p.add_argument('--output',type=Path,default=ROOT/'deliverables/ordered_audit')
    args=p.parse_args();s=Scene(ROOT/'data'/args.scene);base=args.output/args.scene
    cfg=SimpleNamespace(lag=3,stride=8)
    for label,fn in [('01_rgb',step2),('02_depth',step3),('03_timestamp',step1)]:
        out=base/label;out.mkdir(parents=True,exist_ok=True)
        if args.reuse_intrinsics and label in ('01_rgb','02_depth'):
            for file in ('summary.json','motions.json'):
                if not (out/file).exists(): raise FileNotFoundError(out/file)
            print('REUSE',label,flush=True)
        else:
            print('START',label,flush=True);fn(s,out,cfg)
    pairs=np.array(json.loads((ROOT/'variant'/s.name/'sync.json').read_text())['pairs'],float)
    # Only ORIGINAL timestamp-matched pairs can constrain the extrinsic.
    original_good=np.abs(pairs[:,1]-pairs[:,0])<1e-6
    pair_root=base/'03_timestamp'/'original_only'
    dump(pair_root/s.name/'pairing.json',[i if original_good[i] else None for i in range(len(s))])
    print('START 04_extrinsic',flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/check_alignment.py'),'--scenes',s.name,'--output',str(base/'04_extrinsic'),'--pairing-root',str(pair_root),'--models',*args.models],check=True)
    report=json.loads((base/'04_extrinsic'/s.name/'summary.json').read_text());rot=report['models'].get('rotation')
    # Application is conservative: rotation must beat alternatives and pass held-out checks.
    selected=report['selected_by_validation']
    accept=selected in (['rotation','principal'] if args.accept_principal else ['rotation']) and report['status'].startswith('candidate merits')
    if accept:
        calib=json.loads((base/'04_extrinsic'/s.name/f'calib_candidate_{selected}.json').read_text())
    else:calib=s.calib
    dump(base/'working_calib.json',calib)
    # Freeze calibration before considering any reassociation.
    s.calib=calib
    print('START 04b_pairing_repair',flush=True)
    mapping=[];repair_rows=[]
    for i in range(len(s)):
        delta=np.abs(pairs[:,1]-pairs[i,0]);j=i if original_good[i] else int(delta.argmin())
        chosen=j if delta[j]<1e-6 else None
        mapping.append(chosen)
        row=dict(rgb_frame=i,depth_frame=chosen,original_offset_ms=float(1000*(pairs[i,1]-pairs[i,0])),
                 status='unchanged' if original_good[i] else 'missing_exposure' if chosen is None else 'timestamp_reassociated',
                 original_score_px=None,repaired_score_px=None)
        if chosen is not None and chosen!=i:
            gray=cv2.cvtColor(s.rgb(i),cv2.COLOR_RGB2GRAY)
            edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),50,120)
            distance=cv2.distanceTransform((edges==0).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            for f,key in [(i,'original_score_px'),(chosen,'repaired_score_px')]:
                uv=projected_edges(s,f,.05,.03)
                if len(uv)>=30 and np.count_nonzero(edges)>=30:
                    row[key]=float(np.minimum(distance[uv[:,1],uv[:,0]],10).mean())
        repair_rows.append(row)
    repair_out=base/'04b_pairing_repair'
    dump(repair_out/s.name/'pairing.json',mapping)
    table(repair_out/'pairing.csv',repair_rows)
    scored=[r for r in repair_rows if r['original_score_px'] is not None and r['repaired_score_px'] is not None]
    dump(repair_out/'summary.json',dict(original_timestamp_matched=int(original_good.sum()),
        reassociated=sum(i!=j and j is not None for i,j in enumerate(mapping)),
        unavailable=[i for i,j in enumerate(mapping) if j is None],scorable=len(scored),
        original_mean_px=float(np.mean([r['original_score_px'] for r in scored])) if scored else None,
        repaired_mean_px=float(np.mean([r['repaired_score_px'] for r in scored])) if scored else None,
        improved_fraction=float(np.mean([r['repaired_score_px']<r['original_score_px'] for r in scored])) if scored else None,
        note='Candidates selected by exposure timestamps, then evaluated with frozen calibration. No clock values edited. Per-frame scores are evidence, not an automatic truth label.'))

    if args.det_only:
        determinants=np.array([np.linalg.det(s.pose(i)[:3,:3]) for i in range(len(s))])
        failed=np.flatnonzero(~np.isfinite(determinants)|(np.abs(determinants-1)>1e-3)).tolist()
        dump(base/'05_twc_det'/'summary.json',dict(frames=len(s),threshold=1e-3,min_det=float(determinants.min()),max_det=float(determinants.max()),max_absolute_error=float(np.max(np.abs(determinants-1))),failed_frames=failed,poses_modified=False,limitation='Determinant only; does not verify orthogonality or physical trajectory accuracy.'))
        dump(base/'manifest.json',dict(scene=s.name,order=['RGB-only Kc','depth-only Kd','timestamp screening only','Tcd and intrinsic alternatives on original matched pairs','freeze calibration then timestamp-only reassociation control','Twc determinant only'],scale_m=calib['depth']['scale_m'],scale_searched=False,rotation_applied=accept and selected=='rotation',accepted_model=selected if accept else None,principal_acceptance_enabled=args.accept_principal,extrinsic_models=args.models,poses_modified=False))
        print('COMPLETE',base,flush=True)
        return

    print('START 05_trajectory',flush=True)
    out=base/'05_trajectory';out.mkdir(parents=True,exist_ok=True)
    rgb=json.loads((base/'01_rgb/motions.json').read_text());depth=json.loads((base/'02_depth/motions.json').read_text())
    # Depth odometry indices identify depth files. Associate their EXPOSURE times
    # to RGB pose times; never assume depth file i belongs to RGB pose i.
    exposure_to_rgb={}
    for i,t in enumerate(pairs[:,0]):exposure_to_rgb[round(float(t),6)]=i
    rows=[];E=np.array(calib['depth']['T_color_depth'])
    for source,records in [('rgb',rgb),('depth',depth)]:
        for r in records:
            i,j=r['frame_i'],r['frame_j'];a,b=i,j
            if source=='depth':
                a=exposure_to_rgb.get(round(float(pairs[i,1]),6));b=exposure_to_rgb.get(round(float(pairs[j,1]),6))
                if a is None or b is None or a==b:continue
            expected=np.linalg.inv(s.pose(b))@s.pose(a)
            if source=='rgb':
                R=np.array(r['R']);te=None;ok=r['cheirality_fraction']>.7 and r['homography_fraction']<.9
            else:
                expected=np.linalg.inv(E)@expected@E;observed=np.array(r['T']);R=observed[:3,:3]
                te=float(np.linalg.norm(observed[:3,3]-expected[:3,3]))
                ok=r['converged'] and r['fitness']>.5 and r['condition']<1000 and r['reverse_cycle_translation_m'] is not None and r['reverse_cycle_translation_m']<.05 and r['reverse_cycle_rotation_deg']<2
            rows.append(dict(source=source,source_file=i,target_file=j,pose_i=a,pose_j=b,rotation_error_deg=rotation_error(R,expected[:3,:3]),translation_error_m=te,reliable=bool(ok)))
    table(out/'comparison.csv',rows)
    summary={}
    for source in ['rgb','depth']:
        rr=[r for r in rows if r['source']==source and r['reliable']]
        summary[source]={'reliable_pairs':len(rr)}
        for key in ['rotation_error_deg','translation_error_m']:
            values=[r[key] for r in rr if r[key] is not None]
            if values:summary[source][key]={'median':float(np.median(values)),'p95':float(np.percentile(values,95))}
    trajectory=[]
    for i in range(1,len(s)):
        A=s.pose(i-1);B=s.pose(i);dt=s.frames[i]['t']-s.frames[i-1]['t']
        trajectory.append(dict(frame=i,translation_m=float(np.linalg.norm(B[:3,3]-A[:3,3])),speed_m_s=float(np.linalg.norm(B[:3,3]-A[:3,3])/dt),rotation_deg=rotation_error(B[:3,:3],A[:3,:3])))
    table(out/'increments.csv',trajectory)
    dump(out/'summary.json',summary)
    dump(base/'manifest.json',dict(order=['RGB-only Kc','depth-only Kd','timestamp screening only','Tcd on original matched pairs','freeze calibration then reassociate timestamps','trajectory'],scale_m=calib['depth']['scale_m'],scale_searched=False,rotation_applied=accept and selected=='rotation',accepted_model=selected if accept else None,translation_extrinsic_checked=any(m in args.models for m in ('translation','rigid')),notes=['Kc/Kd sweeps are sensitivity screens; originals retained.','Only original timestamp-matched pairs used to fit/validate extrinsic; repaired pairs evaluated afterwards with frozen calibration.','Depth-only odometry compares depth frames without robot initialization; trajectory comparison later associates exposure times.','No absolute metric scale verification. No original files changed.']))
    print('COMPLETE',base,flush=True)

if __name__=='__main__':main()
