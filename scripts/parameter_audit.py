"""Shared implementation for six ordered, conditional calibration checks."""
import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

from check_sync import ROOT, Scene
from perception.geometry import intrinsic_matrix, transform, unproject, rotation_from_axis_angle


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def table(path, rows):
    if rows:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def mean(x):
    return float(np.mean(x)) if len(x) else None


def split_frame(scene,i,j):
    start=scene.frames[0]['t']; ti=scene.frames[i]['t']-start; tj=scene.frames[j]['t']-start
    if int(ti/5)!=int(tj/5) or not (.5<=ti%5<4.5 and .5<=tj%5<4.5): return 'audit_only'
    return ('train','validation','test')[int(ti/5)%3]


def step1(scene,out,args):
    pairs=np.array(json.loads((ROOT/'variant'/scene.name/'sync.json').read_text())['pairs'],float)
    if pairs.shape!=(len(scene),2): raise ValueError('Mismatched timestamp count')
    tc,td=pairs.T; delta=(td-tc)*1000; interval=np.diff(tc)
    rows=[dict(frame=i,rgb_t=float(tc[i]),depth_t=float(td[i]),offset_ms=float(delta[i]),
        rgb_interval_ms=float(interval[i-1]*1000) if i else None,
        depth_interval_ms=float((td[i]-td[i-1])*1000) if i else None) for i in range(len(scene))]
    table(out/'timestamps.csv',rows)
    dump(out/'summary.json',dict(frames=len(scene),median_offset_ms=float(np.median(delta)),
        mad_offset_ms=float(np.median(np.abs(delta-np.median(delta)))),p95_abs_offset_ms=float(np.percentile(abs(delta),95)),
        max_abs_offset_ms=float(abs(delta).max()),repeated_depth_rows=int(len(td)-len(np.unique(td))),
        backward_depth_steps=int((np.diff(td)<0).sum()),nonincreasing_rgb_steps=int((interval<=0).sum()),
        rgb_gaps_over_1_5_median=int((interval>1.5*np.median(interval)).sum()),
        conclusion='Timestamp facts only; no calibration or true clock synchronization inferred.'))


def variants(cam):
    yield 'original',cam
    for scale in (.9,.95,1.05,1.1):
        c=dict(cam);c['fx']*=scale;c['fy']*=scale
        yield f'focal_{scale}',c
    for dx,dy in [(-12,0),(12,0),(0,-12),(0,12),(-12,-12),(12,12)]:
        c=dict(cam);c['cx']+=dx;c['cy']+=dy
        yield f'principal_{dx}_{dy}',c


def step2(scene,out,args):
    cv2.setRNGSeed(7)
    orb=cv2.ORB_create(nfeatures=1200,fastThreshold=10)
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    @lru_cache(maxsize=32)
    def features(i):
        im=cv2.cvtColor(scene.rgb(i),cv2.COLOR_RGB2GRAY)
        k,d=orb.detectAndCompute(im,None)
        return k,d
    rows=[]; motions=[]; sweeps=[]
    for i in range(len(scene)-args.lag):
        j=i+args.lag;k1,d1=features(i);k2,d2=features(j)
        row=dict(frame_i=i,frame_j=j,split=split_frame(scene,i,j),matches=0,inliers=0,status='too_few_matches')
        if d1 is not None and d2 is not None:
            matches=[pair[0] for pair in matcher.knnMatch(d1,d2,k=2) if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
            row['matches']=len(matches)
            if len(matches)>=12:
                a=np.float32([k1[m.queryIdx].pt for m in matches]);b=np.float32([k2[m.trainIdx].pt for m in matches])
                F,mask=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,1.,.999)
                if F is not None and F.shape==(3,3) and mask is not None:
                    good=mask.ravel().astype(bool);a,b=a[good],b[good];row['inliers']=len(a)
                    if len(a)>=12:
                        row['status']='ok'
                        H,hm=cv2.findHomography(a,b,cv2.RANSAC,2.)
                        row['homography_fraction']=float(hm.mean()) if hm is not None else 0.
                        for label,cam in variants(scene.color):
                            K=intrinsic_matrix(cam);E=K.T@F@K;s=np.linalg.svd(E,compute_uv=False)
                            err=float(abs(s[0]-s[1])/max(s[0]+s[1],1e-12))
                            sweeps.append(dict(frame_i=i,frame_j=j,split=row['split'],model=label,essential_singular_mismatch=err))
                        K=intrinsic_matrix(scene.color);E=K.T@F@K
                        u,s,v=np.linalg.svd(E);E=u@np.diag([(s[0]+s[1])/2]*2+[0])@v
                        count,R,t,pm=cv2.recoverPose(E,a,b,K)
                        motion=dict(frame_i=i,frame_j=j,R=R.tolist(),translation_direction=t.ravel().tolist(),
                            cheirality_fraction=float(count/len(a)),homography_fraction=row['homography_fraction'],split=row['split'])
                        motions.append(motion)
        rows.append(row)
        if i%300==0: print(scene.name,'RGB',i,flush=True)
    # Uniform schema despite failures.
    for r in rows:r.setdefault('homography_fraction',None)
    table(out/'pairs.csv',rows);table(out/'intrinsic_sweep.csv',sweeps);dump(out/'motions.json',motions)
    summary={}
    for label,_ in variants(scene.color):
        summary[label]={s:mean([r['essential_singular_mismatch'] for r in sweeps if r['model']==label and r['split']==s]) for s in ('train','validation','test')}
    dump(out/'summary.json',dict(pairs=len(rows),recovered_motions=len(motions),models=summary,
        conclusion='Sensitivity screen only. F residual does not validate K. E singular constraints can be weak under planar or low-parallax motion. No intrinsic correction automatically accepted.'))


def cloud(depth,cam,stride):
    h,w=depth.shape;y,x=np.mgrid[0:h,0:w]
    z=depth
    p=np.stack([(x-cam['cx'])*z/cam['fx'],(y-cam['cy'])*z/cam['fy'],z],axis=-1)
    dx=p[1:-1,2:]-p[1:-1,:-2];dy=p[2:,1:-1]-p[:-2,1:-1]
    n=np.cross(dx,dy);norm=np.linalg.norm(n,axis=2)
    valid=(z[1:-1,1:-1]>0)&(z[1:-1,2:]>0)&(z[1:-1,:-2]>0)&(z[2:,1:-1]>0)&(z[:-2,1:-1]>0)&(norm>1e-8)
    valid&=(np.abs(z[1:-1,2:]-z[1:-1,:-2])<.1)&(np.abs(z[2:,1:-1]-z[:-2,1:-1])<.1)
    sub=np.zeros_like(valid);sub[::stride,::stride]=True;valid&=sub
    return p[1:-1,1:-1][valid],n[valid]/norm[valid,None]


def icp(source,target,normals,max_distance=.2,iterations=20):
    T=np.eye(4);tree=cKDTree(target);condition=None;converged=False
    if min(len(source),len(target))<30:return None
    for iteration in range(iterations):
        moved=transform(T,source);dist,index=tree.query(moved)
        use=dist<max_distance
        if use.sum()<30:return None
        x=moved[use];q=target[index[use]];n=normals[index[use]]
        residual=np.sum(n*(x-q),axis=1)
        trim=np.abs(residual)<=np.percentile(abs(residual),85)
        x,q,n,residual=x[trim],q[trim],n[trim],residual[trim]
        A=np.c_[np.cross(x,n),n]
        singular=np.linalg.svd(A,compute_uv=False)
        condition=float(singular[0]/max(singular[-1],1e-12))
        update=np.linalg.lstsq(A,-residual,rcond=1e-6)[0]
        if np.linalg.norm(update[:3])>.3 or np.linalg.norm(update[3:])>.3:return None
        delta=np.eye(4);delta[:3,:3]=rotation_from_axis_angle(update[:3]);delta[:3,3]=update[3:];T=delta@T
        if np.linalg.norm(update)<1e-5:converged=True;break
    moved=transform(T,source);dist,index=tree.query(moved);use=dist<max_distance
    if use.sum()<30:return None
    residual=np.sum((moved[use]-target[index[use]])*normals[index[use]],axis=1)
    return dict(T=T.tolist(),point_plane_rmse_m=float(np.sqrt(np.mean(residual**2))),fitness=float(use.mean()),
        condition=condition,converged=converged,iterations=iteration+1)


def step3(scene,out,args):
    sync=np.array(json.loads((ROOT/'variant'/scene.name/'sync.json').read_text())['pairs'],float)
    # Read depth alone; never initialize ICP from supplied localization.
    @lru_cache(maxsize=32)
    def get(i,model):
        cam=dict(variants(scene.depth_cam))[model]
        return cloud(scene.depth_m(i),cam,args.stride)
    rows=[];motions=[]
    for model,_ in variants(scene.depth_cam):
        for i in range(len(scene)-args.lag):
            j=i+args.lag;p,_=get(i,model);q,n=get(j,model)
            result=icp(p,q,n)
            row=dict(frame_i=i,frame_j=j,model=model,split=split_frame(scene,i,j),
                depth_dt_s=float(sync[j,1]-sync[i,1]),status='failed',rmse_m=None,fitness=None,condition=None,converged=False)
            if result:
                row.update(status='ok',rmse_m=result['point_plane_rmse_m'],fitness=result['fitness'],condition=result['condition'],converged=result['converged'])
                if model=='original':
                    _,pn=get(i,model);reverse=icp(q,p,pn)
                    cycle=np.array(reverse['T'])@np.array(result['T']) if reverse else None
                    motions.append(dict(frame_i=i,frame_j=j,**result,
                        reverse_cycle_translation_m=float(np.linalg.norm(cycle[:3,3])) if cycle is not None else None,
                        reverse_cycle_rotation_deg=rotation_error(cycle[:3,:3],np.eye(3)) if cycle is not None else None))
            rows.append(row)
        print(scene.name,'depth model',model,'done',flush=True)
    table(out/'pairs.csv',rows);dump(out/'motions.json',motions)
    summary={}
    for model,_ in variants(scene.depth_cam):
        summary[model]={}
        for split in ('train','validation','test'):
            subset=[r for r in rows if r['model']==model and r['split']==split]
            ok=[r for r in subset if r['status']=='ok']
            summary[model][split]=dict(pairs=len(subset),successful=len(ok),mean_rmse_m=mean([r['rmse_m'] for r in ok]),mean_fitness=mean([r['fitness'] for r in ok]))
    dump(out/'summary.json',dict(models=summary,conclusion='Conditional Kd sensitivity only; ICP can absorb distortion. Compare coverage, degeneracy and reverse cycles, not RMSE alone. Scale held fixed; absolute scale unobservable. No correction accepted.'))


def rotation_error(a,b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))


def step4(scene,out,args):
    root=args.output/scene.name
    for step in ('02_rgb','03_depth'):
        if not (root/step/'motions.json').exists():raise FileNotFoundError(f'Run {step} first for {scene.name}')
    visual=json.loads((root/'02_rgb/motions.json').read_text());depth=json.loads((root/'03_depth/motions.json').read_text())
    sync=np.array(json.loads((ROOT/'variant'/scene.name/'sync.json').read_text())['pairs'],float)
    rows=[]
    for kind,records in [('rgb',visual),('depth',depth)]:
        for r in records:
            i,j=r['frame_i'],r['frame_j'];relative=np.linalg.inv(scene.pose(j))@scene.pose(i)
            if kind=='rgb':
                observed=np.array(r['R']);direction=np.array(r['translation_direction']);trans=relative[:3,3]
                angle=float(np.degrees(np.arccos(np.clip(direction@trans/max(np.linalg.norm(trans),1e-12),-1,1)))) if np.linalg.norm(trans)>.01 else None
                row=dict(source=kind,frame_i=i,frame_j=j,rotation_error_deg=rotation_error(observed,relative[:3,:3]),translation_direction_error_deg=angle,translation_error_m=None,metric_scale_ratio=None,reliable=r['cheirality_fraction']>.7 and r['homography_fraction']<.9)
            else:
                E=scene.T_color_depth;expected=np.linalg.inv(E)@relative@E;observed=np.array(r['T'])
                same_time=bool(np.all(np.abs(sync[[i,j],1]-sync[[i,j],0])<=.005))
                cycle=r['reverse_cycle_translation_m']
                reliable=same_time and r['fitness']>.5 and r['condition']<1000 and r['converged'] and cycle is not None and cycle<.05 and r['reverse_cycle_rotation_deg']<2
                row=dict(source=kind,frame_i=i,frame_j=j,rotation_error_deg=rotation_error(observed[:3,:3],expected[:3,:3]),translation_direction_error_deg=None,
                    translation_error_m=float(np.linalg.norm(observed[:3,3]-expected[:3,3])),metric_scale_ratio=float(np.linalg.norm(observed[:3,3])/np.linalg.norm(expected[:3,3])) if np.linalg.norm(expected[:3,3])>.03 else None,reliable=bool(reliable))
            rows.append(row)
    table(out/'motion_comparison.csv',rows)
    matrices=[]
    for i in range(len(scene)):
        R=scene.pose(i)[:3,:3]
        matrices.append(dict(frame=i,orthogonality_error=float(np.linalg.norm(R.T@R-np.eye(3))),determinant=float(np.linalg.det(R))))
    table(out/'pose_matrix_checks.csv',matrices)
    dump(out/'summary.json',dict(comparisons=len(rows),usable_comparisons=sum(r['reliable'] for r in rows),
        conclusion='Relative-motion evidence only. RGB depends on handed Kc; depth comparison depends on Kd, scale, extrinsic and pairing. Review flags do not authorize pose correction. Absolute world gauge is unobservable.'))


def main(step):
    p=argparse.ArgumentParser()
    p.add_argument('--scenes',nargs='+',default=['scene_a','scene_b','scene_c','scene_d'])
    p.add_argument('--output',type=Path,default=ROOT/'deliverables/parameter_audit')
    p.add_argument('--lag',type=int,default=3)
    p.add_argument('--stride',type=int,default=8)
    args=p.parse_args()
    if args.lag<1 or args.stride<1:p.error('lag and stride must be positive')
    if step in (5,6):
        script='check_geometry.py' if step==5 else 'check_alignment.py'
        cmd=[sys.executable,str(ROOT/'scripts'/script),'--scenes',*args.scenes,'--output',str(args.output/('05_scale' if step==5 else '06_extrinsic'))]
        if step==5:
            # Explicitly exclude previous RGB-D candidates for this ordered scale check.
            cmd+=['--alignment-root',str(args.output/'unused_rotation_candidates')]
        if step==6:
            calib_root=args.output/'06_inputs'
            for name in args.scenes:
                source=ROOT/'data'/name/'calib.json'
                report_path=args.output/'05_scale'/name/'summary.json'
                if report_path.exists():
                    report=json.loads(report_path.read_text())
                    if report['status']=='conditional candidate merits review':
                        source=report_path.parent/f"calib_candidate_{report['selected_by_validation']}.json"
                dump(calib_root/name/'calib.json',json.loads(source.read_text()))
                dump(calib_root/name/'provenance.json',{'source':str(source),'note':'Kc/Kd hypotheses from steps 2/3 are not automatically accepted.'})
            cmd+=['--calib-root',str(calib_root)]
        subprocess.run(cmd,check=True);return
    labels={1:'01_sync',2:'02_rgb',3:'03_depth',4:'04_trajectory'}
    for name in args.scenes:
        scene=Scene(ROOT/'data'/name);out=args.output/name/labels[step];out.mkdir(parents=True,exist_ok=True)
        if args.lag>=len(scene):p.error('lag exceeds scene length')
        {1:step1,2:step2,3:step3,4:step4}[step](scene,out,args)
        print(name,labels[step],'complete',flush=True)
