"""Audit every RGB-D frame and fit separate, explicitly conditional calibration hypotheses."""
import argparse
import copy
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from check_sync import ROOT, Scene, projected_edges, plt
from perception.geometry import rotation_from_axis_angle


def edge_points(scene, index):
    """Recover foreground edge 3D points with baseline z-buffer visibility."""
    from perception.geometry import unproject, transform, project
    depth = scene.depth_m(index)
    edge = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        a = depth[:-1] if axis == 0 else depth[:, :-1]
        b = depth[1:] if axis == 0 else depth[:, 1:]
        jump = (a > 0) & (b > 0) & (np.abs(a-b) > np.maximum(.05,.03*np.minimum(a,b)))
        if axis == 0:
            edge[:-1] |= jump & (a < b)
            edge[1:] |= jump & (b < a)
        else:
            edge[:,:-1] |= jump & (a < b)
            edge[:,1:] |= jump & (b < a)
    points, source = unproject(depth, scene.depth_cam)
    uv,z = project(transform(scene.T_color_depth,points),scene.color)
    h,w=depth.shape
    valid=(z>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
    points,source,uv,z=points[valid],source[valid].astype(int),np.rint(uv[valid]).astype(int),z[valid]
    flat=uv[:,1]*w+uv[:,0]
    buffer=np.full(h*w,np.inf); np.minimum.at(buffer,flat,z)
    keep=edge[source[:,1],source[:,0]]&(z<=buffer[flat]+.01)
    return points[keep]


def updated(calib, model, values):
    result=copy.deepcopy(calib)
    if model=='principal':
        result['color']['cx']+=float(values[0]); result['color']['cy']+=float(values[1])
    elif model=='focal':
        result['color']['fx']*=float(np.exp(values[0])); result['color']['fy']*=float(np.exp(values[1]))
    elif model in ('translation','rigid'):
        T=np.array(result['depth']['T_color_depth'])
        if model=='rigid':
            T[:3,:3]=rotation_from_axis_angle(np.radians(values[:3]))@T[:3,:3]
            T[:3,3]+=np.asarray(values[3:])
        else:
            T[:3,3]+=np.asarray(values)
        result['depth']['T_color_depth']=T.tolist()
    elif model=='rotation':
        T=np.array(result['depth']['T_color_depth'])
        T[:3,:3]=rotation_from_axis_angle(np.radians(values))@T[:3,:3]
        result['depth']['T_color_depth']=T.tolist()
    return result


def scores(calib, points, frame_ids, distances, counts):
    """Fixed 3D support and out-of-image penalty prevent improving by discarding points."""
    T=np.array(calib['depth']['T_color_depth'])
    pc=points@T[:3,:3].T+T[:3,3]
    c=calib['color']; h,w=distances.shape[1:]
    z=pc[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        u=c['fx']*pc[:,0]/z+c['cx']; v=c['fy']*pc[:,1]/z+c['cy']
    valid=(z>0)&np.isfinite(u)&np.isfinite(v)&(u>=0)&(u<w-1)&(v>=0)&(v<h-1)
    errors=np.full(len(points),10.)
    x,y=u[valid],v[valid]; ix=x.astype(int); iy=y.astype(int); f=frame_ids[valid]
    dx=x-ix; dy=y-iy
    errors[valid]=np.minimum(10,(1-dx)*(1-dy)*distances[f,iy,ix]+dx*(1-dy)*distances[f,iy,ix+1]+(1-dx)*dy*distances[f,iy+1,ix]+dx*dy*distances[f,iy+1,ix+1])
    totals=np.bincount(frame_ids,weights=errors,minlength=len(counts))
    result=np.full(len(counts),np.nan)
    np.divide(totals,counts,out=result,where=counts>0)
    return result


def aggregate(before,after,mask):
    mask=mask&np.isfinite(before)&np.isfinite(after)
    if not mask.any(): return {'frames':0}
    b,a=before[mask],after[mask]
    return dict(frames=int(mask.sum()),before_mean_px=float(b.mean()),after_mean_px=float(a.mean()),mean_gain_px=float((b-a).mean()),improved_fraction=float(np.mean(a<b)),before_p90_px=float(np.percentile(b,90)),after_p90_px=float(np.percentile(a,90)))


def audit(name,args):
    scene=Scene(args.data_root/name, calib_path=(args.calib_root/name/'calib.json') if args.calib_root else None)
    out=args.output/name; out.mkdir(parents=True,exist_ok=True)
    pairs=np.array(json.loads((args.sync_root/name/'sync.json').read_text())['pairs'],float)
    if pairs.shape!=(len(scene),2): raise ValueError('Sync length mismatch')
    # No unreviewed re-pairing: all originals are audited, only synchronous rows fit.
    depth_indices=np.arange(len(scene))
    if args.pairing_root:
        mapping=json.loads((args.pairing_root/name/'pairing.json').read_text())
        if len(mapping)!=len(scene): raise ValueError('Pairing length mismatch')
        depth_indices=np.array([i if j is None else int(j) for i,j in enumerate(mapping)])
        if np.any((depth_indices<0)|(depth_indices>=len(scene))): raise ValueError('Invalid depth index')
        pairing_valid=np.array([j is not None for j in mapping])
    else:
        pairing_valid=np.ones(len(scene),bool)
    offset=1000*(pairs[depth_indices,1]-pairs[:,0])
    blocks=((pairs[:,0]-pairs[0,0])/args.block_s).astype(int)
    phase=(pairs[:,0]-pairs[0,0])%args.block_s
    guarded=(phase>=args.guard_s)&(phase<args.block_s-args.guard_s)
    points=[]; distances=[]; counts=[]; rgb_counts=[]
    for i in range(len(scene)):
        rgb=scene.rgb(i)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),50,120)
        distances.append(cv2.distanceTransform((edges==0).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE))
        p=edge_points(scene,int(depth_indices[i])); points.append(p); counts.append(len(p)); rgb_counts.append(np.count_nonzero(edges))
        if i%300==0: print(f'{name}: loading {i}/{len(scene)}',flush=True)
    counts=np.array(counts); distances=np.stack(distances)
    ids=np.repeat(np.arange(len(scene)),counts); all_points=np.concatenate(points)
    measurable=(counts>=30)&(np.array(rgb_counts)>=30)
    eligible=pairing_valid&measurable&(np.abs(offset)<=args.max_offset_ms)&guarded
    train=eligible&(blocks%3==0); validation=eligible&(blocks%3==1); test=eligible&(blocks%3==2)
    base=scores(scene.calib,all_points,ids,distances,counts); base[~measurable]=np.nan
    results={'original':base}; calibrations={}; summaries={}
    models={'principal':([(-20,20),(-20,20)],[0.,0.]),'focal':([(-.12,.12),(-.12,.12)],[0.,0.]),'rotation':([(-4,4)]*3,[0.,0.,0.])}
    models.update(translation=([(-.1,.1)]*3,[0.]*3),rigid=([(-4,4)]*3+[(-.1,.1)]*3,[0.]*6))
    models={k:models[k] for k in args.models}
    if min(train.sum(),validation.sum(),test.sum())>=10:
        fit_point_mask=train[ids]
        fit_points=all_points[fit_point_mask]; fit_ids=ids[fit_point_mask]
        fit_counts=np.bincount(fit_ids,minlength=len(scene))
        for model,(bounds,zero) in models.items():
            def objective(x):
                per_frame=scores(updated(scene.calib,model,x),fit_points,fit_ids,distances,fit_counts)
                return float(per_frame[train].mean())
            # Coarse training-only initialization helps avoid local edge minima.
            starts=[zero]
            if model=='principal': starts += [[x,y] for x in (-12,0,12) for y in (-12,0,12)]
            if model=='rotation': starts += [[x,y,0] for x in (-2,0,2) for y in (-2,0,2)]
            start=min(starts,key=objective)
            fit=minimize(objective,start,method='Powell',bounds=bounds,options={'maxfev':args.max_evals,'xtol':.01,'ftol':1e-4})
            candidate=updated(scene.calib,model,fit.x)
            after=scores(candidate,all_points,ids,distances,counts); after[~measurable]=np.nan
            results[model]=after; calibrations[model]=candidate
            bound_hit=any(abs(v-lo)<.02*(hi-lo) or abs(v-hi)<.02*(hi-lo) for v,(lo,hi) in zip(fit.x,bounds))
            summaries[model]=dict(parameter_delta=fit.x.tolist(),units='pixels' if model=='principal' else 'log focal multiplier' if model=='focal' else 'translation metres' if model=='translation' else 'rotation vector degrees then translation metres' if model=='rigid' else 'rotation vector degrees, left-multiplied',optimizer_success=bool(fit.success),optimizer_message=str(fit.message),evaluations=int(fit.nfev),near_search_bound=bound_hit,train=aggregate(base,after,train),validation=aggregate(base,after,validation),test=aggregate(base,after,test),all_frames=aggregate(base,after,measurable))
            (out/f'calib_candidate_{model}.json').write_text(json.dumps(candidate,indent=2)+'\n')
            print(f'{name}: {model} {fit.x.round(3)} validation gain={summaries[model]["validation"]["mean_gain_px"]:.3f}px',flush=True)
    chosen=min(summaries,key=lambda m:summaries[m]['validation']['after_mean_px']) if summaries else None
    status='insufficient eligible frames'
    if chosen:
        s=summaries[chosen]; v=s['validation']; t=s['test']
        supported=s['optimizer_success'] and not s['near_search_bound'] and all(x['mean_gain_px']>=.25 and x['mean_gain_px']>=.1*x['before_mean_px'] and x['improved_fraction']>=.6 for x in (v,t))
        status='candidate merits review; cause is not uniquely identified' if supported else 'no correction supported by configured screening thresholds'
    rows=[]
    for i in range(len(scene)):
        split='train' if train[i] else 'validation' if validation[i] else 'test' if test[i] else 'audit_only'
        row=dict(frame=i,depth_frame=int(depth_indices[i]),pairing_valid=bool(pairing_valid[i]),offset_ms=offset[i],time_block=int(blocks[i]),split=split,depth_edge_count=int(counts[i]),rgb_edge_count=int(rgb_counts[i]))
        row.update({f'{key}_score_px':value[i] for key,value in results.items()}); rows.append(row)
    with (out/'frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary=dict(scene=name,frames=len(scene),unscorable_frames=int((~measurable).sum()),train_frames=int(train.sum()),validation_frames=int(validation.sum()),test_frames=int(test.sum()),selected_by_validation=chosen,status=status,models=summaries,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},limitations=['Separate conditional models; principal point and rotation are coupled.','Depth intrinsics, depth scale and trajectory are fixed; translation only fitted in translation/rigid models. Parameters remain coupled.','Original pairings retained; only timestamp-close rows enter fitting/validation/test.','Fixed baseline-visible edge support; candidate visibility is not re-optimized.','RGB texture, occlusion, missing depth and moving objects can bias edge distance.','Temporal blocks reduce adjacent-frame leakage but do not guarantee independent room geometry.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    axes[0].plot(offset); axes[0].set_ylabel('Depth - RGB (ms)')
    for model,values in results.items(): axes[1].plot(values,label=model,alpha=.7)
    axes[1].set_ylabel('Fixed-support edge score (px)'); axes[1].set_xlabel('Frame'); axes[1].legend(); fig.suptitle(name); fig.tight_layout(); fig.savefig(out/'timeline.png',dpi=140); plt.close(fig)
    if chosen:
        # Visualize held-out test quantiles of improvement, not just best examples.
        indices=np.flatnonzero(test); order=indices[np.argsort((base-results[chosen])[indices])]
        selected=np.unique(order[np.linspace(0,len(order)-1,min(6,len(order))).astype(int)])
        for i in selected:
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for ax,calib,title in [(axes[0],scene.calib,'Original'),(axes[1],calibrations[chosen],chosen)]:
                original=scene.calib; scene.calib=calib
                uv=projected_edges(scene,int(depth_indices[i]),.05,.03)
                scene.calib=original
                ax.imshow(scene.rgb(int(i))); ax.scatter(uv[:,0],uv[:,1],s=1,c='lime'); ax.axis('off'); ax.set_title(title)
            fig.suptitle(f'{name} held-out frame {i}: {base[i]:.2f} -> {results[chosen][i]:.2f} px'); fig.tight_layout(); fig.savefig(out/f'overlay_{i:06d}.png',dpi=140); plt.close(fig)
    print(f'{name}: {status}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=['principal','focal','rotation','translation','rigid'],default=['principal','focal','rotation'])
    parser.add_argument('--data-root',type=Path,default=ROOT/'data')
    parser.add_argument('--sync-root',type=Path,default=ROOT/'variant')
    parser.add_argument('--pairing-root',type=Path,help='Root with <scene>/pairing.json: RGB index -> depth index or null')
    parser.add_argument('--calib-root',type=Path,help='Optional root containing <scene>/calib.json')
    parser.add_argument('--scenes',nargs='+',default=['scene_a','scene_b','scene_c','scene_d'])
    parser.add_argument('--output',type=Path,default=ROOT/'deliverables'/'alignment_audit')
    parser.add_argument('--max-offset-ms',type=float,default=5)
    parser.add_argument('--block-s',type=float,default=5)
    parser.add_argument('--guard-s',type=float,default=.5)
    parser.add_argument('--max-evals',type=int,default=400)
    args=parser.parse_args()
    if args.block_s<=2*args.guard_s or args.guard_s<0 or args.max_offset_ms<0 or args.max_evals<1: parser.error('Invalid time split, offset threshold or optimizer budget')
    for name in args.scenes: audit(name,args)

if __name__=='__main__': main()
