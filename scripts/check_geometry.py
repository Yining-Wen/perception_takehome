"""Full-sequence cross-frame depth audit, conditional scale fitting and trajectory flags."""
import argparse
import copy
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from check_sync import ROOT, Scene, plt
from perception.geometry import unproject, transform, project


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_depth(depth, points, cam):
    """Bilinear depth only within valid, locally smooth 2x2 patches."""
    uv, z = project(points, cam)
    h, w = depth.shape
    inside = (z > 0) & np.isfinite(uv).all(1)
    inside &= (uv[:, 0] >= 0) & (uv[:, 0] < w-1) & (uv[:, 1] >= 0) & (uv[:, 1] < h-1)
    safe = np.where(inside[:, None], uv, 0)
    x, y = safe[:, 0].astype(int), safe[:, 1].astype(int)
    a, b, c, d = depth[y, x], depth[y, x+1], depth[y+1, x], depth[y+1, x+1]
    corners = np.stack([a, b, c, d])
    valid = inside & (corners.min(0) > 0)
    valid &= corners.max(0)-corners.min(0) < np.maximum(.05, .03*corners.min(0))
    dx, dy = safe[:, 0]-x, safe[:, 1]-y
    measured = (1-dx)*(1-dy)*a + dx*(1-dy)*b + (1-dx)*dy*c + dx*dy*d
    return measured, z, valid


def relative_depth(pose_i, pose_j, extrinsic):
    # T_depth_j_depth_i = inv(T_color_depth) inv(T_world_color_j)
    #                       T_world_color_i T_color_depth
    return np.linalg.inv(extrinsic) @ np.linalg.inv(pose_j) @ pose_i @ extrinsic


def direction_score(points, depth_j, cam, relative, multiplier, support, cap):
    """Score in ORIGINAL depth units (metres at input scale), avoiding shrink-to-win."""
    # Divide both prediction and observation by multiplier. Projection is unchanged.
    predicted = points @ relative[:3, :3].T + relative[:3, 3]/multiplier
    observed, z, valid = sample_depth(depth_j, predicted, cam)
    error = np.abs(z-observed)
    chosen_valid = valid[support]
    residual = error[support]
    penalized = np.where(chosen_valid, np.minimum(residual, cap), cap)
    count = len(penalized)
    return (float(penalized.mean()) if count else np.nan,
            float(chosen_valid.mean()) if count else 0.,
            float(np.median(residual[chosen_valid]))*multiplier if chosen_valid.any() else np.nan,
            int(chosen_valid.sum()))


def paired_stats(base, after, coverage, mask):
    use = mask & np.isfinite(base) & np.isfinite(after)
    if not use.any():
        return {'pairs': 0}
    b, a = base[use], after[use]
    return dict(pairs=int(use.sum()),before_score_mm=float(1000*b.mean()),
                after_score_mm=float(1000*a.mean()),gain_mm=float(1000*(b-a).mean()),
                improved_fraction=float((a<b).mean()),mean_support_retained=float(coverage[use].mean()))


def audit(name, args):
    scene = Scene(ROOT/'data'/name)
    out = args.output/name
    out.mkdir(parents=True, exist_ok=True)
    sync = np.asarray(json.loads((ROOT/'variant'/name/'sync.json').read_text())['pairs'], float)
    if sync.shape != (len(scene), 2):
        raise ValueError('Frame and timestamp counts disagree')
    times = sync[:, 0]
    if np.any(np.diff(times) <= 0):
        raise ValueError('RGB timestamps must increase')
    poses = np.array([scene.pose(i) for i in range(len(scene))])
    depth = [scene.depth_m(i) for i in range(len(scene))]
    points = [unproject(d, scene.depth_cam, stride=args.stride)[0] for d in depth]
    original_T = scene.T_color_depth
    models = {'original': scene.calib}
    rotation_file = args.alignment_root/name/'calib_candidate_rotation.json'
    if rotation_file.exists():
        candidate = json.loads(rotation_file.read_text())
        if candidate['depth']['scale_m'] != scene.depth_cam['scale_m']:
            raise ValueError('Rotation candidate unexpectedly changes depth scale')
        models['rotation'] = candidate
    # RGB-only candidate changes have exactly no effect on this depth-only check.
    baseline_relative = {}
    support = {}
    rows = []
    elapsed = times-times[0]
    blocks = (elapsed/args.block_s).astype(int)
    phase = elapsed % args.block_s
    guard = (phase >= args.guard_s) & (phase < args.block_s-args.guard_s)
    close = np.abs(sync[:, 1]-times)*1000 <= args.max_offset_ms
    for lag in args.lags:
        for i in range(len(scene)-lag):
            j = i+lag
            translation = float(np.linalg.norm(poses[j,:3,3]-poses[i,:3,3]))
            counts = []
            for a,b in ((i,j),(j,i)):
                rel = relative_depth(poses[a],poses[b],original_T)
                baseline_relative[a,b] = rel
                observed, z, valid = sample_depth(depth[b],transform(rel,points[a]),scene.depth_cam)
                # Reject points hidden behind target surfaces at baseline; keep foreground conflicts.
                keep = valid & (z-observed <= args.occlusion_m)
                support[a,b] = np.flatnonzero(keep)
                counts.append(len(support[a,b]))
            eligible = close[i] and close[j] and guard[i] and guard[j] and blocks[i]==blocks[j]
            eligible = eligible and translation >= args.min_translation_m and min(counts)>=args.min_points
            split = ('train','validation','test')[blocks[i]%3] if eligible else 'audit_only'
            rows.append(dict(frame_i=i,frame_j=j,lag=lag,dt_s=float(times[j]-times[i]),
                             translation_m=translation,split=split,
                             offset_i_ms=float(1000*(sync[i,1]-times[i])),
                             offset_j_ms=float(1000*(sync[j,1]-times[j])),
                             support_forward=counts[0],support_backward=counts[1],
                             source_points_forward=len(points[i]),source_points_backward=len(points[j])))
    masks = {s:np.array([r['split']==s for r in rows]) for s in ('train','validation','test')}
    relative_cache = {}
    for model, calib in models.items():
        T = np.array(calib['depth']['T_color_depth'])
        relative_cache[model] = baseline_relative if model=='original' else {
            (a,b):relative_depth(poses[a],poses[b],T) for a,b in baseline_relative}

    def evaluate(model, multiplier, selected=None):
        indices = range(len(rows)) if selected is None else selected
        scores, coverage, metric = [], [], []
        for k in indices:
            row = rows[k]; i,j=row['frame_i'],row['frame_j']
            if min(row['support_forward'],row['support_backward']) < args.min_points:
                scores.append(np.nan); coverage.append(0.); metric.append(np.nan); continue
            values = [direction_score(points[a],depth[b],scene.depth_cam,
                       relative_cache[model][a,b],multiplier,support[a,b],args.cap_m)
                      for a,b in ((i,j),(j,i))]
            scores.append(np.mean([v[0] for v in values]))
            coverage.append(np.mean([v[1] for v in values]))
            metric.append(np.mean([v[2] for v in values]))
        return np.array(scores),np.array(coverage),np.array(metric)

    base, base_cov, base_metric = evaluate('original',1.)
    evaluations = {'original':(base,base_cov,base_metric)}
    summaries = {}; curves=[]
    enough = all(mask.sum()>=10 for mask in masks.values())
    grid = np.unique(np.r_[np.linspace(args.scale_min,args.scale_max,args.scale_steps),1.])
    train_indices = np.flatnonzero(masks['train'])
    for model in models:
        fixed = evaluations['original'] if model=='original' else evaluate(model,1.)
        evaluations[model] = fixed
        if not enough:
            continue
        train_curve=[]
        for multiplier in grid:
            score,_,_ = evaluate(model,float(multiplier),train_indices)
            value=float(np.mean(score)); train_curve.append(value)
            curves.append(dict(model=model,multiplier=float(multiplier),train_score_mm=value*1000))
        best_index=int(np.argmin(train_curve)); multiplier=float(grid[best_index])
        key=model+'_scale'
        values=evaluate(model,multiplier); evaluations[key]=values
        summaries[key]=dict(multiplier=multiplier,scale_m=float(scene.depth_cam['scale_m']*multiplier),
            search_boundary=best_index in (0,len(grid)-1),
            train_curve_range_mm=float(1000*np.ptp(train_curve)),
            fixed_scale={s:paired_stats(base,fixed[0],fixed[1],mask) for s,mask in masks.items()},
            **{s:paired_stats(base,values[0],values[1],mask) for s,mask in masks.items()})
        calib=copy.deepcopy(models[model]); calib['depth']['scale_m']*=multiplier
        (out/f'calib_candidate_{key}.json').write_text(json.dumps(calib,indent=2)+'\n')
        print(f'{name}: {key} multiplier={multiplier:.3f}, validation gain={summaries[key]["validation"]["gain_mm"]:.2f} mm',flush=True)
    chosen=min(summaries,key=lambda k:summaries[k]['validation']['after_score_mm']) if summaries else None
    status='insufficient informative synchronized pairs'
    if chosen:
        best=summaries[chosen]
        supported=not best['search_boundary'] and all(
            s['gain_mm']>=1 and s['gain_mm']>=.1*s['before_score_mm'] and
            s['improved_fraction']>=.6 and s['mean_support_retained']>=.8
            for s in (best['validation'],best['test']))
        status='conditional candidate merits review' if supported else 'no candidate passes screening; not proof of correct calibration'
    for key,(score,cov,metric) in evaluations.items():
        for k,row in enumerate(rows):
            row[key+'_score_mm']=float(score[k]*1000)
            row[key+'_support_retained']=float(cov[k])
            row[key+'_median_residual_mm']=float(metric[k]*1000)
    write_csv(out/'pairs.csv',rows); write_csv(out/'scale_search.csv',curves)
    dt=np.diff(times)
    translation=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)
    rot=np.einsum('nji,njk->nik',poses[:-1,:3,:3],poses[1:,:3,:3])
    angle=np.degrees(np.arccos(np.clip((np.trace(rot,axis1=1,axis2=2)-1)/2,-1,1)))
    speed=translation/dt; angular=angle/dt
    def threshold(x):
        return float(np.median(x)+8*max(1e-6,1.4826*np.median(np.abs(x-np.median(x)))))
    speed_limit,angular_limit=threshold(speed),threshold(angular)
    trajectory=[dict(frame=i+1,translation_m=float(translation[i]),rotation_deg=float(angle[i]),
        speed_m_s=float(speed[i]),angular_deg_s=float(angular[i]),
        review_spike=bool(speed[i]>speed_limit or angular[i]>angular_limit)) for i in range(len(dt))]
    write_csv(out/'trajectory.csv',trajectory)
    summary=dict(scene=name,frames=len(scene),pairs=len(rows),eligible_pairs={s:int(m.sum()) for s,m in masks.items()},
        unscorable_pairs=int((~np.isfinite(base)).sum()),selected_by_validation=chosen,status=status,models=summaries,
        trajectory_review_frames=[r['frame'] for r in trajectory if r['review_spike']],
        speed_spike_threshold_m_s=speed_limit,angular_spike_threshold_deg_s=angular_limit,
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        limitations=['Scale is relative to supplied trajectory translation, not independent metric truth.',
        'RGB-only principal/focal candidates are mathematically indistinguishable in depth-only geometry.',
        'Original pairings retained; timestamp-mismatched pairs audited but not fitted.',
        'Fixed baseline overlap/occlusion support; report coverage because support selection may hide failures.',
        'Score uses original-scale depth units to avoid preferring small scales; metric residual separately reported.',
        'Only local fixed-lag overlap checked, no global loop closure or surveyed-box verification.',
        'Trajectory spikes are review flags, not confirmed localization failures; no pose optimization.',
        'Step-two rotation candidate already used its own validation/test data: these are conditional cross-checks, not wholly unseen end-to-end tests.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(3,1,figsize=(14,11))
    for key,(score,_,_) in evaluations.items():
        idx=[k for k,r in enumerate(rows) if r['lag']==max(args.lags)]
        axes[0].plot([rows[k]['frame_i'] for k in idx],1000*score[idx],label=key,alpha=.7)
    axes[0].set_ylabel('Fixed-support score (mm)'); axes[0].set_xlabel('Source frame'); axes[0].legend()
    for model in models:
        c=[r for r in curves if r['model']==model]
        if c: axes[1].plot([r['multiplier'] for r in c],[r['train_score_mm'] for r in c],marker='.',label=model)
    axes[1].set_xlabel('Depth scale multiplier'); axes[1].set_ylabel('Training score (mm)')
    if curves: axes[1].legend()
    axes[2].plot(np.arange(1,len(scene)),speed); axes[2].axhline(speed_limit,color='red',linestyle='--'); axes[2].set_ylabel('Given trajectory speed (m/s)'); axes[2].set_xlabel('Frame')
    fig.suptitle(name); fig.tight_layout(); fig.savefig(out/'geometry.png',dpi=140); plt.close(fig)
    print(f'{name}: {len(rows)} pairs; {status}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes',nargs='+',default=['scene_a','scene_b','scene_c','scene_d'])
    p.add_argument('--output',type=Path,default=ROOT/'deliverables'/'geometry_audit')
    p.add_argument('--alignment-root',type=Path,default=ROOT/'deliverables'/'alignment_audit')
    p.add_argument('--lags',nargs='+',type=int,default=[1,5,10])
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--min-points',type=int,default=50)
    p.add_argument('--max-offset-ms',type=float,default=5)
    p.add_argument('--min-translation-m',type=float,default=.03)
    p.add_argument('--block-s',type=float,default=5)
    p.add_argument('--guard-s',type=float,default=.5)
    p.add_argument('--occlusion-m',type=float,default=.15)
    p.add_argument('--cap-m',type=float,default=.15)
    p.add_argument('--scale-min',type=float,default=.8)
    p.add_argument('--scale-max',type=float,default=1.2)
    p.add_argument('--scale-steps',type=int,default=21)
    a=p.parse_args()
    if min(a.lags)<1 or a.stride<1 or a.min_points<1 or not 0<a.scale_min<1<a.scale_max or a.scale_steps<3 or a.cap_m<=0 or a.occlusion_m<=0 or a.block_s<=2*a.guard_s or a.guard_s<0 or a.max_offset_ms<0 or a.min_translation_m<0:
        p.error('Invalid bounds, lags, stride, thresholds or time split')
    a.lags=sorted(set(a.lags))
    for name in a.scenes:
        audit(name,a)

if __name__=='__main__': main()
