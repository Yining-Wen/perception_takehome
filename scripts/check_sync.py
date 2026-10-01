"""Full-sequence pairing audit. Proposals are evidence for review, not calibration truth."""
import argparse
import csv
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'perception-mpl'))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.scene import Scene
from perception.geometry import unproject, transform, project


def depth_edges(depth, jump_m=0.05, jump_relative=0.03):
    """Foreground-side discontinuities; zero depth is invalid, not an edge."""
    edge = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        a = depth[:-1, :] if axis == 0 else depth[:, :-1]
        b = depth[1:, :] if axis == 0 else depth[:, 1:]
        good = (a > 0) & (b > 0)
        discontinuity = good & (np.abs(a-b) > np.maximum(jump_m, jump_relative*np.minimum(a,b)))
        # Only foreground side of each jump: background edges are often occluded.
        if axis == 0:
            edge[:-1, :] |= discontinuity & (a < b)
            edge[1:, :] |= discontinuity & (b < a)
        else:
            edge[:, :-1] |= discontinuity & (a < b)
            edge[:, 1:] |= discontinuity & (b < a)
    return edge


def projected_edges(scene, index, jump_m, jump_relative):
    """Project valid depth discontinuities, rejecting invalid boundaries and occlusion."""
    depth = scene.depth_m(index)
    edge = depth_edges(depth, jump_m, jump_relative)
    points, source = unproject(depth, scene.depth_cam)
    uv, z = project(transform(scene.T_color_depth, points), scene.color)
    h, w = scene.color['height'], scene.color['width']
    valid = (z > 0) & np.isfinite(uv).all(axis=1)
    valid &= (uv[:,0] >= 0) & (uv[:,0] < w-1) & (uv[:,1] >= 0) & (uv[:,1] < h-1)
    uv = np.rint(uv[valid]).astype(int)
    z = z[valid]
    src = source[valid].astype(int)
    flat = uv[:,1]*w + uv[:,0]
    zbuffer = np.full(h*w, np.inf)
    np.minimum.at(zbuffer, flat, z)
    visible = z <= zbuffer[flat] + 0.01
    keep = visible & edge[src[:,1], src[:,0]]
    return np.unique(uv[keep], axis=0)


def stats(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return {'median': float(np.median(x)), 'p95': float(np.percentile(x,95)), 'max':float(x.max())} if x.size else None


def audit(name, args):
    scene = Scene(ROOT/'data'/name)
    sync = json.loads((ROOT/'variant'/name/'sync.json').read_text())
    pairs = np.asarray(sync['pairs'], float)
    if pairs.shape != (len(scene),2):
        raise ValueError(f'{name}: frame and pairing counts disagree')
    tc, td = pairs.T
    if np.any(np.diff(tc) <= 0):
        raise ValueError(f'{name}: color timestamps must be strictly increasing')
    out = args.output/name
    out.mkdir(parents=True, exist_ok=True)
    get_edges = lru_cache(maxsize=64)(lambda j: projected_edges(scene,j,args.jump_m,args.jump_relative))
    poses = np.array([scene.pose(i) for i in range(len(scene))])
    dt = np.diff(tc)
    speed = np.r_[np.nan, np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)/dt]
    rotations = np.einsum('nji,njk->nik',poses[:-1,:3,:3],poses[1:,:3,:3])
    angular = np.r_[np.nan, np.degrees(np.arccos(np.clip((np.trace(rotations,axis1=1,axis2=2)-1)/2,-1,1)))/dt]
    rows, candidates = [], []
    for i in range(len(scene)):
        rgb = scene.rgb(i)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb_edges = cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),args.canny_low,args.canny_high)
        distance = cv2.distanceTransform((rgb_edges == 0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        # Search by actual depth time, not file index; deduplicate reused exposures.
        indices = np.flatnonzero(np.abs(td-tc[i]) <= args.window_s).tolist()
        unique = {td[i]:i}
        for j in indices:
            unique.setdefault(td[j],j)
        measured = {}
        for j in unique.values():
            uv = get_edges(j)
            errors = distance[uv[:,1],uv[:,0]]
            sufficient = len(errors) >= args.min_edges and np.count_nonzero(rgb_edges) >= args.min_edges
            median = float(np.median(errors)) if sufficient else float('nan')
            p90 = float(np.percentile(errors,90)) if sufficient else float('nan')
            # Clipped mean reduces domination by edges without a visual counterpart.
            score = float(np.minimum(errors,10).mean()) if sufficient else float('nan')
            measured[j] = (score,median,p90,len(errors))
            candidates.append(dict(frame=i,depth_index=j,depth_t=td[j],offset_ms=1000*(td[j]-tc[i]),score_px=score,median_px=median,p90_px=p90,edge_count=len(errors)))
        base = measured[i][0]
        usable = [j for j in measured if np.isfinite(measured[j][0])]
        best = min(usable,key=lambda j:(measured[j][0],abs(td[j]-tc[i]))) if usable else i
        best_score = measured[best][0]
        gain = base-best_score
        closer = abs(td[best]-tc[i]) < abs(td[i]-tc[i])-1e-6
        supported = best != i and closer and np.isfinite(gain) and gain >= args.min_gain_px and gain >= args.min_gain_fraction*base and measured[best][3] >= 0.8*measured[i][3]
        rows.append(dict(frame=i,color_t=tc[i],depth_t=td[i],offset_ms=1000*(td[i]-tc[i]),rgb_interval_ms=1000*(tc[i]-tc[i-1]) if i else np.nan,depth_interval_ms=1000*(td[i]-td[i-1]) if i else np.nan,speed_m_s=speed[i],angular_deg_s=angular[i],original_score_px=base,best_depth_index=best,best_offset_ms=1000*(td[best]-tc[i]),best_score_px=best_score,gain_px=gain,original_edge_count=measured[i][3],best_edge_count=measured[best][3],review_candidate=bool(supported)))
        if i % 250 == 0:
            print(f'{name}: {i}/{len(scene)}',flush=True)
    for filename, records in [('frames.csv',rows),('candidates.csv',candidates)]:
        with (out/filename).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    proposed = [r for r in rows if r['review_candidate']]
    summary = dict(scene=name,frames=len(scene),nonzero_offset_frames=int(np.count_nonzero(np.abs(td-tc)>1e-6)),reused_depth_rows=int(len(td)-len(np.unique(td))),backward_depth_steps=int(np.count_nonzero(np.diff(td)<0)),rgb_gaps_over_1_5x_median=int(np.count_nonzero(dt>1.5*np.median(dt))),absolute_offset_ms=stats(1000*np.abs(td-tc)),original_score_px=stats([r['original_score_px'] for r in rows]),unscorable_frames=sum(not np.isfinite(r['original_score_px']) for r in rows),review_candidate_count=len(proposed),review_frames=[r['frame'] for r in proposed],parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},warning='Candidate search is exploratory, not independent validation. Fixed handed-over calibration may bias scores. No source files modified; no clock offset or calibration inferred.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(4,1,figsize=(14,11),sharex=True)
    axes[0].plot(1000*(td-tc)); axes[0].set_ylabel('Depth - RGB (ms)')
    axes[1].plot(speed,label='translation m/s'); axes[1].legend(loc='upper left')
    twin=axes[1].twinx(); twin.plot(angular,color='orange',alpha=.5,label='rotation deg/s'); twin.legend(loc='upper right')
    axes[2].plot([r['original_score_px'] for r in rows],label='original')
    axes[2].plot([r['best_score_px'] for r in rows],alpha=.7,label='best nearby (exploratory)'); axes[2].set_ylabel('Edge score (px)'); axes[2].legend()
    axes[3].plot([r['gain_px'] for r in rows]); axes[3].scatter([r['frame'] for r in proposed],[r['gain_px'] for r in proposed],color='red',s=12,label='review candidate'); axes[3].legend(); axes[3].set_ylabel('Improvement (px)'); axes[3].set_xlabel('Frame index')
    fig.suptitle(name); fig.tight_layout(); fig.savefig(out/'timeline.png',dpi=140); plt.close(fig)
    # Visual review includes improvements and worst baseline frames.
    selected = sorted(proposed,key=lambda r:r['gain_px'],reverse=True)[:6]
    selected += sorted([r for r in rows if np.isfinite(r['original_score_px'])],key=lambda r:r['original_score_px'],reverse=True)[:4]
    for i in dict.fromkeys(r['frame'] for r in selected):
        r=rows[i]; fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,j,title in [(axes[0],i,'Original'),(axes[1],r['best_depth_index'],'Best candidate')]:
            ax.imshow(scene.rgb(i)); uv=get_edges(j); ax.scatter(uv[:,0],uv[:,1],s=1,c='lime'); ax.set_title(f'{title}: depth {j}, dt={1000*(td[j]-tc[i]):.1f} ms'); ax.axis('off')
        fig.suptitle(f'{name} RGB {i}: score {r["original_score_px"]:.2f} -> {r["best_score_px"]:.2f} px'); fig.tight_layout(); fig.savefig(out/f'overlay_{i:06d}.png',dpi=140); plt.close(fig)
    print(f'{name}: complete; {len(proposed)} candidates for review',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes',nargs='+',default=['scene_a','scene_b','scene_c','scene_d'])
    p.add_argument('--output',type=Path,default=ROOT/'deliverables'/'sync_audit')
    p.add_argument('--window-s',type=float,default=.35)
    p.add_argument('--jump-m',type=float,default=.05)
    p.add_argument('--jump-relative',type=float,default=.03)
    p.add_argument('--min-edges',type=int,default=30)
    p.add_argument('--canny-low',type=int,default=50)
    p.add_argument('--canny-high',type=int,default=120)
    p.add_argument('--min-gain-px',type=float,default=.5)
    p.add_argument('--min-gain-fraction',type=float,default=.2)
    args=p.parse_args()
    if args.window_s <= 0 or args.min_edges < 1:
        p.error('window-s and min-edges must be positive')
    for name in args.scenes:
        audit(name,args)

if __name__ == '__main__':
    main()
