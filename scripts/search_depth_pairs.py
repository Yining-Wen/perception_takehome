"""Image-based depth candidate search after freezing calibration; no source mutation."""
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
from check_sync import ROOT, Scene, projected_edges, plt


def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',default='scene_a')
    p.add_argument('--window-ms',type=float,default=1000)
    p.add_argument('--output',type=Path,default=ROOT/'deliverables/depth_pair_search')
    p.add_argument('--calib',type=Path)
    args=p.parse_args()
    if args.window_ms<=0:p.error('window-ms must be positive')
    calib=args.calib or ROOT/'deliverables/ordered_audit'/args.scene/'working_calib.json'
    if not calib.exists():raise FileNotFoundError(f'Freeze calibration first: {calib}')
    s=Scene(ROOT/'data'/args.scene,calib_path=calib)
    pairs=np.asarray(json.loads((ROOT/'variant'/args.scene/'sync.json').read_text())['pairs'],float)
    out=args.output/args.scene;out.mkdir(parents=True,exist_ok=True)
    edges=lru_cache(maxsize=128)(lambda j:projected_edges(s,j,.05,.03))
    rows=[];details=[];mapping=list(range(len(s)));accepted=list(range(len(s)))
    for i in np.flatnonzero(abs(pairs[:,1]-pairs[:,0])>1e-6):
        i=int(i);rgb=s.rgb(i)
        e=cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(3,3),0),50,120)
        distance=cv2.distanceTransform((e==0).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        unique={pairs[i,1]:i}
        for j in np.flatnonzero(abs(pairs[:,1]-pairs[i,0])<=args.window_ms/1000):
            unique.setdefault(pairs[j,1],int(j))
        measured={}
        for j in unique.values():
            uv=edges(j);v=distance[uv[:,1],uv[:,0]]
            score=float(np.minimum(v,10).mean()) if len(v)>=30 and np.count_nonzero(e)>=30 else None
            cell=dict(rgb_frame=i,depth_frame=j,offset_ms=float(1000*(pairs[j,1]-pairs[i,0])),edge_count=len(v),score_px=score,median_px=float(np.median(v)) if len(v) else None,p90_px=float(np.percentile(v,90)) if len(v) else None)
            details.append(cell);measured[j]=cell
        usable=sorted([r for r in measured.values() if r['score_px'] is not None],key=lambda r:(r['score_px'],abs(r['offset_ms']),r['depth_frame']))
        baseline=measured[i];best=usable[0] if usable else None;second=usable[1] if len(usable)>1 else None
        j=best['depth_frame'] if best else None;mapping[i]=j
        gain=baseline['score_px']-best['score_px'] if best and baseline['score_px'] is not None else None
        gap=second['score_px']-best['score_px'] if second else None
        # Preliminary confidence only; exact timestamps are NOT required to win.
        reasons=[]
        if best is None:reasons.append('no_scorable_candidate')
        elif j==i:reasons.append('original_is_best')
        else:
            if gain is None or gain<.5 or gain<.2*baseline['score_px']:reasons.append('weak_improvement')
            if gap is None or gap<.15 or gap<.1*second['score_px']:reasons.append('ambiguous_top_two')
            if best['edge_count']<.8*baseline['edge_count']:reasons.append('edge_support_drop')
        accepted[i]=j if not reasons else None
        rows.append(dict(rgb_frame=i,original_depth=i,original_offset_ms=float(1000*(pairs[i,1]-pairs[i,0])),candidate_count=len(measured),original_score_px=baseline['score_px'],best_depth=j,best_offset_ms=best['offset_ms'] if best else None,best_score_px=best['score_px'] if best else None,second_depth=second['depth_frame'] if second else None,second_score_px=second['score_px'] if second else None,gain_px=gain,top_two_gap_px=gap,original_edge_count=baseline['edge_count'],best_edge_count=best['edge_count'] if best else None,sequence_backwards=False,sequence_reused=False,reliable=not reasons,reason=';'.join(reasons)))
    # Inspect the full chronological best mapping, including untouched good pairs.
    by_frame={r['rgb_frame']:r for r in rows}
    previous=None
    for i,j in enumerate(mapping):
        if j is None:continue
        if previous is not None:
            prev_i,prev_j=previous
            backward=pairs[j,1]<pairs[prev_j,1]-1e-6
            reused=abs(pairs[j,1]-pairs[prev_j,1])<1e-6
            if backward or reused:
                for idx in (prev_i,i):
                    if idx in by_frame:
                        r=by_frame[idx];r['sequence_backwards']|=bool(backward);r['sequence_reused']|=bool(reused)
                        r['reliable']=False;accepted[idx]=None
                        reason='sequence_backwards' if backward else 'sequence_reused'
                        if reason not in r['reason']:r['reason']=(r['reason']+';'+reason).strip(';')
        previous=(i,j)
    write_csv(out/'frames.csv',rows);write_csv(out/'candidates.csv',details)
    for name,value in [('best_pairing.json',mapping),('reviewed_pairing.json',accepted)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    valid=[r for r in rows if r['original_score_px'] is not None and r['best_score_px'] is not None]
    summary=dict(scene=args.scene,calibration=str(calib),window_ms=args.window_ms,mismatched_frames=len(rows),candidate_comparisons=len(details),scorable_comparisons=len(valid),best_changed=sum(r['best_depth'] is not None and r['best_depth']!=r['rgb_frame'] for r in rows),reliable_changed=sum(r['reliable'] for r in rows),no_scorable_candidate=sum(r['best_depth'] is None for r in rows),nonzero_best_offset=sum(r['best_offset_ms'] is not None and abs(r['best_offset_ms'])>.001 for r in rows),original_mean_px=float(np.mean([r['original_score_px'] for r in valid])),best_mean_px=float(np.mean([r['best_score_px'] for r in valid])),warning='Best-of-window scores have selection bias. Raw best mapping is not automatically trusted; reviewed mapping excludes weak, ambiguous or temporally inconsistent changes. Unchanged timestamp-matched frames were not image-validated here. No clock values edited.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(2,1,figsize=(13,7),sharex=True)
    x=[r['rgb_frame'] for r in rows]
    ax[0].plot(x,[r['original_score_px'] for r in rows],'.',label='original');ax[0].plot(x,[r['best_score_px'] for r in rows],'.',label='minimum in window');ax[0].set_ylabel('Edge score (px)');ax[0].legend()
    ax[1].plot(x,[r['best_offset_ms'] for r in rows],'.');ax[1].set_ylabel('Best depth - RGB (ms)');ax[1].set_xlabel('RGB frame');fig.tight_layout();fig.savefig(out/'search.png',dpi=140);plt.close(fig)
    show=sorted([r for r in valid if r['best_depth']!=r['rgb_frame']],key=lambda r:-r['gain_px'])[:4]
    show+= [r for r in valid if r['best_offset_ms'] is not None and abs(r['best_offset_ms'])>.001][:4]
    for i in dict.fromkeys(r['rgb_frame'] for r in show):
        r=by_frame[i];fig,axes=plt.subplots(1,2,figsize=(10,4))
        for a,j,label in [(axes[0],i,'Original'),(axes[1],r['best_depth'],'Best score')]:
            a.imshow(s.rgb(i));uv=edges(j);a.scatter(uv[:,0],uv[:,1],s=1,c='lime');a.set_title(f'{label}: depth {j}, dt {1000*(pairs[j,1]-pairs[i,0]):.1f} ms');a.axis('off')
        fig.suptitle(f'RGB {i}: {r["original_score_px"]:.2f} -> {r["best_score_px"]:.2f}px; reliable={r["reliable"]}');fig.tight_layout();fig.savefig(out/f'overlay_{i:06d}.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
