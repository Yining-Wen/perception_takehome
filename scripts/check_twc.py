"""Check camera-to-world poses; optionally inspect flagged intervals using depth."""
import argparse
import json
from pathlib import Path

import numpy as np
from check_sync import ROOT, Scene, plt
from parameter_audit import table, dump, rotation_error, cloud, icp
from check_geometry import sample_depth, relative_depth
from perception.geometry import transform


def basic(scene):
    records=[]
    for i in range(len(scene)):
        T=scene.pose(i);finite=bool(np.isfinite(T).all());R=T[:3,:3]
        orth=float(np.linalg.norm(R.T@R-np.eye(3))) if finite else None
        det=float(np.linalg.det(R)) if finite else None
        bottom=bool(finite and np.allclose(T[3],[0,0,0,1],atol=1e-6))
        dt=scene.frames[i]['t']-scene.frames[i-1]['t'] if i else None
        dist=angle=None
        if i and finite and np.isfinite(scene.pose(i-1)).all():
            dist=float(np.linalg.norm(T[:3,3]-scene.pose(i-1)[:3,3]))
            angle=rotation_error(R,scene.pose(i-1)[:3,:3])
        records.append(dict(frame=i,finite=finite,orthogonality_error=orth,determinant=det,bottom_row_valid=bottom,
            matrix_valid=bool(finite and bottom and orth<1e-3 and abs(det-1)<1e-3),dt_s=dt,
            timestamp_valid=bool(np.isfinite(scene.frames[i]['t']) and (i==0 or dt>0)),translation_m=dist,rotation_deg=angle,
            speed_m_s=dist/dt if dist is not None and dt>0 else None,
            angular_deg_s=angle/dt if angle is not None and dt>0 else None))
    def limit(key):
        x=np.array([r[key] for r in records if r[key] is not None and np.isfinite(r[key])])
        return float(np.median(x)+8*max(1e-6,1.4826*np.median(abs(x-np.median(x))))) if len(x) else None
    linear,angular=limit('speed_m_s'),limit('angular_deg_s')
    for r in records:
        r['motion_flag']=bool((r['speed_m_s'] is not None and linear is not None and r['speed_m_s']>linear) or (r['angular_deg_s'] is not None and angular is not None and r['angular_deg_s']>angular))
    return records,linear,angular


def inspect_pair(scene,pairs,i,j,stride,tolerance):
    row=dict(frame_i=i,frame_j=j,status='excluded',offset_i_ms=float(1000*(pairs[i,1]-pairs[i,0])),offset_j_ms=float(1000*(pairs[j,1]-pairs[j,0])),
        reprojection_median_m=None,reprojection_p90_m=None,overlap=None,foreground_conflict_fraction=None,
        icp_fitness=None,icp_rmse_m=None,icp_condition=None,icp_converged=None,icp_reliable=False,
        rotation_difference_deg=None,translation_difference_m=None,reverse_cycle_translation_m=None,reverse_cycle_rotation_deg=None)
    if max(abs(row['offset_i_ms']),abs(row['offset_j_ms']))>tolerance:
        row['status']='exposure_mismatch';return row
    if abs(pairs[i,1]-pairs[j,1])<1e-6:
        row['status']='same_depth_exposure';return row
    a,b=scene.depth_m(i),scene.depth_m(j)
    p,pn=cloud(a,scene.depth_cam,stride);q,qn=cloud(b,scene.depth_cam,stride)
    if min(len(p),len(q))<30:
        row['status']='insufficient_depth';return row
    expected=relative_depth(scene.pose(i),scene.pose(j),scene.T_color_depth)
    if not np.isfinite(expected).all():row['status']='invalid_pose';return row
    errors=[];overlaps=[];conflicts=[]
    for pts,depth,T in [(p,b,expected),(q,a,np.linalg.inv(expected))]:
        measured,z,valid=sample_depth(depth,transform(T,pts),scene.depth_cam)
        overlaps.append(float(valid.mean()))
        # Do not reject large depth residuals: they are the signal under test.
        errors.extend(np.abs(z[valid]-measured[valid]).tolist())
        if valid.any():conflicts.append(float(np.mean(abs(z[valid]-measured[valid])>.15)))
    row['status']='evaluated';row['overlap']=float(np.mean(overlaps))
    if errors:
        row['reprojection_median_m']=float(np.median(errors));row['reprojection_p90_m']=float(np.percentile(errors,90))
        row['foreground_conflict_fraction']=float(np.mean(conflicts))
    result=icp(p,q,qn);reverse=icp(q,p,pn)
    if result:
        T=np.array(result['T']);row.update(icp_fitness=result['fitness'],icp_rmse_m=result['point_plane_rmse_m'],icp_condition=result['condition'],icp_converged=result['converged'],
            rotation_difference_deg=rotation_error(T[:3,:3],expected[:3,:3]),translation_difference_m=float(np.linalg.norm(T[:3,3]-expected[:3,3])))
        if reverse:
            cycle=np.array(reverse['T'])@T
            row['reverse_cycle_translation_m']=float(np.linalg.norm(cycle[:3,3]));row['reverse_cycle_rotation_deg']=rotation_error(cycle[:3,:3],np.eye(3))
            row['icp_reliable']=bool(result['converged'] and reverse['converged'] and min(result['fitness'],reverse['fitness'])>.5 and max(result['condition'],reverse['condition'])<1000 and row['reverse_cycle_translation_m']<.05 and row['reverse_cycle_rotation_deg']<2)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-root',type=Path,default=ROOT/'data/repaired/scene_a')
    p.add_argument('--output',type=Path,default=ROOT/'deliverables/twc_audit/scene_a')
    p.add_argument('--basic-only',action='store_true')
    p.add_argument('--frames',type=int,nargs='+',help='Additional suspected transition end frames')
    p.add_argument('--radius',type=int,default=8)
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--max-offset-ms',type=float,default=1e-3)
    args=p.parse_args()
    if args.radius<1 or args.stride<1 or args.max_offset_ms<0:p.error('Invalid radius, stride or tolerance')
    s=Scene(args.scene_root);out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.frames and any(i<0 or i>=len(s) for i in args.frames):p.error('Frame outside sequence')
    records,linear,angular=basic(s);table(out/'poses.csv',records)
    flags=[r['frame'] for r in records if r['motion_flag'] or not r['matrix_valid'] or not r['timestamp_valid']]
    selected=sorted(set(flags+(args.frames or [])));rows=[]
    if not args.basic_only:
        pairs=np.asarray(json.loads((args.scene_root/'sync.json').read_text())['pairs'],float)
        if pairs.shape!=(len(s),2) or not np.isfinite(pairs).all():raise ValueError('Need finite paired timestamps for every frame')
        indices=set()
        for center in selected:
            lo=max(0,center-args.radius);hi=min(len(s)-1,center+args.radius)
            for lag in (1,3,5):
                indices.update((i,i+lag) for i in range(lo,hi-lag+1))
        for i,j in sorted(indices):
            if not records[i]['matrix_valid'] or not records[j]['matrix_valid']:continue
            rows.append(inspect_pair(s,pairs,i,j,args.stride,args.max_offset_ms))
        table(out/'local_geometry.csv',rows)
    summary=dict(scene_root=str(args.scene_root),frames=len(s),invalid_matrices=[r['frame'] for r in records if not r['matrix_valid']],
        invalid_timestamps=[r['frame'] for r in records if not r['timestamp_valid']],motion_flags=flags,
        speed_threshold_m_s=linear,angular_threshold_deg_s=angular,basic_only=args.basic_only,
        inspected_centers=selected,local_pairs=len(rows),evaluated_pairs=sum(r['status']=='evaluated' for r in rows),
        reliable_icp_pairs=sum(r['icp_reliable'] for r in rows),
        limitations=['No scale, intrinsic, extrinsic or pose changes.',
        'Matrix legality and smoothness do not prove trajectory accuracy.',
        'Geometry checks are local to flagged/explicitly requested intervals, not global loop-closure validation.',
        'Exposure mismatch pairs excluded; depth pairing itself was image-selected and is not independent truth.',
        'Reprojection includes occlusion/dynamics; residuals are not uniquely attributable to trajectory.',
        'ICP starts from identity, can fail on large motion/weak geometry; screening is heuristic.'])
    dump(out/'summary.json',summary)
    fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True)
    for a,key,threshold in [(ax[0],'speed_m_s',linear),(ax[1],'angular_deg_s',angular)]:
        a.plot([r['frame'] for r in records],[r[key] for r in records]);a.set_ylabel(key)
        if threshold is not None:a.axhline(threshold,color='red',linestyle='--')
        for i in selected:a.axvline(i,color='orange',alpha=.3)
    ax[1].set_xlabel('Frame');fig.tight_layout();fig.savefig(out/'motion.png',dpi=140);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
