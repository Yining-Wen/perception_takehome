"""Part 3: single-frame lifting, appearance-aware tracklets and conservative merging.
No surveyed boxes or query-target labels are used for inference.
"""
import argparse
import heapq
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ground_queries import Grounder, ROOT, iou


def appearance(rgb,box,orb):
    h,w=rgb.shape[:2];x0,y0,x1,y1=np.array(box,int)
    patch=rgb[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]
    patch=cv2.resize(patch,(96,96))
    hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256]).ravel()
    hist=hist/max(hist.sum(),1)
    _,desc=orb.detectAndCompute(cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY),None)
    return hist,desc


def similarity(a,b):
    color=float(np.sqrt(a[0]*b[0]).sum())
    da,db=a[1],b[1];local=0.
    if da is not None and db is not None and min(len(da),len(db))>=4:
        matches=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(da,db)
        local=sum(m.distance<48 for m in matches)/min(len(da),len(db))
    return .7*color+.3*local


def center(track):return np.median([o['point'] for o in track['obs']],axis=0)
def gallery(track):
    obs=track['obs'];ids=np.unique(np.linspace(0,len(obs)-1,min(5,len(obs))).astype(int))
    return [obs[i]['appearance'] for i in ids]
def appearance_to(track,obs):return max(similarity(a,obs['appearance']) for a in gallery(track))


def merge_eligible(a,b):
    if a['label']!=b['label'] or a['frames'] & b['frames']:return None
    dist=float(np.linalg.norm(center(a)-center(b)))
    if dist>.65:return None
    app=max(similarity(x,y) for x in gallery(a) for y in gallery(b))
    if app<.62:return None
    pts=np.array([o['point'] for o in a['obs']+b['obs']]);c=np.median(pts,axis=0)
    if np.percentile(np.linalg.norm(pts-c,axis=1),90)>.75:return None
    return dist+.6*(1-app)


def associate(active,observations,frame):
    """Hungarian one-to-one assignment with explicit unmatched columns."""
    if not observations:return []
    cost=np.full((len(observations),len(active)+len(observations)),.95)
    cost[:,:len(active)]=1e6
    for i,o in enumerate(observations):
        for j,t in enumerate(active):
            if o['label']!=t['label']:continue
            dist=np.linalg.norm(np.asarray(o['point'])-center(t))
            if dist>.75:continue
            app=appearance_to(t,o)
            if app<.35:continue
            overlap=iou(np.asarray(o['box']),np.asarray(t['obs'][-1]['box']))
            cost[i,j]=.55*dist/.75+.35*(1-app)+.1*(1-overlap)
    # Near ties between existing tracks are not forced into identities.
    for i in range(len(observations)):
        vals=np.sort(cost[i,:len(active)])
        if len(vals)>1 and vals[1]<.95 and vals[1]-vals[0]<.06:cost[i,:len(active)]=1e6
    rr,cc=linear_sum_assignment(cost)
    return [(int(i),int(j),float(cost[i,j])) for i,j in zip(rr,cc) if j<len(active) and cost[i,j]<.95]


def build(name,detections,args):
    g=Grounder(name,detections,SimpleNamespace(data_root=args.data_root,min_detection=.18,min_confidence=.12))
    tracks=[];events=[];next_id=1;skipped=[]
    orb=cv2.ORB_create(nfeatures=96,edgeThreshold=8,fastThreshold=10)
    for f in range(len(g.original_to_new)):
        k=g.original_to_new[f]
        if k is None:skipped.append(dict(frame=f,reason='flagged_in_part1'));continue
        observations=g.observations(f)
        # Avoid caching every projected-frame result in addition to track observations.
        g.cache.pop(f,None)
        if not observations:continue
        rgb=g.scene.rgb(k)
        for o in observations:o['appearance']=appearance(rgb,o['box'],orb)
        active=[t for t in tracks if f-t['last']<=10]
        matches=associate(active,observations,f);used=set()
        for i,j,cost in matches:
            o=observations[i];t=active[j];used.add(i)
            t['obs'].append(o);t['frames'].add(f);t['last']=f
            events.append(dict(frame=f,detection_index=o['index'],track=t['id'],action='associate',cost=cost))
        for i,o in enumerate(observations):
            if i in used:continue
            t=dict(id=next_id,label=o['label'],obs=[o],frames={f},last=f);tracks.append(t);next_id+=1
            events.append(dict(frame=f,detection_index=o['index'],track=t['id'],action='new_track',cost=None))
        if f%200==0:print(name,'frame',f,'tracks',len(tracks),flush=True)
    original_count=len(tracks);merges=[]
    # Versioned priority queue: recompute only pairs affected by a merge.
    live={t['id']:t for t in tracks};versions={i:0 for i in live};queue=[]
    centers={i:center(t) for i,t in live.items()}
    def offer(i,j):
        if i>j:i,j=j,i
        if live[i]['label']!=live[j]['label'] or np.linalg.norm(centers[i]-centers[j])>.65:return
        score=merge_eligible(live[i],live[j])
        if score is not None:heapq.heappush(queue,(score,i,j,versions[i],versions[j]))
    ids=sorted(live)
    for n,i in enumerate(ids):
        for j in ids[n+1:]:offer(i,j)
    while queue:
        score,i,j,vi,vj=heapq.heappop(queue)
        if i not in live or j not in live or versions[i]!=vi or versions[j]!=vj:continue
        a,b=live[i],live[j]
        merges.append(dict(kept_id=i,removed_id=j,score=score))
        a['obs']=sorted(a['obs']+b['obs'],key=lambda o:(o['frame'],o['index']))
        a['frames']|=b['frames'];a['last']=max(a['last'],b['last'])
        del live[j];versions[i]+=1;centers[i]=center(a)
        for other in sorted(live):
            if other!=i:offer(i,other)
    tracks=[live[i] for i in sorted(live)]
    instances=[];rejected=[];diagnostics=[]
    for t in tracks:
        obs=t['obs'];c=center(t)
        if len(t['frames'])<3:
            rejected.append(dict(track=t['id'],reason='fewer_than_three_frames',observations=[[o['frame'],o['index']] for o in obs]));continue
        # Median is robust; keep all claimed observations visible in diagnostics.
        instances.append(dict(id=t['id'],label=t['label'],center_world=c.tolist(),observations=[[o['frame'],o['index']] for o in obs]))
        diagnostics.append(dict(id=t['id'],label=t['label'],frames=len(t['frames']),first=min(t['frames']),last=max(t['frames']),radius_p90_m=float(np.percentile([np.linalg.norm(np.asarray(o['point'])-c) for o in obs],90))))
    out=args.diagnostics/name;out.mkdir(parents=True,exist_ok=True)
    for file,obj in [('associations.json',events),('merges.json',merges),('rejected_tracks.json',rejected),('instances_audit.json',diagnostics),('skipped_frames.json',skipped)]:
        (out/file).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    summary=dict(source=str(g.root),original_frames=len(g.original_to_new),retained_frames=len(g.scene),tracklets=original_count,merges=len(merges),instances=len(instances),rejected_tracks=len(rejected),claimed_observations=sum(len(x['observations']) for x in instances),settings=dict(min_detection=.18,min_track_frames=3,active_gap_original_frames=10,association_radius_m=.75,merge_radius_m=.65,merge_appearance_min=.62,merge_p90_radius_m=.75),limitations=['Handcrafted appearance is not viewpoint invariant.','Same-frame conflict prevents merging but does not guarantee identity.','No trajectory or scale corrections; no true object-center recovery.','No explicit splitting after a mistaken association.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(name,summary,flush=True)
    return instances


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data/repaired_filtered')
    p.add_argument('--output',type=Path,default=ROOT/'instances.json')
    p.add_argument('--diagnostics',type=Path,default=ROOT/'deliverables/mapping_audit')
    args=p.parse_args();cv2.setRNGSeed(7)
    detections=json.loads((ROOT/'variant/detections.json').read_text())['scenes']
    results={name:build(name,d,args) for name,d in detections.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
