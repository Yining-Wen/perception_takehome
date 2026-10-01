import sys,json,csv
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_sync import Scene,projected_edges
s=Scene(ROOT/'data/scene_a');p=np.array(json.load(open(ROOT/'variant/scene_a/sync.json'))['pairs'],float)
out=ROOT/'deliverables/scene_a_recheck/scene_a/01_sync';out.mkdir(parents=True,exist_ok=True);rows=[]
for i in range(len(s)):
 if abs(p[i,1]-p[i,0])<1e-6:continue
 j=int(np.argmin(abs(p[:,1]-p[i,0])));dt=float(p[j,1]-p[i,0])
 row=dict(frame=i,original_offset_ms=float(1000*(p[i,1]-p[i,0])),candidate_depth=j,candidate_offset_ms=dt*1000,available_within20ms=abs(dt)<=.02,original_px=None,candidate_px=None)
 if abs(dt)<=.02:
  rgb=s.rgb(i);e=cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(3,3),0),50,120)
  d=cv2.distanceTransform((e==0).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
  for f,key in [(i,'original_px'),(j,'candidate_px')]:
   uv=projected_edges(s,f,.05,.03)
   if len(uv)>=30 and np.count_nonzero(e)>=30:row[key]=float(np.minimum(d[uv[:,1],uv[:,0]],10).mean())
 rows.append(row)
with (out/'nearest_timestamp_recheck.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for label,subset in [('all_mismatched',rows),('persistent_interval',[r for r in rows if 135<=r['frame']<=372])]:
 valid=[r for r in subset if r['original_px'] is not None and r['candidate_px'] is not None]
 summary[label]=dict(frames=len(subset),available=sum(r['available_within20ms'] for r in subset),scorable=len(valid),original_mean_px=float(np.mean([r['original_px'] for r in valid])),candidate_mean_px=float(np.mean([r['candidate_px'] for r in valid])),improved_fraction=float(np.mean([r['candidate_px']<r['original_px'] for r in valid])))
summary['note']='Selected by timestamps only, not minimum image score. Fixed handed-over spatial calibration; remains a conditional check.'
(out/'nearest_timestamp_recheck.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
