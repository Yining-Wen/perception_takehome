"""Part 2: detection-driven grounding. Surveyed boxes and query-target labels are never used for inference."""
import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perception.scene import Scene
from perception.geometry import unproject, transform, project


def parse_query(text):
    match = re.fullmatch(r'the (\w+)(?: nearest the (\w+))?', text.lower().strip())
    if not match:
        raise ValueError(f'Unsupported query grammar: {text}')
    return tuple('tv_monitor' if x == 'tv' else x for x in match.groups())


def iou(a,b):
    lo=np.maximum(a[:2],b[:2]); hi=np.minimum(a[2:],b[2:])
    inter=float(np.maximum(hi-lo,0).prod())
    area=lambda x:float(np.maximum(np.asarray(x[2:])-x[:2],0).prod())
    return inter/max(1e-9,area(a)+area(b)-inter)


class Grounder:
    def __init__(self,name,detections,args):
        self.root=args.data_root/name
        self.scene=Scene(self.root)
        self.provenance=[str(self.root/'calib.json')]
        self.original_to_new=json.loads((self.root/'original_to_new_rgb.json').read_text())
        self.index_map=json.loads((self.root/'index_map.json').read_text())
        if len(self.index_map)!=len(self.scene): raise ValueError('Frame/provenance counts differ')
        for i,k in enumerate(self.original_to_new):
            if k is not None and self.index_map[k]['original_rgb_index']!=i:
                raise ValueError('Original-to-filtered mapping inconsistent')
        self.detections=detections
        self.args=args
        self.pairs=np.array(json.loads((self.root/'sync.json').read_text())['pairs'],float)
        if self.pairs.shape!=(len(self.scene),2): raise ValueError('Sync/frame counts differ')
        self.cache={}
        self.diagnostics={}

    def observations(self,frame):
        if frame in self.cache: return self.cache[frame]
        scene=self.scene
        if not 0<=frame<len(self.original_to_new): raise IndexError(frame)
        k=self.original_to_new[frame]
        if k is None:
            self.cache[frame]=[]
            self.diagnostics[frame]={'reason':'flagged_in_part1','filtered_frame':None}
            return []
        # Images have already been paired. Never apply the old depth mapping twice.
        j=self.index_map[k]['original_depth_index']
        delta=float(self.pairs[k,1]-self.pairs[k,0])
        depth=scene.depth_m(k)
        points,_=unproject(depth,scene.depth_cam)
        camera=transform(scene.T_color_depth,points)
        uv,z=project(camera,scene.color)
        h,w=scene.color['height'],scene.color['width']
        good=(z>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
        camera,uv,z=camera[good],uv[good],z[good]
        pix=np.rint(uv).astype(int); flat=pix[:,1]*w+pix[:,0]
        # Keep nearest point per RGB pixel after projection.
        order=np.argsort(z); _,first=np.unique(flat[order],return_index=True)
        keep=order[first]; camera,uv,z=camera[keep],uv[keep],z[keep]
        dets=[]
        for detection_index,d in sorted(enumerate(self.detections.get(str(frame),[])),key=lambda item:-item[1]['score']):
            box=np.clip(np.array(d['box'],float),[0,0,0,0],[w,h,w,h])
            if d['score']<self.args.min_detection or np.any(box[2:]-box[:2]<4): continue
            if any(d['label']==x['label'] and iou(box,x['box'])>.5 for x in dets): continue
            center=(box[:2]+box[2:])/2; half=(box[2:]-box[:2])*.3
            inside=(np.abs(uv-center)<=half).all(1)
            n=int(inside.sum())
            if n<12: continue
            local=camera[inside]; zz=local[:,2]
            near=float(np.percentile(zz,25))
            band=np.abs(zz-near)<max(.08,.025*near)
            if band.sum()<8: continue
            foreground=local[band]
            median=np.median(foreground,axis=0)
            # A real observed surface point, not a mean in empty space.
            point=foreground[np.argmin(np.linalg.norm(foreground-median,axis=1))]
            world=transform(scene.pose(k),point[None])[0]
            spread=float(np.median(np.linalg.norm(foreground-median,axis=1)))
            coverage=min(1.,n/max(1.,float((2*half).prod())))
            quality=float(np.clip(coverage,0,1)**.3 * np.exp(-spread/.8))
            dets.append(dict(frame=frame,filtered_frame=k,index=detection_index,label=d['label'],score=float(d['score']),box=box.tolist(),
                             point=world.tolist(),depth_quality=quality,depth_points=int(band.sum()),
                             depth_frame=j,offset_ms=delta*1000,spread_m=spread))
        self.cache[frame]=dets
        self.diagnostics[frame]={'filtered_frame':k,'depth_frame':j,'offset_ms':delta*1000,'lifted_detections':len(dets)}
        return dets

    def candidates(self,frame,label):
        current=[copy.deepcopy(d) for d in self.observations(frame) if d['label']==label]
        # Supporting observations cannot create a currently missing target.
        neighbors=[f for f in range(max(0,frame-10),min(len(self.original_to_new),frame+11),2) if f!=frame]
        context=[d for f in neighbors for d in self.observations(f) if d['label']==label]
        for d in current:
            p=np.array(d['point']); supporting={}
            for other in context:
                dist=float(np.linalg.norm(np.array(other['point'])-p))
                if dist<.45:
                    f=other['frame']
                    if f not in supporting or other['score']>supporting[f]['score']: supporting[f]=other
            d['support_frames']=len(supporting)
            d['quality']=float(d['score']* (.6+.4*d['depth_quality'])*(.75+.25*min(1,len(supporting)/3)))
            # Bounded local temporal smoothing; never move far from visible surface.
            if len(supporting)>=2:
                pts=np.array([d['point']]+[o['point'] for o in supporting.values()])
                median=np.median(pts,axis=0)
                if np.linalg.norm(median-p)<.25: d['point']=median.tolist()
        return sorted(current,key=lambda d:-d['quality'])

    def answer(self,frame,label,anchor):
        candidates=self.candidates(frame,label)
        trace={'target_label':label,'anchor_label':anchor,'candidates':candidates}
        def decline(reason):
            trace['decision']=reason
            return {'goal_world':None,'confidence':0.},trace
        if not candidates: return decline(self.diagnostics.get(frame,{}).get('reason','no_current_detection_with_reliable_depth'))
        ambiguity=1.
        if anchor:
            anchors=self.candidates(frame,anchor)
            if not anchors: return decline('reference_object_not_grounded_in_current_frame')
            if len(anchors)>1 and anchors[1]['quality']>.8*anchors[0]['quality'] and np.linalg.norm(np.array(anchors[0]['point'])-anchors[1]['point'])>.45:
                return decline('ambiguous_reference_object')
            reference=anchors[0]
            for d in candidates: d['reference_distance_m']=float(np.linalg.norm(np.array(d['point'])-reference['point']))
            candidates.sort(key=lambda d:d['reference_distance_m'])
            trace['reference']=reference
            if len(candidates)>1:
                gap=candidates[1]['reference_distance_m']-candidates[0]['reference_distance_m']
                if gap<.25: return decline('nearest_relation_not_separated_by_25cm')
                ambiguity=float(np.clip(gap/.8,.4,1))
            ambiguity*=min(1.,reference['quality']/.35)
        elif len(candidates)>1:
            other=next((d for d in candidates[1:] if np.linalg.norm(np.array(d['point'])-candidates[0]['point'])>.45),None)
            if other:
                ratio=other['quality']/max(candidates[0]['quality'],1e-9)
                if ratio>.85: return decline('multiple_similarly_supported_objects')
                ambiguity=max(.4,1-.6*ratio)
        best=candidates[0]
        confidence=float(np.clip(best['quality']*ambiguity,0,.95))
        if best['score']<.18 or confidence<self.args.min_confidence:
            return decline('insufficient_detection_or_grounding_confidence')
        trace['decision']='answered'; trace['selected_detection']=best['index']
        return {'goal_world':best['point'],'confidence':confidence},trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'answers.json')
    p.add_argument('--diagnostics',type=Path,default=ROOT/'deliverables/grounding_filtered')
    p.add_argument('--data-root',type=Path,default=ROOT/'data/repaired_filtered')
    p.add_argument('--min-detection',type=float,default=.12)
    p.add_argument('--min-confidence',type=float,default=.12)
    args=p.parse_args()
    if not 0<=args.min_detection<=1 or not 0<=args.min_confidence<=1: p.error('Thresholds must be in [0,1]')
    queries=json.loads((ROOT/'variant/queries.json').read_text())['scenes']
    detections=json.loads((ROOT/'variant/detections.json').read_text())['scenes']
    answers={}; traces={}; manifest={}
    args.diagnostics.mkdir(parents=True,exist_ok=True)
    for name,qs in queries.items():
        g=Grounder(name,detections[name],args)
        answers[name]={}; traces[name]={}
        for q in qs:
            label,anchor=parse_query(q['text'])
            answers[name][q['id']]={}; traces[name][q['id']]={}
            for f in q['frames']:
                answer,trace=g.answer(f,label,anchor)
                answers[name][q['id']][str(f)]=answer; traces[name][q['id']][str(f)]=trace
        (args.diagnostics/f'{name}_calib.json').write_text(json.dumps(g.scene.calib,indent=2)+'\n')
        manifest[name]={'calibration_sources':g.provenance,'data_root':str(g.root),'processed_frames':len(g.cache),'pairing_policy':'Use materialized depth directly; exposure offset recorded for provenance only','min_detection':args.min_detection,'min_confidence':args.min_confidence,
                        'index_convention':'Output/query/detection IDs are original RGB indices; image and pose accesses use filtered indices.',
                        'answered':sum(a['goal_world'] is not None for q in answers[name].values() for a in q.values()),
                        'requested':sum(len(q['frames']) for q in qs),'frame_diagnostics':g.diagnostics}
        print(name,manifest[name]['answered'],'/',manifest[name]['requested'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(answers,indent=2,allow_nan=False)+'\n')
    (args.diagnostics/'decisions.json').write_text(json.dumps(traces,indent=2,allow_nan=False)+'\n')
    (args.diagnostics/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
