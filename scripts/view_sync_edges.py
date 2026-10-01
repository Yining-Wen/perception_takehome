"""Export exact step-one edge masks and readable depth previews for every frame."""
import argparse
import html
import json

import cv2
import numpy as np
from check_sync import ROOT, Scene, depth_edges, projected_edges


def save(path, data):
    if not cv2.imwrite(str(path), data):
        raise OSError(f'Could not write {path}')


def export(name,args):
    scene=Scene(ROOT/'data'/name)
    pairs=json.loads((ROOT/'variant'/name/'sync.json').read_text())['pairs']
    frames=list(range(len(scene))) if args.frames is None else args.frames
    if any(f<0 or f>=len(scene) for f in frames): raise ValueError('Frame index outside scene')
    out=args.output/name; out.mkdir(parents=True,exist_ok=True)
    records=[]
    for i in frames:
        j=i if args.depth_frame is None else args.depth_frame
        if not 0<=j<len(scene): raise ValueError('Depth frame outside scene')
        rgb=cv2.cvtColor(scene.rgb(i),cv2.COLOR_RGB2BGR)
        depth=scene.depth_m(j)
        gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),50,120)
        native=depth_edges(depth)
        valid=depth>0
        lo,hi=(np.percentile(depth[valid],[2,98]) if valid.any() else (0.,1.))
        if args.depth_range: lo,hi=args.depth_range
        hi=max(float(hi),float(lo)+.001); lo=float(lo)
        # TURBO: near=blue, far=red; only the visualization is normalized.
        normalized=np.clip((depth-lo)/(hi-lo),0,1)
        colored=cv2.applyColorMap(np.rint(normalized*255).astype('uint8'),cv2.COLORMAP_TURBO)
        colored[~valid]=(80,80,80)
        projected=np.zeros(edges.shape,'uint8')
        uv=projected_edges(scene,j,.05,.03)
        projected[uv[:,1],uv[:,0]]=255
        overlay=rgb.copy(); overlay[projected>0]=(0,255,0)
        depth_overlay=colored.copy(); depth_overlay[native]=(255,255,255)
        prefix=f'rgb_{i:06d}_depth_{j:06d}'
        save(out/f'{prefix}_rgb_edges.png',edges)
        save(out/f'{prefix}_depth_edges.png',native.astype('uint8')*255)
        save(out/f'{prefix}_projected_edges.png',projected)
        save(out/f'{prefix}_depth_color.png',colored)
        h,w=edges.shape
        def panel(im,title):
            if im.ndim==2: im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
            im=cv2.resize(im,(w*2,h*2),interpolation=cv2.INTER_NEAREST)
            bar=np.full((38,w*2,3),245,'uint8')
            cv2.putText(bar,title,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.49,(20,20,20),1,cv2.LINE_AA)
            return np.vstack([bar,im])
        top=np.hstack([panel(rgb,'RGB'),panel(edges,'RGB Canny (white = edge)'),panel(overlay,'RGB + projected depth edges (green)')])
        bottom=np.hstack([panel(colored,f'Depth: blue {lo:.2f}m -> red {hi:.2f}m'),panel(native.astype('uint8')*255,'Depth edges (native depth coordinates)'),panel(depth_overlay,'Depth + edges (white); invalid = gray')])
        delta=1000*(float(pairs[j][1])-float(pairs[i][0]))
        header=np.full((38,top.shape[1],3),255,'uint8')
        cv2.putText(header,f'{name} | RGB {i} | depth {j} | depth - RGB: {delta:.1f} ms | raw calibration',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1,cv2.LINE_AA)
        save(out/f'{prefix}_overview.jpg',np.vstack([header,top,bottom]))
        records.append(dict(rgb_frame=i,depth_frame=j,offset_ms=delta,depth_min_m=lo,depth_max_m=hi,
                            valid_depth_fraction=float(valid.mean()),prefix=prefix))
        if len(records)%250==0: print(name,len(records),'/',len(frames),flush=True)
    cards=[]
    for r in records:
        stem=r['prefix']
        links=' | '.join(f'<a href="{stem}_{suffix}">{label}</a>' for suffix,label in [
            ('rgb_edges.png','RGB edge mask'),('depth_edges.png','Depth edge mask'),
            ('projected_edges.png','Projected depth edge mask'),('depth_color.png','Color depth')])
        cards.append(f'<article><p>RGB {r["rgb_frame"]}, depth {r["depth_frame"]}, offset {r["offset_ms"]:.1f} ms — {links}</p><a href="{stem}_overview.jpg"><img loading="lazy" src="{stem}_overview.jpg"></a></article>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(name)+' edge inspection</title><style>body{font-family:system-ui;margin:24px;background:#eee}img{width:100%;max-width:1536px}article{margin-bottom:24px}a{color:#1454a0}</style><h1>'+html.escape(name)+'</h1><p>Depth display: near blue, far red; invalid gray. Default range is per-frame P2–P98. Masks are unmodified 256×192 outputs. Depth mask uses depth coordinates; projected mask uses RGB coordinates.</p>'+''.join(cards))
    (out/'frames.json').write_text(json.dumps(records,indent=2)+'\n')
    print(name,'complete:',out/'index.html',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes',nargs='+',default=['scene_a','scene_b','scene_c','scene_d'])
    p.add_argument('--frames',nargs='+',type=int,help='Default: all frames')
    p.add_argument('--depth-frame',type=int,help='Inspect one candidate depth with one RGB frame')
    p.add_argument('--depth-range',nargs=2,type=float,metavar=('MIN_M','MAX_M'))
    p.add_argument('--output',type=type(ROOT),default=ROOT/'deliverables/sync_edges')
    a=p.parse_args()
    if a.depth_frame is not None and (a.frames is None or len(a.frames)!=1 or len(a.scenes)!=1): p.error('--depth-frame requires one scene and one RGB frame')
    if a.depth_range and not 0<=a.depth_range[0]<a.depth_range[1]: p.error('Invalid depth range')
    for name in a.scenes: export(name,a)

if __name__=='__main__': main()
