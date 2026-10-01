"""Export retained spatial calibration plus explicit, conservative depth associations."""
import copy
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    original=json.loads((ROOT/'data/scene_a/calib.json').read_text())
    sync_original=json.loads((ROOT/'variant/scene_a/sync.json').read_text())
    sync=sync_original['pairs']
    search=ROOT/'deliverables/depth_pair_search/scene_a'
    best=json.loads((search/'best_pairing.json').read_text())
    reviewed=json.loads((search/'reviewed_pairing.json').read_text())
    rows={int(r['rgb_frame']):r for r in csv.DictReader((search/'frames.csv').open())}
    calib=copy.deepcopy(original)
    for i,frame in enumerate(calib['frames']):
        j=reviewed[i]
        frame.update(depth_index=j,rgb_timestamp=float(sync[i][0]),
            depth_timestamp=float(sync[j][1]) if j is not None else None,
            depth_offset_ms=1000*(float(sync[j][1])-float(sync[i][0])) if j is not None else None,
            pairing_status='original_timestamp_matched' if i not in rows else 'image_search_screened' if j is not None else 'excluded_pending_review',
            best_depth_candidate=best[i],pairing_review_reason=rows[i]['reason'] if i in rows else '')
        if i==764:frame['pose_warning']='Unresolved localization jump into this frame; inspect subsequent map alignment. Pose unchanged.'
    calib['processing']={'schema':'rgbd_pairing_extension_v1','scene':'scene_a',
        'policy':'Conservative image-search mapping; unscreened abnormal pairs have depth_index=null.',
        'window_ms':1000,'spatial_parameters_changed':False,'poses_changed':False,
        'scale_estimated':False,'extrinsic_translation_verified':False,
        'original_timestamp_matched':sum(i not in rows for i in range(len(reviewed))),
        'reassociated':sum(j is not None and j!=i for i,j in enumerate(reviewed)),
        'excluded':sum(j is None for j in reviewed),
        'loader':'perception.corrected_scene.CorrectedScene',
        'warning':'Base Scene ignores depth_index. Use CorrectedScene. Image minima are not synchronization ground truth; nonzero offsets are not motion-compensated.'}
    out=ROOT/'deliverables/corrected/scene_a';out.mkdir(parents=True,exist_ok=True)
    sync_corrected=copy.deepcopy(sync_original)
    sync_corrected['pairs']=[[sync[i][0],sync[j][1] if j is not None else None] for i,j in enumerate(reviewed)]
    sync_corrected['processing']={
        'schema':'rgbd_sync_nullable_v1',
        'source':'variant/scene_a/sync.json',
        'pairing_source':'pairing.json',
        'frame_index_preserved':True,
        'null_depth_means':'Excluded pending review; not necessarily a missing sensor exposure.',
        'warning':'Do not pass to original fetch.py: depth timestamps may be null, and existing image indices still follow original extraction.'}
    for name,obj in [('sync.json',sync_corrected),('calib.json',calib),('pairing.json',reviewed),('best_pairing_for_review.json',best)]:
        (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    assert calib['color']==original['color'] and calib['depth']==original['depth']
    assert all(f['T_world_camera']==g['T_world_camera'] and f['t']==g['t'] for f,g in zip(calib['frames'],original['frames']))
    print(json.dumps(calib['processing'],indent=2))

if __name__=='__main__':main()
