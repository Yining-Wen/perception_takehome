"""Build a standalone scene by copying original depth exposures into new paired indices."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/scene_a')
    parser.add_argument('--sync',type=Path,default=ROOT/'variant/scene_a/sync.json')
    parser.add_argument('--pairing',type=Path,default=ROOT/'deliverables/depth_pair_search/scene_a/best_pairing.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/repaired/scene_a')
    parser.add_argument('--calib',type=Path,help='Frozen working calibration; defaults to source calib.json')
    args=parser.parse_args()
    source=args.source.resolve();out=args.output.resolve()
    if out.exists():raise SystemExit(f'Refusing to overwrite existing destination: {out}')
    if out==source or source in out.parents:raise SystemExit('Destination must be outside the source scene')
    mapping=json.loads(args.pairing.read_text());sync=json.loads(args.sync.read_text())
    calib_path=(args.calib or source/'calib.json').resolve()
    calib=json.loads(calib_path.read_text());n=len(calib['frames'])
    if calib['frames']!=json.loads((source/'calib.json').read_text())['frames']:
        raise ValueError('Working calibration must preserve original frame timestamps and poses')
    if len(mapping)!=n or len(sync['pairs'])!=n:raise ValueError('Frame, pairing and timestamp counts differ')
    for i,j in enumerate(mapping):
        if type(j) is not int or not 0<=j<n:raise ValueError(f'Frame {i} has invalid/null depth index; resolve before materialization')
        for path in (source/'rgb'/f'{i:06d}.png',source/'depth'/f'{j:06d}.png'):
            if not path.is_file():raise FileNotFoundError(path)
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_name(out.name+'.building')
    if temp.exists():raise SystemExit(f'Staging directory exists; inspect first: {temp}')
    (temp/'rgb').mkdir(parents=True);(temp/'depth').mkdir()
    for i,j in enumerate(mapping):
        shutil.copy2(source/'rgb'/f'{i:06d}.png',temp/'rgb'/f'{i:06d}.png')
        shutil.copy2(source/'depth'/f'{j:06d}.png',temp/'depth'/f'{i:06d}.png')
    shutil.copy2(calib_path,temp/'calib.json')
    for name in ('boxes.json',):
        if (source/name).exists():shutil.copy2(source/name,temp/name)
    updated=dict(sync)
    updated['pairs']=[[sync['pairs'][i][0],sync['pairs'][j][1]] for i,j in enumerate(mapping)]
    (temp/'sync.json').write_text(json.dumps(updated,indent=2)+'\n')
    (temp/'original_depth_indices.json').write_text(json.dumps(mapping,indent=2)+'\n')
    changed=sum(i!=j for i,j in enumerate(mapping))
    manifest=dict(source=str(source),source_sync=str(args.sync.resolve()),pairing=str(args.pairing.resolve()),
                  frames=n,replaced_depth_files=changed,unique_depth_exposures=len({sync['pairs'][j][1] for j in mapping}),
                  source_sync_sha256=digest(args.sync),pairing_sha256=digest(args.pairing),
                  calibration_source=str(calib_path),calibration_sha256=digest(calib_path),
                  policy='Use every minimum-score candidate, including candidates that failed conservative screening.',
                  loader='perception.scene.Scene',
                  warning='Images already reordered. Do not apply the source mapping again. Nonzero exposure offsets and duplicate exposures are preserved, not corrected clock values.')
    # Check every output against its original source, not only sample frames.
    for i,j in enumerate(mapping):
        assert digest(temp/'rgb'/f'{i:06d}.png')==digest(source/'rgb'/f'{i:06d}.png')
        assert digest(temp/'depth'/f'{i:06d}.png')==digest(source/'depth'/f'{j:06d}.png')
        assert updated['pairs'][i]==[sync['pairs'][i][0],sync['pairs'][j][1]]
    assert digest(temp/'calib.json')==digest(calib_path)
    manifest['verification']='All RGB/depth copies and selected calibration SHA-256 verified; frame timestamps and poses unchanged.'
    (temp/'materialization.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (temp/'README.md').write_text('# Repaired '+source.name+'''\n\nRGB i and depth i now contain the selected pair. Read directly with the ordinary Scene loader.\n\n```python\nfrom perception.scene import Scene\ns = Scene("'''+str(out)+'''")\nrgb, depth, pose = s.rgb(i), s.depth_m(i), s.pose(i)\n```\n\nsync.json contains the RGB and selected depth exposure timestamps. calib.json is copied unchanged from the original dataset. original_depth_indices.json is provenance only: do NOT apply it again. All minimum-score candidates are used, including uncertain ones; this is not the conservative reviewed mapping. The original dataset is untouched.\n''')
    readme=temp/'README.md'
    readme.write_text(readme.read_text().replace('calib.json is copied unchanged from the original dataset.',f'calib.json is copied from {calib_path}. Frame timestamps and poses are unchanged.'))
    temp.rename(out)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
