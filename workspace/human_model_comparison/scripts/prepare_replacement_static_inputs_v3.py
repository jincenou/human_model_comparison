"""Frozen real-exposure inputs for native asset production, not metric eligibility."""
import hashlib
import json
from pathlib import Path
from PIL import Image

H = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    selection = H/'inputs/benchmark_human100_v3/selection.json'
    rows = [r for r in json.loads(selection.read_text())['objects'] if r['id']=='HN0646']
    output = H/'inputs/human_static_replacement_v3'
    output.mkdir(exist_ok=False)
    records = []
    for row in rows:
        sid = row['id']
        candidate = H/'configs/human_source_common_v1/replacement_v3'/sid
        camera = json.loads((candidate/'camera.json').read_text())
        assert camera['selection_sha256'] == sha(selection)
        assert camera['timestamp_us'] == row['t_ref_us']
        rgb = Image.open(candidate/'reference_rgb.png').convert('RGB')
        mask = Image.open(candidate/'reference_mask.png').convert('L')
        assert rgb.size == mask.size == (512,512) and mask.getbbox()
        folder = output/sid; folder.mkdir()
        rgba = rgb.copy(); rgba.putalpha(mask); rgba.save(folder/'input.png')
        rgb.save(folder/'source.jpeg',quality=100,subsampling=0)
        mask.save(folder/'mask.png')
        record = dict(sample_id=sid, identity_key=row['identity_key'], scene=row['scene'],
            timestamp_us=row['t_ref_us'], selection_path=str(selection), selection_sha256=sha(selection),
            raw_source_path=row['reference_frame']['raw_source_path'],
            raw_source_sha256=row['reference_frame']['raw_source_sha256'],
            exact_mask_path=row['reference_mask_candidate']['path'], exact_mask_sha256=row['reference_mask_candidate']['sha256'],
            camera_candidate=str(candidate/'camera.json'), camera_candidate_sha256=sha(candidate/'camera.json'),
            image_rgba=str(folder/'input.png'),image_rgba_sha256=sha(folder/'input.png'),
            image=str(folder/'source.jpeg'),source_image_sha256=sha(folder/'source.jpeg'),
            mask=str(folder/'mask.png'),source_mask_sha256=sha(folder/'mask.png'),
            real_input_count=1,real_duration_s=0,interpolation_frames=0,input_generated_images=0,
            generated_input_count=0,data_nature='real_photo',stage='native_asset_production_only',
            formal_reference_eligible=False,common_camera_valid=False,scores_produced=False,
            pose_policy='native source-conditioned prediction; no reposing or pose correction',
            preprocessing='frozen candidate reference ROI, RGB bicubic, alpha nearest, 512; AH JPEG q100 4:4:4')
        ahroot = H/'inputs/ah/jobs'/f'human_native_v1_{sid}'
        ahfolder = ahroot/sid; ahfolder.mkdir(parents=True,exist_ok=False)
        # Hardlinks preserve exact bytes without redundant storage.
        (ahfolder/'frame.jpeg').hardlink_to(folder/'source.jpeg')
        (ahfolder/'mask.png').hardlink_to(folder/'mask.png')
        ahrecord = dict(record,image=str(ahfolder/'frame.jpeg'),mask=str(ahfolder/'mask.png'))
        (ahroot/'source_manifest.json').write_text(json.dumps(ahrecord,indent=2)+'\n')
        records.append(record)
    (output/'manifest.json').write_text(json.dumps(dict(selection=str(selection),selection_sha256=sha(selection),
        population=1,scope='native production; reference and registration qualification remain separate',objects=records),indent=2)+'\n')
    print(json.dumps(dict(objects=len(records),manifest=str(output/'manifest.json'))))

if __name__=='__main__': main()
