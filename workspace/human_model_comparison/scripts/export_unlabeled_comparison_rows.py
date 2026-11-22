"""Remove the model-name header from locally generated comparison rows."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    args = parser.parse_args()
    for sid in args.ids:
        if not (sid.startswith('HN') and sid[2:].isdigit()):
            parser.error('IDs must use HN followed by digits')
    out = args.gallery / 'no_text'
    out.mkdir(exist_ok=True)
    records = []
    for sid in args.ids:
        for folder, suffix, columns in [('rows', '', 7), ('models_only', '_models_only', 6)]:
            source = args.gallery / folder / (sid + '.png')
            destination = out / (sid + suffix + '.png')
            with Image.open(source) as image:
                if image.size != (columns * 512, 576):
                    raise ValueError(f'Unexpected labeled canvas: {source}: {image.size}')
                image.crop((0, 64, image.width, 576)).save(destination)
            records.append(dict(id=sid, source=str(source), output=str(destination),
                                columns=columns, removed_header_pixels=64,
                                sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    (out / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    print(f'Exported {len(records)} unlabeled PNG rows to {out}')


if __name__ == '__main__':
    main()
