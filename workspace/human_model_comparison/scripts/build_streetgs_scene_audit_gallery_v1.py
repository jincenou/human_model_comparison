"""Build a CPU-only, self-contained visual audit of unchanged StreetGS scenes."""
from __future__ import annotations

import argparse
import base64
from html import escape
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


H = Path(__file__).resolve().parents[1]
DEFAULT_OUT = H / "reports/streetgs_human_scene_audit_v1"
DEFAULT_IDS = ("HN0646", "HN0108", "HN0273", "HN0317")
IMAGE_KEYS = (
    ("source", "Source"),
    ("scene_native", "Scene (native)"),
    ("scene_source_camera", "Scene (source camera)"),
    ("background_source_camera", "Background (source camera)"),
    ("subset_source_camera", "Extracted (source camera)"),
)


def font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    candidates.extend(Path(sys.prefix).glob("lib/python*/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"))
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def expanded_roi(roi, size, factor):
    x0, y0, x1, y1 = map(float, roi)
    assert 0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * factor, (y1 - y0) * factor
    return (max(0, math.floor(cx - w / 2)), max(0, math.floor(cy - h / 2)),
            min(size[0], math.ceil(cx + w / 2)), min(size[1], math.ceil(cy + h / 2)))


def fitted(image, size):
    """Keep aspect ratio; never independently crop or center the depicted target."""
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)),
                           Image.Resampling.LANCZOS)
    result = Image.new("RGB", size, "#e5e7eb")
    result.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return result


def make_comparison(record, directory, factor):
    sid = record["id"]
    images = {"source": Image.open(record["source_rgb"]).convert("RGB")}
    for key, _ in IMAGE_KEYS[1:]:
        images[key] = Image.open(record["images"][key]).convert("RGB")
    size = images["source"].size
    assert all(im.size == size for im in images.values()), "Full-frame render sizes differ"
    assert size == (1600, 900), f"Unexpected original resolution: {size}"
    roi = expanded_roi(record["reference"]["roi_xyxy"], size, factor)
    # A lossless format conversion of the source JPEG; no mask or enhancement.
    images["source"].save(directory / "source.png")

    width, margin, gap = 1800, 28, 16
    top_width = (width - 2 * margin - gap) // 2
    top_height = round(top_width * size[1] / size[0])
    crop_width = (width - 2 * margin - 4 * gap) // 5
    header_y, full_y = 24, 104
    crop_label_y = full_y + top_height + 28
    crop_y = crop_label_y + 35
    footer_y = crop_y + crop_width + 20
    canvas = Image.new("RGB", (width, footer_y + 52), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, header_y), f"{sid}  |  {record['scene']}", fill="#111827", font=font(28))
    for index, (key, label) in enumerate((
        ("source", "Original full frame"),
        ("scene_native", "StreetGS full scene (native CamPose)"),
    )):
        x = margin + index * (top_width + gap)
        draw.text((x, 68), label, fill="#111827", font=font(23))
        panel = images[key].resize((top_width, top_height), Image.Resampling.LANCZOS)
        box = tuple(round(v * (top_width / size[0] if i % 2 == 0 else top_height / size[1]))
                    for i, v in enumerate(roi))
        ImageDraw.Draw(panel).rectangle(box, outline="#ff3344", width=3)
        canvas.paste(panel, (x, full_y))
    for index, (key, label) in enumerate(IMAGE_KEYS):
        x = margin + index * (crop_width + gap)
        draw.text((x, crop_label_y), label, fill="#111827", font=font(18))
        canvas.paste(fitted(images[key].crop(roi), (crop_width, crop_width)), (x, crop_y))
    draw.text((margin, footer_y),
              f"Same crop in all five columns: {roi}; {factor:g}x frozen ROI. No masking or image enhancement.",
              fill="#475569", font=font(19))
    path = directory / "comparison.jpg"
    canvas.save(path, quality=95, subsampling=0)
    return canvas, path, roi


def data_uri(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--context-factor", type=float, default=1.3)
    args = parser.parse_args()
    if args.context_factor < 1:
        parser.error("--context-factor must be >= 1")
    records = []
    for sid in args.ids:
        path = args.out / sid / "audit.json"
        record = json.loads(path.read_text())
        assert record["id"] == sid and record["status"] == "passed"
        assert record["trained"] is False and record["gaussian_parameters_modified"] is False
        assert record["reference_mask_used_in_render"] is False
        records.append(record)
    cards, pictures = [], []
    for record in records:
        sid = record["id"]
        picture, path, roi = make_comparison(record, args.out / sid, args.context_factor)
        pictures.append(picture)
        uri = data_uri(path)
        links = ' · '.join(
            f'<a href="{escape(sid)}/{escape("source.png" if key == "source" else Path(record["images"][key]).name)}" download>{escape(label)}</a>'
            for key, label in IMAGE_KEYS
        )
        flags = record["reference"].get("reference_quality_flags", [])
        blur = '原图已有低分辨率或模糊标记，展示保留原状。' if any('blur' in flag or 'resolution' in flag for flag in flags) else '按原始图像展示，不进行去模糊或细节补全。'
        cards.append(f'''<section id="{escape(sid)}"><h2>{escape(sid)} · {escape(str(record['scene']))}</h2>
<p>{blur} 所有局部图使用相同像素范围 {roi}。</p>
<a href="{uri}" download="{escape(sid)}_comparison.jpg"><img src="{uri}" alt="{escape(sid)} 原照片与 StreetGS 全场景、背景和提取对象的同位置对比"></a>
<p><a href="{uri}" download="{escape(sid)}_comparison.jpg">下载此对象拼图</a> · <a href="{escape(sid)}/audit.json">原始核查记录</a></p>
<p class="raw">1600 × 900 PNG：{links}</p></section>''')

    thumb_w, gap = 900, 20
    thumb_h = max(round(image.height * thumb_w / image.width) for image in pictures)
    cols = min(2, len(pictures)); rows = math.ceil(len(pictures) / cols)
    overview = Image.new("RGB", (cols * thumb_w + (cols - 1) * gap,
                                  rows * thumb_h + (rows - 1) * gap), "#cbd5e1")
    for index, picture in enumerate(pictures):
        overview.paste(fitted(picture, (thumb_w, thumb_h)),
                       ((index % cols) * (thumb_w + gap), (index // cols) * (thumb_h + gap)))
    overview.save(args.out / "overview.jpg", quality=95, subsampling=0)
    overview_uri = data_uri(args.out / "overview.jpg")
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreetGS 人体原场景核查</title><style>
body{{font:16px/1.65 system-ui,sans-serif;background:#eef2f7;color:#172033;margin:0 auto;padding:24px;max-width:1800px}}h1{{font-size:28px}}h2{{font-size:22px}}p{{max-width:1400px}}a{{color:#1254a0}}section{{background:white;padding:22px;margin:26px 0;border-radius:12px}}img{{display:block;width:100%;height:auto;border:1px solid #d8dee8}}.raw{{font-size:14px}}nav a{{margin-right:22px}}aside{{padding:14px 18px;background:#fff;border-left:4px solid #cc3344}}@media(max-width:700px){{body{{padding:12px}}section{{padding:12px}}}}
</style><h1>StreetGS：原照片、整场景与提取人体对比</h1>
<p>四个对象保持原 StreetGS 检查点，不重新训练或修改高斯。上排给出完整原照片与 StreetGS 原生整场景，红框标出下排取景区域。下排五列在同一原图像素坐标裁切，不用 GT mask 擦除背景，也不单独移动或缩放某列中的人物。</p>
<aside><strong>两种场景视角的区别：</strong>Scene (native) 使用模型原生渲染，包括学习得到的 CamPose 相机修正；Scene (source camera) 保留同一场景，但关闭这项相机修正以匹配冻结源相机。Background 和 Extracted 均使用源相机：前者只显示背景高斯，后者只显示此前从背景中提取的目标高斯；这两列空白区域采用白色背景。位置差异与残影均原样保留。</aside>
<p>局部区域为冻结 ROI 的 {args.context_factor:g} 倍，用于显示周边关系。原图自身的模糊会保留，不做图像增强。点击拼图可下载 JPG。此 HTML 已内嵌全部对比 JPG，可单文件离线观看；下方原分辨率 PNG 链接需与对应对象文件夹一同保留。</p>
<nav>{''.join(f'<a href="#{escape(sid)}">{escape(sid)}</a>' for sid in args.ids)}</nav>
<p><a href="{overview_uri}" download="streetgs_scene_audit_overview.jpg">下载四例总览图</a></p>
{''.join(cards)}</html>'''
    (args.out / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({"status": "ok", "objects": list(args.ids), "html": str(args.out / "index.html"),
                      "overview": str(args.out / "overview.jpg")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
