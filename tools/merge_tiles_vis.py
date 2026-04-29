"""
6×8 (또는 임의 그리드) 타일 라벨을 원본 이미지 좌표로 병합해 시각화.

사용:
    python tools/merge_tiles_vis.py \
        --orig  data/역사이미지/slope/DJI_20260306113838_0004.JPG \
        --labels output/autolabel/tile6x8/labels \
        --output output/autolabel/tile6x8/merged_vis.jpg \
        --cols 6 --rows 8
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

CLASS_NAMES  = ["catenary_pole", "bracket"]
CLASS_COLORS = [(0, 200, 255), (255, 130, 0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig",   required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cols",   type=int, default=6)
    parser.add_argument("--rows",   type=int, default=8)
    parser.add_argument("--alpha",  type=float, default=0.3)
    args = parser.parse_args()

    buf = np.fromfile(args.orig, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    tw, th = W // args.cols, H // args.rows

    label_dir = Path(args.labels)
    counts = [0] * len(CLASS_NAMES)

    for r in range(args.rows):
        for c in range(args.cols):
            label_file = label_dir / f"tile_r{r+1:02d}_c{c+1:02d}.txt"
            if not label_file.exists():
                continue
            x0 = c * tw
            y0 = r * th
            tile_w = tw if c < args.cols - 1 else W - x0
            tile_h = th if r < args.rows - 1 else H - y0

            text = label_file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for line in text.splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                pts = np.array(
                    [[coords[i] * tile_w + x0, coords[i + 1] * tile_h + y0]
                     for i in range(0, len(coords), 2)],
                    dtype=np.int32,
                )
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, args.alpha, img, 1 - args.alpha, 0, img)
                cv2.polylines(img, [pts], True, color, 3)
                if cls_id < len(counts):
                    counts[cls_id] += 1

    # 범례
    for i, name in enumerate(CLASS_NAMES):
        y = 30 + i * 50
        cv2.rectangle(img, (15, y - 20), (55, y + 10), CLASS_COLORS[i], -1)
        cv2.putText(img, f"{name}: {counts[i]}",
                    (65, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, CLASS_COLORS[i], 3)

    total = sum(counts)
    print(f"총 {total}개  " + ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(len(counts))))

    # 4096px 이하로 미리보기 저장
    scale = min(1.0, 4096 / max(H, W))
    vis = cv2.resize(img, (int(W * scale), int(H * scale)))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(out_path.suffix, vis)[1].tofile(str(out_path))
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
