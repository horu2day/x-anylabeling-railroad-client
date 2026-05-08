"""
이미지를 cols×rows 타일로 분할하고 번호를 좌상단에 표시.
사용법:
    python tools/show_tiles.py --input <이미지> --cols 8 --rows 6 --overlap 0.10
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   required=True)
    ap.add_argument("--output",  default=None)
    ap.add_argument("--cols",    type=int,   default=8)
    ap.add_argument("--rows",    type=int,   default=6)
    ap.add_argument("--overlap", type=float, default=0.10)
    args = ap.parse_args()

    img_path  = Path(args.input)
    buf       = np.fromfile(str(img_path), dtype=np.uint8)
    img       = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    H, W      = img.shape[:2]

    base_w = W / args.cols
    base_h = H / args.rows
    pad_x  = int(base_w * args.overlap)
    pad_y  = int(base_h * args.overlap)

    vis = img.copy()

    for r in range(args.rows):
        for c in range(args.cols):
            idx = r * args.cols + c + 1
            x0 = max(0, int(c * base_w) - pad_x)
            x1 = min(W, int((c + 1) * base_w) + pad_x)
            y0 = max(0, int(r * base_h) - pad_y)
            y1 = min(H, int((r + 1) * base_h) + pad_y)

            # 타일 경계선 (기준선 = 중복 없는 경계)
            bx0 = int(c * base_w)
            by0 = int(r * base_h)
            bx1 = min(W, int((c + 1) * base_w))
            by1 = min(H, int((r + 1) * base_h))
            cv2.rectangle(vis, (bx0, by0), (bx1, by1), (0, 200, 255), 3)

            # 번호 좌상단
            font_scale = max(0.8, min(W, H) / 3000)
            thickness  = max(2, int(font_scale * 3))
            tx, ty = bx0 + 10, by0 + int(base_h * 0.12)

            # 배경 박스
            label = str(idx)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, thickness)
            cv2.rectangle(vis, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(vis, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, (0, 200, 255), thickness, cv2.LINE_AA)

    # 출력 크기 제한
    h, w = vis.shape[:2]
    if max(h, w) > 4096:
        s = 4096 / max(h, w)
        vis = cv2.resize(vis, (int(w * s), int(h * s)))

    out = (Path(args.output) if args.output
           else img_path.parent / (img_path.stem + "_tiles.jpg"))
    cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(out))
    print(f"저장: {out}  ({args.cols}×{args.rows}={args.cols*args.rows}타일)")

if __name__ == "__main__":
    main()
