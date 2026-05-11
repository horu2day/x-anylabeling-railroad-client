"""
Railway Detection Web UI
사용법: python tools/web_ui.py
브라우저: http://localhost:7000
"""
import asyncio
import base64
import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    import uvicorn
except ImportError:
    print("FastAPI/uvicorn 설치 필요: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI()
jobs: dict = {}  # job_id -> {queue, status, result_path}

ROOT = Path(__file__).parent.parent  # 프로젝트 루트


# ── 미리보기 이미지 생성 ───────────────────────────────────────────────────────
def make_preview(image_path: str, cols: int, rows: int, selected_rows: list) -> str:
    buf = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지 로드 실패: {image_path}")

    H, W = img.shape[:2]
    tw = 1200
    scale = tw / W
    vis = cv2.resize(img, (tw, int(H * scale)))
    H_s, W_s = vis.shape[:2]

    tile_w = W_s / cols
    tile_h = H_s / rows

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c + 1
            x0, y0 = int(c * tile_w), int(r * tile_h)
            x1, y1 = min(W_s, int((c + 1) * tile_w)), min(H_s, int((r + 1) * tile_h))
            row_num = r + 1

            if row_num in selected_rows:
                overlay = vis.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 0), -1)
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            else:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (180, 180, 180), 1)

            label = str(idx)
            fs = max(0.35, tile_w / 120)
            (tw2, th2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cv2.putText(vis, label, (cx - tw2 // 2, cy + th2 // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

        # Row 번호
        y0 = int(r * tile_h)
        row_num = r + 1
        color = (0, 255, 0) if row_num in selected_rows else (160, 160, 160)
        cv2.putText(vis, f"Row{row_num}", (4, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    _, enc = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(enc).decode("utf-8")


def tiles_from_rows(selected_rows: list, cols: int) -> str:
    if not selected_rows:
        return "all"
    ids = []
    for r in sorted(selected_rows):
        ids.extend(range((r - 1) * cols + 1, r * cols + 1))
    if ids and ids[-1] - ids[0] + 1 == len(ids):
        return f"{ids[0]}-{ids[-1]}"
    return ",".join(str(t) for t in ids)


# ── API ───────────────────────────────────────────────────────────────────────
@app.post("/api/preview")
async def api_preview(data: dict):
    try:
        b64 = make_preview(
            data["image_path"],
            data.get("cols", 8),
            data.get("rows", 6),
            data.get("selected_rows", []),
        )
        return {"ok": True, "image": b64}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/detect")
async def api_detect(data: dict):
    job_id = str(uuid.uuid4())[:8]
    q: queue.Queue = queue.Queue()
    jobs[job_id] = {"queue": q, "status": "running", "result_path": None, "proc": None}

    def run():
        tiles_str = tiles_from_rows(data.get("selected_rows", []), data.get("cols", 8))
        cmd = [
            sys.executable, "-u", str(ROOT / "tools" / "detect_all_objects.py"),
            "--input",      data["image_path"],
            "--categories", data.get("categories", "configs/railway_zone.json"),
            "--tiles",      tiles_str,
            "--cols",       str(data.get("cols", 8)),
            "--rows",       str(data.get("rows", 6)),
            "--overlap",    str(data.get("overlap", 0.20)),
            "--conf",       str(data.get("conf", 0.20)),
            "--workers",    str(data.get("workers", 8)),
        ]
        if data.get("debug"):
            cmd.append("--debug")

        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=env, cwd=str(ROOT), creationflags=flags,
            )
            jobs[job_id]["proc"] = proc
            for raw in proc.stdout:
                for line in raw.replace("\r", "\n").split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    q.put(("log", line))
                    if line.startswith("저장:"):
                        jobs[job_id]["result_path"] = line.replace("저장:", "").strip()
            proc.wait()
            if jobs[job_id]["status"] == "stopped":
                q.put(("error", "사용자가 중지했습니다"))
            elif proc.returncode == 0:
                q.put(("done", jobs[job_id]["result_path"] or "완료"))
                jobs[job_id]["status"] = "done"
            else:
                q.put(("error", f"종료코드 {proc.returncode}"))
                jobs[job_id]["status"] = "error"
        except Exception as e:
            q.put(("error", str(e)))
            jobs[job_id]["status"] = "error"

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/stop/{job_id}")
async def api_stop(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"ok": False, "error": "not found"}
    job["status"] = "stopped"
    proc = job.get("proc")
    if proc and proc.poll() is None:
        if os.name == "nt":
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.terminate()
    return {"ok": True}


@app.get("/api/progress/{job_id}")
async def api_progress(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "not found"}, status_code=404)

    async def event_gen():
        q = jobs[job_id]["queue"]
        while True:
            try:
                type_, msg = q.get_nowait()
                yield f"data: {json.dumps({'type': type_, 'msg': msg})}\n\n"
                if type_ in ("done", "error"):
                    break
            except queue.Empty:
                await asyncio.sleep(0.25)
                yield ": ping\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/result/{job_id}")
async def api_result(job_id: str):
    job = jobs.get(job_id, {})
    path = job.get("result_path")
    if not path or not Path(path).exists():
        return {"ok": False, "error": "결과 없음"}
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "이미지 로드 실패"}
    h, w = img.shape[:2]
    if max(h, w) > 3000:
        s = 3000 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"ok": True, "image": base64.b64encode(enc).decode(), "path": path}


@app.post("/api/open-folder")
async def api_open_folder(data: dict):
    path = Path(data.get("path", "output/detect"))
    folder = path.parent if path.is_file() else path
    if os.name == "nt":
        os.startfile(str(folder.resolve()))
    return {"ok": True}


# ── HTML ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Railway Detection UI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{background:#16213e;padding:10px 20px;border-bottom:2px solid #0f3460;flex-shrink:0}
header h1{font-size:1.1rem;color:#e94560;letter-spacing:1px}
.tabs{display:flex;background:#16213e;border-bottom:1px solid #0f3460;flex-shrink:0}
.tab-btn{padding:9px 26px;background:none;border:none;color:#888;cursor:pointer;font-size:0.92rem;border-bottom:3px solid transparent;transition:all .2s}
.tab-btn.active{color:#e94560;border-bottom-color:#e94560}
.tab-content{display:none;flex:1;overflow:hidden;min-height:0}
.tab-content.active{display:flex}

/* 입력 탭 */
#tab-input{flex-direction:row}
.controls{width:320px;min-width:260px;background:#16213e;padding:14px;overflow-y:auto;border-right:1px solid #0f3460;display:flex;flex-direction:column;gap:12px;flex-shrink:0}
.preview-area{flex:1;background:#0d0d1f;display:flex;align-items:center;justify-content:center;overflow:hidden}
.preview-area img{max-width:100%;max-height:100%;object-fit:contain}
.preview-area .hint{color:#444;font-size:.9rem}
.fg{display:flex;flex-direction:column;gap:4px}
.fg>label{font-size:.75rem;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.path-row{display:flex;gap:6px}
.path-row input{flex:1;min-width:0}
input[type=text],input[type=number]{background:#0f3460;border:1px solid #2a4a7a;color:#e0e0e0;padding:6px 9px;border-radius:4px;font-size:.88rem;width:100%}
input[type=text]:focus,input[type=number]:focus{outline:none;border-color:#e94560}
.row-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.row-lbl{display:flex;align-items:center;gap:6px;padding:7px 9px;background:#0f3460;border:1px solid #2a4a7a;border-radius:4px;cursor:pointer;transition:all .15s;font-size:.85rem}
.row-lbl:hover{border-color:#0c6}
.row-lbl.on{border-color:#0f6;background:#0d3320;color:#0f6}
.row-lbl input{width:14px;height:14px;cursor:pointer;accent-color:#0f6}
.params-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.pi{display:flex;flex-direction:column;gap:3px}
.pi label{font-size:.72rem;color:#888}
.debug-row{display:flex;align-items:center;gap:8px;font-size:.84rem;color:#aaa;cursor:pointer}
.debug-row input{width:15px;height:15px;accent-color:#e94560;cursor:pointer}
.btn-load{background:#0f3460;color:#e0e0e0;border:1px solid #2a4a7a;padding:6px 13px;border-radius:4px;cursor:pointer;font-size:.84rem;white-space:nowrap;transition:background .2s}
.btn-load:hover{background:#1a5299}
.btn-run{background:#e94560;color:#fff;border:none;padding:11px;border-radius:6px;cursor:pointer;font-size:.98rem;font-weight:700;width:100%;margin-top:2px;transition:background .2s}
.btn-run:hover{background:#c73652}
.btn-run:disabled{background:#555;cursor:not-allowed}
.btn-stop{background:#555;color:#fff;border:none;padding:11px;border-radius:6px;cursor:pointer;font-size:.98rem;font-weight:700;width:100%;margin-top:2px;display:none}
.btn-stop.visible{display:block}
.btn-stop:hover{background:#c73652}

/* 결과 탭 */
#tab-result{flex-direction:column}
.res-toolbar{background:#16213e;padding:9px 14px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #0f3460;flex-shrink:0}
.btn-act{background:#0f3460;color:#e0e0e0;border:1px solid #2a4a7a;padding:6px 13px;border-radius:4px;cursor:pointer;font-size:.84rem;transition:background .2s}
.btn-act:hover{background:#1a5299}
#result-path{font-size:.78rem;color:#888;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}
.viewer{flex:1;overflow:hidden;cursor:grab;position:relative;background:#0a0a1a;min-height:0}
.viewer.drag{cursor:grabbing}
.viewer img{position:absolute;transform-origin:0 0;user-select:none;pointer-events:none}
.viewer-hint{display:flex;align-items:center;justify-content:center;height:100%;color:#333;font-size:.95rem}

/* 진행 */
.prog-section{background:#16213e;padding:8px 16px;border-top:1px solid #0f3460;flex-shrink:0}
.prog-track{background:#0a0a1a;border-radius:4px;height:7px;overflow:hidden;margin-bottom:5px}
.prog-fill{height:100%;background:linear-gradient(90deg,#e94560,#0f6);width:0%;transition:width .4s;border-radius:4px}
.prog-log{font-size:.76rem;color:#777;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
</style>
</head>
<body>
<header><h1>&#x1F6E4; Railway Detection UI</h1></header>

<div class="tabs">
  <button class="tab-btn active" id="tbtn-input"  onclick="switchTab('input')">입력</button>
  <button class="tab-btn"        id="tbtn-result" onclick="switchTab('result')">결과</button>
</div>

<div id="tab-input" class="tab-content active">
  <div class="controls">

    <div class="fg">
      <label>이미지 경로</label>
      <div class="path-row">
        <input type="text" id="image-path" placeholder="경로 입력 후 Enter 또는 버튼">
        <button class="btn-load" onclick="loadPreview()">미리보기</button>
      </div>
    </div>

    <div class="fg">
      <label>Row 선택 (철도 구역)</label>
      <div class="row-grid" id="row-grid"></div>
    </div>

    <div class="fg">
      <label>Categories JSON</label>
      <input type="text" id="categories" value="configs/railway_zone.json">
    </div>

    <div class="fg">
      <label>파라미터</label>
      <div class="params-grid">
        <div class="pi"><label>Cols</label><input type="number" id="cols" value="8" min="1" onchange="loadPreview()"></div>
        <div class="pi"><label>Rows</label><input type="number" id="rows" value="6" min="1" onchange="initRows();loadPreview()"></div>
        <div class="pi"><label>Overlap</label><input type="number" id="overlap" value="0.20" step="0.05" min="0" max="0.5"></div>
        <div class="pi"><label>Conf</label><input type="number" id="conf" value="0.20" step="0.05" min="0.05" max="1"></div>
        <div class="pi"><label>Workers</label><input type="number" id="workers" value="8" min="1" max="32"></div>
      </div>
    </div>

    <label class="debug-row">
      <input type="checkbox" id="debug"> Debug 이미지 저장
    </label>

    <button class="btn-run"  id="btn-run"  onclick="startDetect()">&#x25B6; 검출 시작</button>
    <button class="btn-stop" id="btn-stop" onclick="stopDetect()">&#x25A0; 중지</button>

  </div>

  <div class="preview-area" id="preview-area">
    <span class="hint">이미지 경로 입력 후 미리보기 클릭</span>
  </div>
</div>

<div id="tab-result" class="tab-content">
  <div class="res-toolbar">
    <button class="btn-act" onclick="openFolder()">&#x1F4C1; 출력 폴더 열기</button>
    <span id="result-path">결과 없음</span>
    <button class="btn-act" onclick="resetZoom()" style="margin-left:auto">화면 맞춤</button>
    <span style="font-size:.76rem;color:#555">휠: 확대/축소 &nbsp;|&nbsp; 드래그: 이동</span>
  </div>
  <div class="viewer" id="viewer">
    <div class="viewer-hint" id="viewer-hint">검출 완료 후 결과가 표시됩니다</div>
    <img id="result-img" style="display:none">
  </div>
</div>

<div class="prog-section">
  <div class="prog-track"><div class="prog-fill" id="prog-fill"></div></div>
  <div class="prog-log" id="prog-log">대기 중</div>
</div>

<script>
let resultPath = null;
let sc = 1, tx = 0, ty = 0, dragging = false, sx, sy;

// ── 탭 전환
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('tbtn-' + name).classList.add('active');
}

// ── Row 체크박스
function initRows() {
  const numRows = parseInt(document.getElementById('rows').value) || 6;
  const grid = document.getElementById('row-grid');
  grid.innerHTML = '';
  for (let r = 1; r <= numRows; r++) {
    const lbl = document.createElement('label');
    lbl.className = 'row-lbl';
    lbl.innerHTML = `<input type="checkbox" value="${r}" onchange="onRowChange(this)"><span>Row ${r}</span>`;
    grid.appendChild(lbl);
  }
}

function onRowChange(cb) {
  cb.parentElement.classList.toggle('on', cb.checked);
  loadPreview();
}

function getSelectedRows() {
  return Array.from(document.querySelectorAll('#row-grid input:checked')).map(c => parseInt(c.value));
}

// ── 미리보기
async function loadPreview() {
  const imgPath = document.getElementById('image-path').value.trim();
  if (!imgPath) return;
  document.getElementById('preview-area').innerHTML = '<span class="hint">로딩 중...</span>';
  try {
    const res = await post('/api/preview', {
      image_path: imgPath,
      cols: +document.getElementById('cols').value,
      rows: +document.getElementById('rows').value,
      selected_rows: getSelectedRows()
    });
    if (res.ok) {
      document.getElementById('preview-area').innerHTML =
        `<img src="data:image/jpeg;base64,${res.image}" alt="미리보기">`;
    } else {
      document.getElementById('preview-area').innerHTML =
        `<span class="hint" style="color:#e94560">오류: ${res.error}</span>`;
    }
  } catch(e) {
    document.getElementById('preview-area').innerHTML =
      `<span class="hint" style="color:#e94560">${e}</span>`;
  }
}

// ── 검출 시작
async function startDetect() {
  const rows = getSelectedRows();
  if (!rows.length) { alert('Row를 하나 이상 선택하세요'); return; }
  const imgPath = document.getElementById('image-path').value.trim();
  if (!imgPath) { alert('이미지 경로를 입력하세요'); return; }

  document.getElementById('btn-run').disabled = true;
  document.getElementById('btn-stop').classList.add('visible');
  document.getElementById('prog-fill').style.width = '0%';
  document.getElementById('prog-log').textContent = '시작 중...';

  const res = await post('/api/detect', {
    image_path: imgPath,
    selected_rows: rows,
    cols:     +document.getElementById('cols').value,
    rows:     +document.getElementById('rows').value,
    overlap:  +document.getElementById('overlap').value,
    conf:     +document.getElementById('conf').value,
    workers:  +document.getElementById('workers').value,
    categories: document.getElementById('categories').value.trim(),
    debug: document.getElementById('debug').checked
  });
  listenProgress(res.job_id);
}

// ── SSE 진행
function listenProgress(jobId) {
  const es = new EventSource(`/api/progress/${jobId}`);
  es.onmessage = async e => {
    const {type, msg} = JSON.parse(e.data);
    document.getElementById('prog-log').textContent = msg || '';
    if (type === 'log') {
      const m = msg.match(/타일\s+(\d+)\/(\d+)/);
      if (m) {
        document.getElementById('prog-fill').style.width =
          Math.round(+m[1] / +m[2] * 100) + '%';
      }
    } else if (type === 'done') {
      es.close();
      document.getElementById('prog-fill').style.width = '100%';
      document.getElementById('prog-log').textContent = '완료! 결과 탭으로 이동합니다.';
      setRunning(false);
      await showResult(jobId);
    } else if (type === 'error') {
      es.close();
      document.getElementById('prog-log').textContent = msg;
      setRunning(false);
    }
  };
  es.onerror = () => { es.close(); setRunning(false); };
}

// ── 결과 표시
async function showResult(jobId) {
  const res = await fetch(`/api/result/${jobId}`).then(r => r.json());
  if (!res.ok) return;
  resultPath = res.path;
  document.getElementById('result-path').textContent = res.path;
  const img = document.getElementById('result-img');
  img.src = 'data:image/jpeg;base64,' + res.image;
  img.style.display = 'block';
  document.getElementById('viewer-hint').style.display = 'none';
  img.onload = resetZoom;
  switchTab('result');
}

// ── Zoom / Pan
const viewer = document.getElementById('viewer');
const rimg   = document.getElementById('result-img');

viewer.addEventListener('wheel', e => {
  e.preventDefault();
  const rect = viewer.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const f = e.deltaY < 0 ? 1.15 : 0.87;
  const ns = Math.min(Math.max(0.05, sc * f), 40);
  tx = mx - (mx - tx) * (ns / sc);
  ty = my - (my - ty) * (ns / sc);
  sc = ns;
  applyT();
}, {passive: false});

viewer.addEventListener('mousedown', e => {
  dragging = true; sx = e.clientX - tx; sy = e.clientY - ty;
  viewer.classList.add('drag');
});
document.addEventListener('mousemove', e => {
  if (!dragging) return;
  tx = e.clientX - sx; ty = e.clientY - sy; applyT();
});
document.addEventListener('mouseup', () => { dragging = false; viewer.classList.remove('drag'); });

function applyT() { rimg.style.transform = `translate(${tx}px,${ty}px) scale(${sc})`; }

function resetZoom() {
  const vw = viewer.clientWidth, vh = viewer.clientHeight;
  if (!rimg.naturalWidth) return;
  sc = Math.min(vw / rimg.naturalWidth, vh / rimg.naturalHeight) * 0.95;
  tx = (vw - rimg.naturalWidth  * sc) / 2;
  ty = (vh - rimg.naturalHeight * sc) / 2;
  applyT();
}

// ── 중지
let currentES = null;
async function stopDetect() {
  if (!currentJobId) return;
  await post('/api/stop/' + currentJobId, {});
  document.getElementById('prog-log').textContent = '중지 요청됨...';
}

function setRunning(on) {
  document.getElementById('btn-run').disabled = on;
  document.getElementById('btn-stop').classList.toggle('visible', on);
}

// ── 폴더 열기
async function openFolder() {
  await post('/api/open-folder', {path: resultPath || 'output/detect'});
}

// ── 유틸
async function post(url, data) {
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
  return r.json();
}

// ── 초기화
initRows();
document.getElementById('image-path').addEventListener('keydown', e => {
  if (e.key === 'Enter') loadPreview();
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("Railway Detection UI 시작: http://localhost:7000")
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level="warning")
