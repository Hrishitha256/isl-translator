"""
web_app.py - "Prajna": browser version of the ISL translator.
Camera on the left, detected sign + sentence on the right, speech in the browser.

Run:  python web_app.py
Then open (in the address bar):  127.0.0.1:5000
"""

import threading
import time

import cv2
from flask import Flask, Response, jsonify, request

from app import SignRecognizer      # recognition logic lives in app.py

server = Flask(__name__)
rec = SignRecognizer()
lock = threading.Lock()
latest_jpeg = None
camera_ok = False


def camera_loop():
    """Read the webcam forever, run recognition, keep the newest JPEG."""
    global latest_jpeg, camera_ok
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        camera_ok = bool(ok)
        if not ok:
            time.sleep(0.1)
            continue
        with lock:
            frame = rec.process(frame)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            latest_jpeg = buf.tobytes()


def mjpeg():
    while True:
        if latest_jpeg is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + latest_jpeg + b"\r\n")
        time.sleep(0.03)


@server.route("/video")
def video():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@server.route("/state")
def state():
    with lock:
        return jsonify(text=rec.text, label=rec.label, conf=rec.conf,
                       hand=rec.label is not None, camera=camera_ok,
                       classes=rec.classes)


@server.route("/action", methods=["POST"])
def action():
    do = (request.get_json(silent=True) or {}).get("do")
    with lock:
        if do == "space":
            rec.space()
        elif do == "backspace":
            rec.backspace()
        elif do == "clear":
            rec.clear()
    return jsonify(ok=True)


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prajna - ISL Translator</title>
<style>
  :root {
    --bg: #0e1130; --card: #1b2050; --feed: #0a0c26; --chip: #232a63;
    --text: #ffffff; --muted: #a9b0e0; --amber: #f59e0b; --green: #1f8f63; --grey: #4b5280;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font-family: "Segoe UI", Arial, sans-serif; }
  .page { max-width: 1180px; margin: 0 auto; padding: 28px 32px; }
  header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 22px; }
  h1 { margin: 0; font-size: 28px; }
  .sub { color: var(--muted); font-size: 14px; margin-top: 2px; }
  .pill { padding: 8px 18px; border-radius: 999px; font-size: 14px; font-weight: 600; background: var(--grey); }
  .pill.on { background: var(--green); }
  .grid { display: grid; grid-template-columns: 1.55fr 1fr; gap: 22px; }
  .card { background: var(--card); border-radius: 20px; padding: 24px; }
  .feed { position: relative; background: var(--feed); border-radius: 20px; min-height: 420px;
          overflow: hidden; display: flex; align-items: center; justify-content: center; }
  .feed .ph { color: var(--muted); font-size: 16px; position: absolute; }
  .feed img { position: relative; width: 100%; display: block; }
  .badge { position: absolute; top: 14px; left: 14px; z-index: 2; padding: 6px 14px;
           border-radius: 999px; font-size: 13px; font-weight: 700; background: var(--grey); color: #fff; }
  .badge.on { background: var(--amber); color: #1a1200; }
  .label { color: var(--muted); font-size: 14px; margin-bottom: 8px; }
  #sign { font-size: 112px; font-weight: 800; color: var(--amber); line-height: 1.05; min-height: 118px; }
  .bar { height: 10px; border-radius: 999px; background: #0d1030; overflow: hidden; margin: 14px 0 8px; }
  .bar div { height: 100%; width: 0%; background: var(--amber); transition: width .2s; }
  .side { display: flex; flex-direction: column; gap: 22px; }
  #text { font-size: 40px; font-weight: 800; letter-spacing: 2px; min-height: 56px;
          word-break: break-all; margin: 6px 0 18px; text-transform: uppercase; }
  #text.empty { color: #6b73b0; font-size: 18px; font-weight: 500; letter-spacing: 0;
                text-transform: none; padding-top: 14px; }
  button { font: inherit; font-weight: 700; font-size: 15px; padding: 11px 22px; border-radius: 12px;
           cursor: pointer; margin: 0 8px 8px 0; border: 2px solid #8089d6; background: transparent; color: #fff; }
  button.main { background: var(--amber); border-color: var(--amber); color: #1a1200; }
  button:hover { filter: brightness(1.15); }
  .auto { color: var(--muted); font-size: 14px; display: block; margin-top: 4px; }
  .signs-title { color: var(--muted); font-size: 14px; margin: 26px 0 12px; }
  .chips { display: flex; flex-wrap: wrap; gap: 10px; }
  .chip { width: 40px; height: 40px; display: flex; align-items: center; justify-content: center;
          border-radius: 10px; background: var(--chip); font-weight: 700; font-size: 15px; transition: all .15s; }
  .chip.hot { background: var(--amber); color: #1a1200; transform: scale(1.12); }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="page">
  <header>
    <div>
      <h1>Prajna</h1>
      <div class="sub">Indian Sign Language to text and speech</div>
    </div>
    <div class="pill" id="cam">Camera off</div>
  </header>

  <div class="grid">
    <div class="feed">
      <span class="badge" id="hand">No hand</span>
      <span class="ph">Live camera feed</span>
      <img src="/video" alt="">
    </div>

    <div class="side">
      <div class="card">
        <div class="label">Detected sign</div>
        <div id="sign">-</div>
        <div class="bar"><div id="fill"></div></div>
        <div class="label" id="conf">Confidence 0%</div>
      </div>

      <div class="card">
        <div class="label">Sentence</div>
        <div id="text" class="empty">Show a sign to begin</div>
        <button class="main" onclick="speakAll()">Speak</button>
        <button onclick="act('space')">Space</button>
        <button onclick="act('backspace')">Backspace</button>
        <button onclick="act('clear')">Clear</button>
        <label class="auto"><input type="checkbox" id="auto" checked> Speak each sign automatically</label>
      </div>
    </div>
  </div>

  <div class="signs-title">Supported signs</div>
  <div class="chips" id="chips"></div>
</div>

<script>
let prev = "";
let chipsBuilt = false;

function say(t) {
  if (!t || !t.trim()) return;
  const u = new SpeechSynthesisUtterance(t);
  u.rate = 0.9;
  speechSynthesis.speak(u);
}
function speakAll() {
  const t = document.getElementById("text").dataset.value || "";
  say(t.replace(/\d/g, d => d + " "));        // "123" is read as "1 2 3"
}
async function act(d) {
  await fetch("/action", {method: "POST", headers: {"Content-Type": "application/json"},
                          body: JSON.stringify({do: d})});
}
function buildChips(classes) {
  const box = document.getElementById("chips");
  box.innerHTML = "";
  classes.forEach(c => {
    const el = document.createElement("div");
    el.className = "chip"; el.id = "chip-" + c; el.textContent = c;
    box.appendChild(el);
  });
  chipsBuilt = true;
}
async function tick() {
  try {
    const s = await (await fetch("/state")).json();
    if (!chipsBuilt) buildChips(s.classes);

    const cam = document.getElementById("cam");
    cam.textContent = s.camera ? "Camera on" : "Camera off";
    cam.className = "pill" + (s.camera ? " on" : "");

    const hand = document.getElementById("hand");
    hand.textContent = s.hand ? "Hand detected" : "No hand";
    hand.className = "badge" + (s.hand ? " on" : "");

    document.getElementById("sign").textContent = s.label ? s.label : "-";
    const pct = s.label ? Math.round(s.conf * 100) : 0;
    document.getElementById("fill").style.width = pct + "%";
    document.getElementById("conf").textContent = "Confidence " + pct + "%";

    document.querySelectorAll(".chip").forEach(c => c.classList.remove("hot"));
    if (s.label) {
      const c = document.getElementById("chip-" + s.label);
      if (c) c.classList.add("hot");
    }

    const t = document.getElementById("text");
    t.dataset.value = s.text;
    if (s.text) { t.textContent = s.text; t.className = ""; }
    else { t.textContent = "Show a sign to begin"; t.className = "empty"; }

    if (document.getElementById("auto").checked &&
        s.text.length > prev.length && s.text.startsWith(prev)) {
      say(s.text.slice(prev.length));
    }
    prev = s.text;
  } catch (e) {}
}
setInterval(tick, 200);
</script>
</body>
</html>
"""


@server.route("/")
def index():
    return PAGE


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    print("Open 127.0.0.1:5000 in your browser's address bar (Ctrl+C to stop)")
    server.run(host="127.0.0.1", port=5000, threaded=True, debug=False)