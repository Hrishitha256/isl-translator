# web_app.py
"""
web_app.py - browser version of the ISL translator.
Camera on the left, text + controls on the right, speech through the browser.

Run:  python web_app.py
Then open:  http://127.0.0.1:5000
"""

import threading
import time

import cv2
from flask import Flask, Response, jsonify, request

from app import SignRecognizer      # reuse the recognition logic from app.py

server = Flask(__name__)
rec = SignRecognizer()
lock = threading.Lock()
latest_jpeg = None


def camera_loop():
    """Read the webcam forever, run recognition, keep the newest JPEG."""
    global latest_jpeg
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
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
        return jsonify(text=rec.text, label=rec.label, conf=rec.conf)


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


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ISL Translator</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
  header { padding: 14px 24px; background: #1d4ed8; font-size: 22px; font-weight: 600; }
  .wrap { display: grid; grid-template-columns: 1.3fr 1fr; gap: 24px; padding: 24px; }
  .card { background: #1e293b; border-radius: 14px; padding: 18px; }
  img { width: 100%; border-radius: 10px; display: block; }
  h2 { margin: 0 0 8px; font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
  #seen { font-size: 40px; font-weight: 700; color: #4ade80; margin-bottom: 18px; }
  #text { font-size: 44px; min-height: 120px; word-break: break-all; background: #0f172a;
          border-radius: 10px; padding: 14px; margin-bottom: 16px; }
  button { font-size: 16px; padding: 10px 18px; margin: 0 8px 8px 0; border: 0; border-radius: 8px;
           background: #334155; color: #fff; cursor: pointer; }
  button.main { background: #2563eb; }
  button:hover { filter: brightness(1.2); }
  label { font-size: 15px; }
  @media (max-width: 900px) { .wrap { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>Indian Sign Language to Text / Speech</header>
<div class="wrap">
  <div class="card"><h2>Live camera</h2><img src="/video"></div>
  <div class="card">
    <h2>Detected sign</h2><div id="seen">-</div>
    <h2>Text</h2><div id="text"></div>
    <button class="main" onclick="speakAll()">Speak</button>
    <button onclick="act('space')">Space</button>
    <button onclick="act('backspace')">Backspace</button>
    <button onclick="act('clear')">Clear</button><br>
    <label><input type="checkbox" id="auto" checked> Auto-speak each sign</label>
  </div>
</div>
<script>
let prev = "";
function say(t) {
  if (!t || !t.trim()) return;
  const u = new SpeechSynthesisUtterance(t);
  u.rate = 0.9;
  speechSynthesis.speak(u);
}
function speakAll() { say(document.getElementById("text").textContent); }
async function act(d) {
  await fetch("/action", {method: "POST", headers: {"Content-Type": "application/json"},
                          body: JSON.stringify({do: d})});
}
async function tick() {
  try {
    const s = await (await fetch("/state")).json();
    document.getElementById("text").textContent = s.text;
    document.getElementById("seen").textContent =
      s.label ? s.label + "  (" + Math.round(s.conf * 100) + "%)" : "-";
    if (document.getElementById("auto").checked &&
        s.text.length > prev.length && s.text.startsWith(prev)) {
      say(s.text.slice(prev.length));
    }
    prev = s.text;
  } catch (e) {}
}
setInterval(tick, 250);
</script>
</body>
</html>
"""


@server.route("/")
def index():
    return PAGE


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    print("Open http://127.0.0.1:5000 in your browser (Ctrl+C to stop)")
    server.run(host="127.0.0.1", port=5000, threaded=True, debug=False)