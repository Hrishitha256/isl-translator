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
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prajna - ISL Translator</title>
<style>
  :root {
    --bg: #0b0e2a; --card: #161b47; --feed: #080a1f; --chip: #212860;
    --text: #ffffff; --muted: #a3abdc; --line: #2b3380;
    --amber: #f59e0b; --green: #22c58b; --red: #f0616d; --grey: #4b5280;
    --on-amber: #1a1200;
  }
  [data-theme="light"] {
    --bg: #f2f4fc; --card: #ffffff; --feed: #dfe3f5; --chip: #e6e9f8;
    --text: #141838; --muted: #5a6290; --line: #c4caea;
    --amber: #e08600; --green: #12935f; --red: #d63c49; --grey: #7b82ab;
    --on-amber: #ffffff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font-family: "Segoe UI", Arial, sans-serif; transition: background .2s; }
  .page { max-width: 1180px; margin: 0 auto; padding: 26px 32px 40px; }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; gap: 12px; }
  h1 { margin: 0; font-size: 28px; letter-spacing: .3px; }
  .sub { color: var(--muted); font-size: 14px; margin-top: 2px; }
  .right { display: flex; align-items: center; gap: 10px; }
  .pill { padding: 8px 18px; border-radius: 999px; font-size: 14px; font-weight: 600; background: var(--grey); color: #fff; }
  .pill.on { background: var(--green); }
  .icon-btn { margin: 0; padding: 8px 14px; border-radius: 999px; font-size: 14px; }

  .grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 22px; }
  .card { background: var(--card); border-radius: 20px; padding: 24px; }
  .label { color: var(--muted); font-size: 14px; margin-bottom: 10px; }

  /* camera */
  .feed { position: relative; background: var(--feed); border-radius: 20px; min-height: 420px;
          overflow: hidden; display: flex; align-items: center; justify-content: center;
          border: 3px solid transparent; transition: border-color .2s; }
  .feed.found { border-color: var(--amber); }
  .feed.sure  { border-color: var(--green); }
  .feed .ph { color: var(--muted); position: absolute; }
  .feed img { position: relative; width: 100%; display: block; }
  .badge { position: absolute; top: 14px; left: 14px; z-index: 2; padding: 6px 14px;
           border-radius: 999px; font-size: 13px; font-weight: 700; background: var(--grey); color: #fff; }
  .badge.found { background: var(--amber); color: var(--on-amber); }
  .badge.sure  { background: var(--green); color: #fff; }

  /* detected sign + ring */
  .side { display: flex; flex-direction: column; gap: 22px; }
  .detect { display: flex; align-items: center; gap: 24px; }
  .ring { position: relative; width: 150px; height: 150px; flex: none; }
  .ring svg { transform: rotate(-90deg); }
  .ring .track { stroke: var(--chip); }
  #arc { stroke: var(--grey); transition: stroke-dashoffset .25s, stroke .25s; }
  #sign { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
          font-size: 68px; font-weight: 800; }
  .meta { flex: 1; }
  #conf { font-size: 34px; font-weight: 800; }
  .hint { color: var(--muted); font-size: 14px; margin-top: 4px; }

  /* sentence */
  #text { font-size: 38px; font-weight: 800; letter-spacing: 2px; min-height: 52px;
          word-break: break-all; margin: 4px 0 18px; text-transform: uppercase; }
  #text.empty { color: var(--muted); font-size: 17px; font-weight: 500; letter-spacing: 0;
                text-transform: none; padding-top: 14px; opacity: .8; }
  button { font: inherit; font-weight: 700; font-size: 15px; padding: 10px 20px; border-radius: 12px;
           cursor: pointer; margin: 0 8px 8px 0; border: 2px solid var(--line); background: transparent; color: var(--text); }
  button.main { background: var(--amber); border-color: var(--amber); color: var(--on-amber); }
  button:hover { filter: brightness(1.12); }
  button:focus-visible, select:focus-visible { outline: 3px solid var(--amber); outline-offset: 2px; }
  .opts { display: flex; align-items: center; flex-wrap: wrap; gap: 18px; margin-top: 8px; color: var(--muted); font-size: 14px; }
  select { font: inherit; background: var(--chip); color: var(--text); border: 2px solid var(--line);
           border-radius: 10px; padding: 6px 10px; }

  /* bottom row */
  .bottom { display: grid; grid-template-columns: 1.5fr 1fr; gap: 22px; margin-top: 22px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { width: 38px; height: 38px; display: flex; align-items: center; justify-content: center;
          border-radius: 10px; background: var(--chip); font-weight: 700; font-size: 14px; transition: all .15s; }
  .chip.hot { background: var(--amber); color: var(--on-amber); transform: scale(1.12); }
  #history { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; max-height: 180px; overflow: auto; }
  #history li { background: var(--chip); border-radius: 10px; padding: 9px 14px; cursor: pointer;
                display: flex; justify-content: space-between; gap: 10px; font-weight: 600; word-break: break-all; }
  #history li span { color: var(--muted); font-weight: 500; font-size: 13px; flex: none; }
  .empty-note { color: var(--muted); font-size: 14px; }

  @media (max-width: 900px) { .grid, .bottom { grid-template-columns: 1fr; } }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<div class="page">
  <header>
    <div>
      <h1>Prajna</h1>
      <div class="sub">Indian Sign Language to text and speech</div>
    </div>
    <div class="right">
      <button class="icon-btn" id="theme" onclick="toggleTheme()">Light mode</button>
      <div class="pill" id="cam">Camera off</div>
    </div>
  </header>

  <div class="grid">
    <div class="feed" id="feed">
      <span class="badge" id="hand">Searching for hand</span>
      <span class="ph">Live camera feed</span>
      <img src="/video" alt="">
    </div>

    <div class="side">
      <div class="card">
        <div class="label">Detected sign</div>
        <div class="detect">
          <div class="ring">
            <svg width="150" height="150" viewBox="0 0 150 150" fill="none" stroke-width="12" stroke-linecap="round">
              <circle class="track" cx="75" cy="75" r="64"/>
              <circle id="arc" cx="75" cy="75" r="64" stroke-dasharray="402.1" stroke-dashoffset="402.1"/>
            </svg>
            <div id="sign">-</div>
          </div>
          <div class="meta">
            <div id="conf">0%</div>
            <div class="hint" id="hint">Show a sign to the camera</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="label">Sentence</div>
        <div id="text" class="empty">Your signs will appear here</div>
        <button class="main" onclick="speakAll()">Speak</button>
        <button onclick="act('space')">Space</button>
        <button onclick="act('backspace')">Backspace</button>
        <button onclick="act('clear')">Clear</button>
        <div class="opts">
          <label>Voice language
            <select id="lang">
              <option value="en-IN">English</option>
              <option value="hi-IN">Hindi</option>
              <option value="te-IN">Telugu</option>
              <option value="ta-IN">Tamil</option>
            </select>
          </label>
          <label><input type="checkbox" id="auto" checked> Speak each sign automatically</label>
        </div>
      </div>
    </div>
  </div>

  <div class="bottom">
    <div class="card">
      <div class="label">Supported signs</div>
      <div class="chips" id="chips"></div>
    </div>
    <div class="card">
      <div class="label">Spoken sentences (tap to replay)</div>
      <ul id="history"></ul>
      <div class="empty-note" id="hist-empty">Sentences you speak will be saved here.</div>
    </div>
  </div>
</div>

<script>
const CIRC = 402.1;
let prev = "";
let chipsBuilt = false;
const $ = id => document.getElementById(id);

/* theme */
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  $("theme").textContent = t === "dark" ? "Light mode" : "Dark mode";
  try { localStorage.setItem("prajna-theme", t); } catch (e) {}
}
function toggleTheme() {
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
}
try { setTheme(localStorage.getItem("prajna-theme") || "dark"); } catch (e) {}

/* speech */
function say(t) {
  if (!t || !t.trim()) return;
  const u = new SpeechSynthesisUtterance(t);
  u.lang = $("lang").value;
  u.rate = 0.9;
  const v = speechSynthesis.getVoices().find(v => v.lang === u.lang);
  if (v) u.voice = v;
  speechSynthesis.speak(u);
}
function speakAll() {
  const t = $("text").dataset.value || "";
  if (!t.trim()) return;
  say(t.replace(/\d/g, d => d + " "));        // "123" is read as "1 2 3"
  addHistory(t);
}

/* history */
function addHistory(t) {
  const list = $("history");
  if (list.firstChild && list.firstChild.dataset.value === t) return;
  const li = document.createElement("li");
  li.dataset.value = t;
  const name = document.createElement("div");
  name.textContent = t;
  const tm = document.createElement("span");
  tm.textContent = new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
  li.append(name, tm);
  li.onclick = () => say(t.replace(/\d/g, d => d + " "));
  list.prepend(li);
  $("hist-empty").style.display = "none";
}

async function act(d) {
  await fetch("/action", {method: "POST", headers: {"Content-Type": "application/json"},
                          body: JSON.stringify({do: d})});
}

function buildChips(classes) {
  const box = $("chips");
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

    const cam = $("cam");
    cam.textContent = s.camera ? "Camera on" : "Camera off";
    cam.className = "pill" + (s.camera ? " on" : "");

    /* hand state: searching -> detected -> recognized */
    const pct = s.label ? Math.round(s.conf * 100) : 0;
    const state = !s.hand ? "" : (pct >= 80 ? "sure" : "found");
    $("hand").textContent = !s.hand ? "Searching for hand" : (pct >= 80 ? "Recognized" : "Hand detected");
    $("hand").className = "badge " + state;
    $("feed").className = "feed " + state;

    /* ring */
    const color = !s.hand ? "var(--grey)" : pct >= 80 ? "var(--green)" : pct >= 50 ? "var(--amber)" : "var(--red)";
    $("arc").style.strokeDashoffset = CIRC * (1 - pct / 100);
    $("arc").style.stroke = color;
    $("sign").textContent = s.label ? s.label : "-";
    $("sign").style.color = s.label ? color : "var(--muted)";
    $("conf").textContent = pct + "%";
    $("hint").textContent = !s.hand ? "Show a sign to the camera"
                          : pct >= 80 ? "Clear sign" : "Hold steady for a clearer read";

    document.querySelectorAll(".chip").forEach(c => c.classList.remove("hot"));
    if (s.label) {
      const c = $("chip-" + s.label);
      if (c) c.classList.add("hot");
    }

    const t = $("text");
    t.dataset.value = s.text;
    if (s.text) { t.textContent = s.text; t.className = ""; }
    else { t.textContent = "Your signs will appear here"; t.className = "empty"; }

    if ($("auto").checked && s.text.length > prev.length && s.text.startsWith(prev)) {
      say(s.text.slice(prev.length));
    }
    prev = s.text;
  } catch (e) {}
}
speechSynthesis.getVoices();
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