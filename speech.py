# speech.py
"""
speech.py - text-to-speech for the ISL translator (offline, uses pyttsx3).

Usage from other files:
    from speech import speak
    speak("hello 123")

speak() returns immediately. The talking happens in a background thread,
so the video never freezes. If new text arrives while something is still
being spoken, the old queued text is dropped and only the newest is spoken.
"""

import queue
import threading

import pyttsx3

RATE = 150      # words per minute (default is about 200, lower = slower)
VOLUME = 1.0    # 0.0 to 1.0

_q = queue.Queue()
_worker = None
_lock = threading.Lock()


def _say(text):
    # A fresh engine for every sentence is the most reliable way to use
    # pyttsx3 on Windows from a background thread.
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", RATE)
        engine.setProperty("volume", VOLUME)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("Speech error:", e)


def _run():
    while True:
        text = _q.get()
        # skip ahead to the newest request if several piled up
        while not _q.empty():
            text = _q.get_nowait()
        _say(text)


def speak(text):
    """Speak text out loud without blocking."""
    global _worker
    text = (text or "").strip()
    if not text:
        return
    with _lock:
        if _worker is None:
            _worker = threading.Thread(target=_run, daemon=True)
            _worker.start()
    _q.put(text)


if __name__ == "__main__":
    # Self-test:  python speech.py
    import time
    speak("Speech is working. One, two, three.")
    time.sleep(5)   # keep the program alive long enough to finish talking
