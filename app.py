# app.py

"""
app.py - live sign recognition (Member 2).

Run:  python app.py

Keys (click on the video window first):
  SPACE     = add a space          BACKSPACE = delete last character
  c         = clear the text       s         = speak the text (needs speech.py)
  q         = quit
"""

import os
from collections import deque

import cv2
import joblib
import numpy as np

from features import create_hands, extract_features, draw_landmarks, hand_detected

try:
    from speech import speak        # Member 3 writes speak(text) in speech.py
except ImportError:
    speak = None

MODEL_PATH = os.path.join("models", "model.pkl")

# ---- tuning knobs ----
CONF_THRESHOLD = 0.60   # ignore guesses below this confidence
STABLE_FRAMES = 12      # same sign for this many frames in a row = accepted
RELEASE_FRAMES = 8      # no hand for this many frames = hand "released" (lets you repeat a sign)
# ----------------------


def put(frame, text, pos, color=(255, 255, 255), scale=0.8, thick=2):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


class SignRecognizer:
    """Turns webcam frames into a stable text string. Member 3 can reuse this."""

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise SystemExit(f"Model not found: {model_path}\nRun 'python train.py' or 'git pull' first.")
        self.model = joblib.load(model_path)
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1           # faster for one-sample predictions
        self.classes = list(self.model.classes_)
        self.hands = create_hands()

        self.recent = deque(maxlen=STABLE_FRAMES)
        self.text = ""
        self.label = None                   # current guess
        self.conf = 0.0
        self.progress = 0                   # frames the current guess has been stable
        self.last_accepted = None
        self.released = True
        self.no_hand_frames = 0

    def _accept(self, label):
        if len(label) == 1:                 # digit or letter
            self.text += label
        else:                               # whole word/phrase
            if self.text and not self.text.endswith(" "):
                self.text += " "
            self.text += label + " "

    def process(self, frame):
        """Takes a raw BGR frame, returns the flipped, annotated frame."""
        frame = cv2.flip(frame, 1)          # same flip as collect.py
        features, results = extract_features(frame, self.hands)
        draw_landmarks(frame, results)

        if hand_detected(results):
            self.no_hand_frames = 0
            probs = self.model.predict_proba(features.reshape(1, -1))[0]
            i = int(np.argmax(probs))
            self.label, self.conf = self.classes[i], float(probs[i])
            if self.conf >= CONF_THRESHOLD:
                self.recent.append(self.label)
            else:
                self.recent.clear()
        else:
            self.no_hand_frames += 1
            self.label, self.conf = None, 0.0
            self.recent.clear()
            if self.no_hand_frames >= RELEASE_FRAMES:
                self.released = True

        # how many of the latest guesses agree (for the progress bar)
        self.progress = 0
        if self.recent:
            newest = self.recent[-1]
            for lab in reversed(self.recent):
                if lab != newest:
                    break
                self.progress += 1

        # accept a sign once it has been stable long enough
        if len(self.recent) == STABLE_FRAMES and len(set(self.recent)) == 1:
            candidate = self.recent[0]
            if candidate != self.last_accepted or self.released:
                self._accept(candidate)
                self.last_accepted = candidate
                self.released = False
                self.recent.clear()
                self.progress = 0
        return frame

    def backspace(self):
        self.text = self.text[:-1]

    def space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def clear(self):
        self.text = ""
        self.last_accepted = None
        self.released = True


def draw_ui(frame, rec):
    h, w = frame.shape[:2]
    if rec.label is None:
        put(frame, "Show a sign to the camera", (10, 35), (200, 200, 200))
    else:
        color = (0, 255, 0) if rec.conf >= CONF_THRESHOLD else (0, 165, 255)
        put(frame, f"Seeing: {rec.label}  ({rec.conf * 100:.0f}%)", (10, 35), color, 0.9)
        # progress bar: fills up while the sign is held steady
        bar_w = int(200 * rec.progress / STABLE_FRAMES)
        cv2.rectangle(frame, (10, 50), (210, 62), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 50), (10 + bar_w, 62), (0, 255, 0), -1)

    # text strip at the bottom
    cv2.rectangle(frame, (0, h - 90), (w, h), (30, 30, 30), -1)
    shown = rec.text[-28:] if rec.text else ""
    put(frame, "Text: " + shown + "_", (10, h - 50), (255, 255, 255), 1.1, 2)
    put(frame, "SPACE: space | BACKSPACE: delete | c: clear | s: speak | q: quit",
        (10, h - 15), (180, 180, 180), 0.5, 1)


def main():
    rec = SignRecognizer()
    print("Model knows:", rec.classes)
    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from the webcam.")
            break
        frame = rec.process(frame)
        draw_ui(frame, rec)
        cv2.imshow("ISL Translator - press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:                     # SPACE
            rec.space()
        elif key == 8:                      # BACKSPACE
            rec.backspace()
        elif key == ord("c"):
            rec.clear()
        elif key == ord("s"):
            if speak is None:
                print("(speech not ready yet - speech.py has no speak() function)")
            elif rec.text.strip():
                speak(rec.text.strip())

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()