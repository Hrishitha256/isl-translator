# collect.py
"""
collect.py - record hand-landmark samples for each sign.

Run:  python collect.py --name hrishitha
Each person uses their own --name, so each person gets their own file
(data/<name>.csv). train.py will combine all files in data/.

Controls (click on the video window first):
  n  = next sign          b  = previous sign
  s  = save ONE sample    r  = start/stop AUTO-record (2 second countdown first)
  u  = undo last sample   q  = quit
"""

import argparse
import csv
import os
import time

import cv2

from features import (create_hands, extract_features, draw_landmarks,
                      hand_detected, NUM_FEATURES)

# ---- EDIT THIS LIST once the team decides the vocabulary ----
# Everyone must spell the labels exactly the same way.
LABELS = ["1", "2", "3", "4", "5",
          "A", "B", "C", "D", "E",
          "Hello", "Thank you", "Yes", "No"]
# --------------------------------------------------------------

DATA_DIR = "data"
TARGET_PER_LABEL = 150     # samples to record per sign
AUTO_INTERVAL = 0.1        # seconds between auto-saved samples
COUNTDOWN = 2.0            # seconds to get ready after pressing 'r'


def csv_path(name):
    return os.path.join(DATA_DIR, f"{name}.csv")


def ensure_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])


def load_counts(path):
    counts = {label: 0 for label in LABELS}
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts


def save_sample(path, label, features):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([label] + [f"{v:.6f}" for v in features])


def undo_last(path):
    """Delete the last saved row. Returns its label, or None if nothing to undo."""
    with open(path, newline="") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return None
    last_label = lines[-1].split(",")[0]
    with open(path, "w", newline="") as f:
        f.writelines(lines[:-1])
    return last_label


def put(frame, text, pos, color=(255, 255, 255), scale=0.7):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="your name, e.g. hrishitha (one data file per person)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    path = csv_path(args.name.lower())
    ensure_header(path)
    counts = load_counts(path)

    cap = cv2.VideoCapture(0)
    hands = create_hands()

    idx = 0
    auto = False
    auto_start = 0.0
    last_save = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from the webcam.")
            break
        frame = cv2.flip(frame, 1)      # always flip BEFORE extracting features
        features, results = extract_features(frame, hands)
        draw_landmarks(frame, results)
        detected = hand_detected(results)

        label = LABELS[idx]
        now = time.time()

        # ---- auto-record logic ----
        status, status_color = "Ready", (255, 255, 255)
        if auto:
            wait = COUNTDOWN - (now - auto_start)
            if wait > 0:
                status, status_color = f"Starting in {wait:.1f}s - get ready", (0, 255, 255)
            elif not detected:
                status, status_color = "RECORDING paused - no hand seen", (0, 165, 255)
            else:
                status, status_color = "RECORDING", (0, 0, 255)
                if now - last_save >= AUTO_INTERVAL:
                    save_sample(path, label, features)
                    counts[label] += 1
                    last_save = now
                    if counts[label] >= TARGET_PER_LABEL:
                        auto = False
                        print(f"Reached {TARGET_PER_LABEL} samples for '{label}'. Press n for the next sign.")

        # ---- on-screen info ----
        put(frame, f"Sign: {label}  ({idx + 1}/{len(LABELS)})", (10, 30), (0, 255, 0), 0.9)
        put(frame, f"Samples: {counts[label]}/{TARGET_PER_LABEL}   Total: {sum(counts.values())}", (10, 65))
        put(frame, status, (10, 100), status_color)
        put(frame, "n/b: sign | s: save 1 | r: auto-record | u: undo | q: quit",
            (10, frame.shape[0] - 15), (200, 200, 200), 0.55)

        cv2.imshow("collect.py", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            idx = (idx + 1) % len(LABELS)
            auto = False
        elif key == ord("b"):
            idx = (idx - 1) % len(LABELS)
            auto = False
        elif key == ord("s"):
            if detected:
                save_sample(path, label, features)
                counts[label] += 1
            else:
                print("No hand detected - sample not saved.")
        elif key == ord("r"):
            auto = not auto
            auto_start = time.time()
        elif key == ord("u"):
            undone = undo_last(path)
            if undone in counts and counts[undone] > 0:
                counts[undone] -= 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nSaved to:", path)
    for lab in LABELS:
        print(f"  {lab}: {counts[lab]}")


if __name__ == "__main__":
    main()