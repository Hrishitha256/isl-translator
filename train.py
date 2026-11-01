# train.py  -  ISL sign to text / speech (training + live demo in one file)
#
#   python train.py        -> train on hrishitha.csv, test on saisree.csv, save model
#   python train.py live   -> open camera, predict signs, show text and speak

import os
import pickle
import sys
import threading
from collections import Counter, deque

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TRAIN_FILE = "data/hrishitha.csv"
TEST_FILE = "data/saisree.csv"
MODEL_PATH = "models/model.pkl"

# ---- Must match how collect.py built the CSV (126 = 2 hands x 21 points x 3) ----
NORMALIZE = False          # True if collect.py subtracted the wrist and scaled
USE_HANDEDNESS = True      # True: slot 0 = Left hand, slot 1 = Right hand
                           # False: slots follow MediaPipe detection order
FLIP_FRAME = True          # False if collect.py did NOT mirror the camera image

# ---- Live settings ----
MIN_CONFIDENCE = 0.60      # ignore weak predictions
SMOOTHING_FRAMES = 10      # look at the last N frames
AGREE_FRAMES = 8           # how many must agree before showing/speaking


# ----------------------------------------------------------------------
# Features (the live code must produce the same features as collect.py)
# ----------------------------------------------------------------------
def extract_features(multi_hand_landmarks, multi_handedness=None):
    """Return 126 numbers: [slot 0 (63) | slot 1 (63)], zero-padded."""
    slots = [np.zeros(63, dtype=np.float32), np.zeros(63, dtype=np.float32)]
    for i, hand in enumerate(multi_hand_landmarks[:2]):
        pts = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
        if NORMALIZE:
            pts = pts - pts[0]
            pts = pts / (np.max(np.abs(pts)) or 1.0)
        slot = i
        if USE_HANDEDNESS and multi_handedness is not None:
            label = multi_handedness[i].classification[0].label
            slot = 0 if label == "Left" else 1
        slots[slot] = pts.flatten()
    return np.concatenate(slots)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def load_file(path):
    df = pd.read_csv(path)

    # If the CSV has no header row, reload without one
    try:
        float(df.columns[1])
        df = pd.read_csv(path, header=None)
    except (ValueError, IndexError):
        pass

    # Label column: named label/class/sign/letter, otherwise the first column
    label_col = df.columns[0]
    for c in df.columns:
        if str(c).lower() in ("label", "class", "sign", "letter"):
            label_col = c
            break

    y = df[label_col].astype(str).values
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0).values.astype(np.float32)
    return X, y


def train():
    X_train, y_train = load_file(TRAIN_FILE)
    X_test, y_test = load_file(TEST_FILE)

    print(f"Train ({TRAIN_FILE}): {len(y_train)} samples, {X_train.shape[1]} features")
    print(f"Test  ({TEST_FILE}): {len(y_test)} samples, {X_test.shape[1]} features")

    if X_train.shape[1] != X_test.shape[1]:
        raise SystemExit("Feature counts differ between the two files. Check the CSV columns.")

    print("Train classes:", sorted(set(y_train)))
    print("Test classes: ", sorted(set(y_test)))

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, pred):.3f}\n")
    print(classification_report(y_test, pred, zero_division=0))

    labels = sorted(set(y_train) | set(y_test))
    print("Confusion matrix (rows = true, columns = predicted):")
    print("labels:", labels)
    print(confusion_matrix(y_test, pred, labels=labels))

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model to {MODEL_PATH}")


# ----------------------------------------------------------------------
# Speech
# ----------------------------------------------------------------------
_speak_lock = threading.Lock()


def speak(text):
    def run():
        if not _speak_lock.acquire(blocking=False):
            return                               # already speaking, skip
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("Speech error:", e)
        finally:
            _speak_lock.release()

    threading.Thread(target=run, daemon=True).start()


# ----------------------------------------------------------------------
# Live camera demo
# ----------------------------------------------------------------------
def live():
    import cv2
    import mediapipe as mp

    if not os.path.exists(MODEL_PATH):
        raise SystemExit("No model found. Run: python train.py")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    expected = model.n_features_in_

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")

    history = deque(maxlen=SMOOTHING_FRAMES)
    current = ""
    last_spoken = ""
    sentence = []
    conf = 0.0

    print("Live mode. Keys: q = quit, c = clear sentence, s = speak sentence")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            feats = extract_features(result.multi_hand_landmarks, result.multi_handedness)
            if feats.shape[0] != expected:
                raise SystemExit(
                    f"Feature mismatch: model expects {expected}, live gives {feats.shape[0]}. "
                    "Make extract_features/NORMALIZE match collect.py."
                )

            probs = model.predict_proba([feats])[0]
            best = int(np.argmax(probs))
            label, conf = str(model.classes_[best]), float(probs[best])
            history.append(label if conf >= MIN_CONFIDENCE else "")

            top, count = Counter(history).most_common(1)[0]
            if top and count >= AGREE_FRAMES:
                current = top
                if current != last_spoken:
                    speak(current)
                    sentence.append(current)
                    last_spoken = current
            cv2.putText(frame, f"conf {conf:.2f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        else:
            history.clear()
            last_spoken = ""                     # lets the same sign be repeated later
            current = ""

        cv2.putText(frame, f"Sign: {current}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        cv2.putText(frame, "Text: " + " ".join(sentence[-12:]), (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("ISL Translator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence.clear()
        elif key == ord("s") and sentence:
            speak(" ".join(sentence))

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "live":
        live()
    else:
        train()
