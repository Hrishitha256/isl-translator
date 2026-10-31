"""
features.py - shared landmark extraction for the ISL translator.

Every other script (collect.py, train.py, app.py) imports from here,
so training data and live predictions use exactly the same features.

Feature vector = 126 numbers:
  [0:63]    Left hand  (21 landmarks x (x, y, z))
  [63:126]  Right hand (21 landmarks x (x, y, z))
A hand that is not visible is filled with zeros.

IMPORTANT: always flip the frame (cv2.flip(frame, 1)) BEFORE calling
extract_features, in every script, so Left/Right labels stay consistent.
"""

import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

NUM_LANDMARKS = 21
VALUES_PER_HAND = NUM_LANDMARKS * 3   # 63
NUM_FEATURES = VALUES_PER_HAND * 2    # 126


def create_hands():
    """Create the MediaPipe hand detector (call once, reuse for every frame)."""
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )


def _normalize_hand(hand_landmarks):
    """Turn one hand's 21 landmarks into 63 position- and size-independent numbers."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                   dtype=np.float32)
    pts -= pts[0]                        # make the wrist the origin
    scale = np.linalg.norm(pts[9][:2])   # wrist -> middle finger base = hand size
    if scale < 1e-6:
        scale = 1.0
    pts /= scale                         # same hand size for near/far or big/small hands
    return pts.flatten()


def extract_features(frame_bgr, hands):
    """
    Run MediaPipe on one (already flipped) BGR frame.
    Returns (features, results):
      features: numpy array of length 126
      results:  raw MediaPipe results (use for drawing / checking detection)
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        used_slots = set()
        for hand_lms, handedness in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
            label = handedness.classification[0].label   # "Left" or "Right"
            slot = 0 if label == "Left" else 1
            if slot in used_slots:       # both hands got the same label
                slot = 1 - slot
            if slot in used_slots:
                continue
            used_slots.add(slot)
            start = slot * VALUES_PER_HAND
            features[start:start + VALUES_PER_HAND] = _normalize_hand(hand_lms)

    return features, results


def hand_detected(results):
    """True if at least one hand was found in this frame."""
    return results.multi_hand_landmarks is not None


def draw_landmarks(frame, results):
    """Draw the detected hand skeletons onto the frame."""
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)


if __name__ == "__main__":
    # Quick self-test: python features.py   (press q to quit)
    cap = cv2.VideoCapture(0)
    hands = create_hands()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        features, results = extract_features(frame, hands)
        draw_landmarks(frame, results)

        n = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        cv2.putText(frame, f"Hands: {n} | features: {features.shape[0]} | non-zero: {int((features != 0).sum())}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("features.py test - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
