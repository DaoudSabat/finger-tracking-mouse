"""MediaPipe-based hand landmark detector."""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """Detects hand landmarks in a BGR frame using MediaPipe Hands."""

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
        )

    def process(self, frame: np.ndarray) -> list[dict]:
        """Return list of landmark dicts for each detected hand.

        Each dict has keys: 'landmarks' (list of (x, y) normalised),
        'index_tip', 'thumb_tip' as normalised (x, y) tuples.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        hands = []
        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(frame, hand_lm, self._mp_hands.HAND_CONNECTIONS)
                lm = hand_lm.landmark
                hands.append({
                    "landmarks": [(l.x, l.y) for l in lm],
                    "index_tip": (lm[8].x, lm[8].y),
                    "thumb_tip": (lm[4].x, lm[4].y),
                })
        return hands

    def close(self) -> None:
        self._hands.close()
