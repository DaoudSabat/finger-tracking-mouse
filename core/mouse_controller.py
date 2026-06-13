"""Translates hand gestures to mouse actions via pyautogui."""
from __future__ import annotations

import time

import pyautogui


class MouseController:
    """Maps index-finger position and pinch gesture to mouse events."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        pinch_threshold: float = 0.1,
        hold_threshold: float = 0.59,
    ) -> None:
        self._sw = screen_width
        self._sh = screen_height
        self._pinch_threshold = pinch_threshold
        self._hold_threshold = hold_threshold
        self._pressed = False
        self._pinch_start: float = 0.0

    def update(self, index_tip: tuple[float, float], thumb_tip: tuple[float, float]) -> None:
        """Move cursor and fire click/hold events based on finger positions."""
        ix, iy = int(index_tip[0] * self._sw), int(index_tip[1] * self._sh)
        pyautogui.moveTo(ix, iy)

        distance = (
            (index_tip[0] - thumb_tip[0]) ** 2 + (index_tip[1] - thumb_tip[1]) ** 2
        ) ** 0.5

        if distance < self._pinch_threshold:
            if not self._pressed:
                self._pinch_start = time.time()
                self._pressed = True
                pyautogui.mouseDown()
        else:
            if self._pressed:
                duration = time.time() - self._pinch_start
                if duration < self._hold_threshold:
                    pyautogui.mouseUp()
                    pyautogui.click()
                else:
                    pyautogui.mouseUp()
                self._pressed = False

    @staticmethod
    def pinch_distance(index_tip: tuple[float, float], thumb_tip: tuple[float, float]) -> float:
        """Euclidean distance between two normalised 2D points."""
        return ((index_tip[0] - thumb_tip[0]) ** 2 + (index_tip[1] - thumb_tip[1]) ** 2) ** 0.5
