"""Unit tests for finger-tracking-mouse — no webcam or screen required."""
import pytest
from unittest.mock import MagicMock, patch

from core.mouse_controller import MouseController


# ---------- MouseController tests ----------

def test_pinch_distance_close():
    d = MouseController.pinch_distance((0.1, 0.1), (0.15, 0.15))
    assert d < 0.1


def test_pinch_distance_far():
    d = MouseController.pinch_distance((0.0, 0.0), (0.5, 0.5))
    assert d > 0.1


def test_pinch_distance_symmetric():
    d1 = MouseController.pinch_distance((0.2, 0.3), (0.5, 0.8))
    d2 = MouseController.pinch_distance((0.5, 0.8), (0.2, 0.3))
    assert abs(d1 - d2) < 1e-9


def test_controller_initial_state():
    ctrl = MouseController(1920, 1080)
    assert ctrl._pressed is False


@patch("core.mouse_controller.pyautogui")
def test_update_moves_mouse(mock_pg):
    ctrl = MouseController(1920, 1080)
    ctrl.update((0.5, 0.5), (0.9, 0.9))
    mock_pg.moveTo.assert_called_once_with(960, 540)


@patch("core.mouse_controller.pyautogui")
@patch("core.mouse_controller.time")
def test_short_pinch_triggers_click(mock_time, mock_pg):
    mock_time.time.side_effect = [0.0, 0.3]
    ctrl = MouseController(1920, 1080, pinch_threshold=0.1, hold_threshold=0.59)
    ctrl.update((0.5, 0.5), (0.52, 0.52))
    ctrl.update((0.5, 0.5), (0.9, 0.9))
    mock_pg.click.assert_called_once()
