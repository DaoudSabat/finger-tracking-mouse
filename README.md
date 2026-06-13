# 🖐️ Finger Tracking Mouse

Control your computer mouse using only your index finger and a standard webcam — no extra hardware needed.

## Overview

The system uses MediaPipe to detect hand landmarks in real time from the webcam feed. The tip of the index finger (landmark 8) is mapped to screen coordinates to move the cursor. A pinch gesture (index tip close to thumb tip) triggers a mouse click or drag, distinguished by hold duration.

## Architecture

```
finger-tracking-mouse/
├── core/
│   ├── hand_tracker.py       # HandTracker — MediaPipe Hands wrapper
│   └── mouse_controller.py   # MouseController — gesture → mouse event mapping
├── utils/                    # Reserved for future helpers
├── tests/
│   └── test_gesture_detection.py  # pytest unit tests (no webcam needed)
├── main.py                   # Entry point
└── requirements.txt
```

## Design Patterns

- **Single Responsibility** — `HandTracker` handles only landmark detection; `MouseController` handles only mouse events. Neither depends on the other's internals.
- **Facade** — `main.py` wires the two classes into a clean capture loop without exposing MediaPipe or pyautogui details to the outside.

## Tech Stack

- **Python 3.10+**
- **OpenCV 4.8** — webcam capture and frame display
- **MediaPipe** — real-time hand landmark detection (21 landmarks per hand)
- **pyautogui** — cross-platform mouse control
- **pytest** — unit testing

## Installation

```bash
git clone https://github.com/DaoudSabat/finger-tracking-mouse.git
cd finger-tracking-mouse
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Point your index finger at the screen to move the cursor. Pinch thumb + index finger to click. Hold the pinch for a drag. Press **`e`** to exit.

## Tests

```bash
pytest tests/ -v
```

All tests mock the webcam, MediaPipe, and screen — no hardware required.

## License

MIT
