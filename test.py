import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize Mediapipe hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Track mouse state
mouse_pressed = False
pinch_start_time = 0
hold_threshold = 0.6
move_threshold = 5
right_click_done = False

last_mouse_x, last_mouse_y = 0, 0
pinch_stable_counter = 0
PINCH_STABLE_FRAMES = 3


def get_screen_coords(landmark):
    return int(landmark.x * screen_width), int(landmark.y * screen_height)

def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]

        dist_index_thumb = calculate_distance(index_tip, thumb_tip)
        dist_index_middle = calculate_distance(index_tip, middle_tip)

        mouse_x, mouse_y = get_screen_coords(index_tip)

        if abs(mouse_x - last_mouse_x) > move_threshold or abs(mouse_y - last_mouse_y) > move_threshold:
            pyautogui.moveTo(mouse_x, mouse_y)
            last_mouse_x, last_mouse_y = mouse_x, mouse_y

        if dist_index_thumb < 0.06:
            pinch_stable_counter += 1
            if pinch_stable_counter >= PINCH_STABLE_FRAMES:
                if not mouse_pressed:
                    pinch_start_time = time.time()
                    mouse_pressed = True
                    pyautogui.mouseDown()
        else:
            pinch_stable_counter = 0
            if mouse_pressed:
                duration = time.time() - pinch_start_time
                pyautogui.mouseUp()
                if duration < hold_threshold:
                    pyautogui.click()
                mouse_pressed = False

        # Right click detection using index and middle finger pinch
        if dist_index_middle < 0.05:
            if not right_click_done:
                pyautogui.rightClick()
                right_click_done = True
        else:
            right_click_done = False

    else:
        pinch_stable_counter = 0
        if mouse_pressed:
            pyautogui.mouseUp()
            mouse_pressed = False
        right_click_done = False

    cv2.imshow('Finger Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
