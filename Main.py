import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Capture video from the laptop camera
cap = cv2.VideoCapture(0)

# Screen width and height
screen_width, screen_height = pyautogui.size()

# Variables to track mouse state
mouse_pressed = False
pinch_start_time = 0

# Time threshold to distinguish between click and hold (in seconds)
hold_threshold = 0.5

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with Mediapipe
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the position of the index finger (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            
            # Convert to screen coordinates
            finger_x = int(index_finger_tip.x * screen_width)
            finger_y = int(index_finger_tip.y * screen_height)
            
            # Move the mouse cursor
            pyautogui.moveTo(finger_x, finger_y)
            
            # Simulate a mouse click hold and release based on finger distance
            thumb_tip = hand_landmarks.landmark[4]
            distance = ((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
            
            if distance < 0.07:
                if not mouse_pressed:
                    pinch_start_time = time.time()
                    mouse_pressed = True
                    pyautogui.mouseDown()
            else:
                if mouse_pressed:
                    pinch_duration = time.time() - pinch_start_time
                    if pinch_duration < hold_threshold:
                        pyautogui.mouseUp()
                        pyautogui.click()
                    else:
                        pyautogui.mouseUp()
                    mouse_pressed = False
    
    # Display the frame
    cv2.imshow('Finger Tracking', frame)
    
    # Break the loop when 'e' is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
