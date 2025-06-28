import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
prev_click_time = 0
prev_x, prev_y = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        index_tip = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        x = int(index_tip.x * img.shape[1])
        y = int(index_tip.y * img.shape[0])

        screen_x = np.interp(x, [0, img.shape[1]], [0, screen_width])
        screen_y = np.interp(y, [0, img.shape[0]], [0, screen_height])

        # Smoothing movement
        curr_x = prev_x + (screen_x - prev_x) / 5
        curr_y = prev_y + (screen_y - prev_y) / 5
        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Double tap detection for click
        distance = np.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
        if distance < 0.02:
            curr_time = time.time()
            if curr_time - prev_click_time > 0.5:
                pyautogui.click()
                prev_click_time = curr_time

        mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Mouse', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
