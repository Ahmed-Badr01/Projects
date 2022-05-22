from HandTrackingModule import HandDetector
import numpy as np
import keyboard
import pyautogui
import time
import cv2

vid = cv2.VideoCapture(0)
cam_width, cam_height = 1280, 720
vid.set(3, cam_width)
vid.set(4, cam_height)

detector = HandDetector(num_hands=1)

old_x, old_y = 0, 0

while True:

    success, frame = vid.read()
    frame = detector.detect_hands(frame)
    frame = cv2.flip(frame, 1)
    landmarks_list = detector.find_positions(frame, show_all=False)
    if landmarks_list:
        open_fingers = detector.find_open_fingers(landmarks_list, right_hand=True)
        interpreted_x = np.interp(cam_width - landmarks_list[8][1], [150, cam_width - 150], [0, 1920])
        interpreted_y = np.interp(landmarks_list[8][2], [150, cam_height - 150], [0, 1280])
        if open_fingers[1] and not open_fingers[4]:
            pyautogui.FAILSAFE = False
            smoothed_x = old_x + int((interpreted_x - old_x) / 4 )
            smoothed_y = old_y + int((interpreted_y - old_y) / 4)
            pyautogui.moveTo(smoothed_x, smoothed_y)
        elif open_fingers[1] and open_fingers[4]:
            pyautogui.leftClick()
            time.sleep(0.2)
        old_x = interpreted_x
        old_y = interpreted_y
    cv2.rectangle(frame, (70, 70), (cam_width - 150, cam_height - 150), (255, 0, 0), 3)

    cv2.imshow('cam feed', frame)
    cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        break
