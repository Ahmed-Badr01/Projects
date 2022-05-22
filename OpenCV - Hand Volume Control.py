from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from HandTrackingModule import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import keyboard
import cv2

# Start a cam feed and adjust its window dimensions:
vid = cv2.VideoCapture(0)
vid.set(3, 1280)
vid.set(4, 720)

detector = HandDetector()

# Instantiate the necessary code to manipulate system volume as per the pycaw library documentation:
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume.SetMasterVolumeLevel(-13.7, None)

# levels = [0, 20, 40, 60, 80, 100]
level_values = [-65.25, -23.7, -13.7, -7.7, -3.3, 0]  # These values were found through experimentation.

while True:
    success, frame = vid.read()

    frame = detector.detect_hands(frame)
    landmarks_list = detector.find_positions(frame, show_all=False)  # Gets a list of landmarks' positions on the hand.

    if landmarks_list:
        fingers_up = detector.find_open_fingers(landmarks_list, right_hand=True)
        fingers_up = sum(fingers_up)
        # We'll use the number of open fingers in our right hand to adjust the volume, increasing it by 20% per finger.
        volume.SetMasterVolumeLevel(level_values[fingers_up], None)

        cv2.rectangle(frame, (100, 150), (150, 550), (0, 255, 0), 3)
        cv2.rectangle(frame, (100, 550 - fingers_up * 80), (150, 550), (0, 255, 0), -1)
        cv2.putText(frame, f'Volume: {fingers_up * 20}%', (70, 600), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow('cam feed', frame)
    cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        break
