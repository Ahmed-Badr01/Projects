import mediapipe as mp
import keyboard
import cv2


class HandDetector:

    def __init__(self, static_mode=False, num_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.results = None
        self.mode = static_mode
        self.max_hands = num_hands
        self.min_detect_conf = detection_confidence
        self.min_track_conf = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                         min_detection_confidence=self.min_detect_conf,
                                         min_tracking_confidence=self.min_track_conf)

    def detect_hands(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(frame_rgb)

        hands_landmarks = self.results.multi_hand_landmarks

        if hands_landmarks:
            for hand in hands_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

        return frame

    def find_positions(self, frame, hand_num=0, show_all=True):

        landmarks_list = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]

            for idx, landmark in enumerate(hand.landmark):  # I don't understand how we accessed this list.
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                landmarks_list.append([idx, x, y])
                if show_all:
                    cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
                    cv2.putText(frame, str(idx), (x - 5, y + 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        return landmarks_list

    @staticmethod
    def find_open_fingers(landmarks_list, right_hand=True):
        finger_tips = [8, 12, 16, 20]
        fingers_up = [0, 0, 0, 0, 0]

        for idx, tip in enumerate(finger_tips):
            if landmarks_list[tip][2] > landmarks_list[tip - 2][2]:
                fingers_up[idx + 1] = 0
            else:
                fingers_up[idx + 1] = 1
        if right_hand:
            if landmarks_list[4][1] > landmarks_list[2][1]:
                fingers_up[0] = 1
            else:
                fingers_up[0] = 0
        else:
            if landmarks_list[4][1] < landmarks_list[2][1]:
                fingers_up[0] = 1
            else:
                fingers_up[0] = 0
        return fingers_up


def main():
    vid = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, frame = vid.read()
        frame = detector.detect_hands(frame)
        lm_list = detector.find_positions(frame)
        if lm_list:
            print(lm_list[2])

        cv2.imshow('cam feed', frame)
        cv2.waitKey(1)

        if keyboard.is_pressed('q'):
            break
