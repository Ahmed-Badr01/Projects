from HandTrackingModule import HandDetector
import numpy as np
import keyboard
import math
import cv2

# Open a cam window and adjust its dimensions:

cam = cv2.VideoCapture(0)
cam.set(3, 1280)  # Width
cam.set(4, 720)  # Height

detector = HandDetector(num_hands=1, detection_confidence=0.65)
# Hand tracker which requires 80% confidence (default: 50%) and tracks only one hand (default: 2).
# We only need 1 hand for the game. 2 could confuse it.


class Snake:
    def __init__(self):
        self.points = []  # Current coordinates of the points of the snake.
        self.links = []  # The lines connecting the points.
        self.current_length = sum(self.links)  # Current total length of the snake.
        self.max_length = 10_000  # Max allowed length.
        self.previous_head = (0, 0)
        self.score = 0
        self.collisions = 0
        self.game_over = False
        self.food_location = self.generate_random_location()

    @staticmethod
    def generate_random_location():
        return np.random.randint(low=130, high=1280 - 130), np.random.randint(low=130, high=720 - 130)
        # This generates a random location for food on the cam feed, but avoids crossing the borders.

    def update(self, image, current_head):

        if not self.game_over:
            old_x, old_y = self.previous_head
            x, y = current_head
            self.points.append([x, y])
            link = math.hypot(x - old_x, y - old_y)
            self.links.append(link)
            self.current_length += link

            # Check if snake ate the food:
            food_x, food_y = self.food_location
            if food_x - 25 < x < food_x + 25 and food_y - 25 < y < food_y + 25:
                self.food_location = self.generate_random_location()
                self.max_length *= 1.12
                self.score += 1

            # Draw snake:
            for i, point in enumerate(self.points):
                if i:  # Excludes the first point (number 0).
                    cv2.line(image, self.points[i-1], self.points[i], (0, 0, 0), 20)
            cv2.circle(image, current_head, 20, (200, 0, 200), -1)

            # Control length:
            if self.current_length >= self.max_length:
                for i in range(len(self.points)):
                    self.current_length -= self.links[i]
                    self.links.pop(i)
                    self.points.pop(i)

                    if self.current_length <= self.max_length:
                        break

            # Draw food:
            image = cv2.circle(image, self.food_location, 25, (0, 0, 0), -1)
            image = cv2.circle(image, self.food_location, 18, (255, 255, 255), -1)
            image = cv2.circle(image, self.food_location, 11, (0, 0, 0), -1)
            image = cv2.circle(image, self.food_location, 4, (255, 255, 255), -1)

            # Draw score:
            cv2.rectangle(image, (50, 50), (330, 170), (150, 0, 150), 5)
            cv2.putText(image, f"Score = {self.score}", (75, 120), cv2.FONT_HERSHEY_PLAIN, 2.5, (150, 0, 150), 4)

            # Check for collisions:
            # To check for collisions, we can use OpenCV's polylines and pointPolygonTest functions.

            points = np.array(self.points[:-4], dtype=np.int32)
            points = points.reshape((-1, 1, 2))  # For compatibility with the polygon function. (?)
            cv2.polylines(image, [points], False, (255, 255, 255), 3)  # Must take a list of points for some reason.
            # Also: isClosed means: is the shape meant to be closed? Here: False, because we're dealing with a snake.
            minimum_distance = cv2.pointPolygonTest(points, (x, y), True)  # Returns the minimum distance between the
            # current head (x, y) and the rest of the points it was given.

            if abs(minimum_distance) <= 2:
                self.collisions += 1
            if self.collisions >= 10:
                self.game_over = True

            # Draw score:
            cv2.rectangle(image, (800, 50), (1230, 160), (0, 0, 0), 5)
            cv2.putText(image, f"Watch out for collisions!", (825, 100),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 4)
            cv2.putText(image, f"Allowed collisions remaining: {10 - self.collisions}", (825, 120),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 4)

        else:
            cv2.putText(image, "Game Over!", (280, 260),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 5, (0, 0, 0), 5)
            cv2.putText(image, f"Your score is: {self.score}.", (425, 360),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 3)
            cv2.putText(image, "To start over, press \"s\".", (330, 460),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 3)
            cv2.putText(image, "To quit, press \"q\".", (400, 560),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 3)
            self.points = []
            self.links = []
            self.current_length = sum(self.links)
            self.max_length = 10_000
            self.previous_head = (0, 0)
            if keyboard.is_pressed('s'):
                self.score = 0
                self.collisions = 0
                self.game_over = False
        return image


snake = Snake()


while True:
    success, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Mirrors the image to make playing the game easier.

    img_w_hands = detector.detect_hands(frame)
    landmarks_list = detector.find_positions(img_w_hands, show_all=False)
    # We'd like to locate the tip of the index and draw a circle around it for the game:
    if landmarks_list:  # Only execute if a hand is detected in the frame.
        index_tip_coordinates = landmarks_list[8][1:]  # 8 is the tip of the index, 0:2 returns only x & y coordinates.

        img_w_hands = snake.update(img_w_hands, index_tip_coordinates)
        # Important: Shadow the same name for this variable as the outside variable, as a new variable would
        # depend on this if statement being true, so if no hands are immediately detected in the cam feed,
        # imshow would have nothing to read.

    cv2.imshow("Snake Game", img_w_hands)
    cv2.waitKey(40)

    if keyboard.is_pressed('q'):
        break
