import cv2
import pyautogui
from time import time
import mediapipe as mp

class myPose:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose_video = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def detectPose(self, image, pose, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks and draw:
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return image, results

    def checkPose_LRC(self, image, results, draw=True):
        # Use mid-hip x coordinate to decide one of four grid positions: Left, Center-Left, Center-Right, Right
        landmarks = results.pose_landmarks.landmark
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        mid_x = (left_hip.x + right_hip.x) / 2

        if mid_x < 0.25:
            pos = "Left"
        elif 0.25 <= mid_x < 0.5:
            pos = "Center-Left"
        elif 0.5 <= mid_x < 0.75:
            pos = "Center-Right"
        else:
            pos = "Right"

        if draw:
            cv2.putText(image, f"Horizontal: {pos}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        return image, pos

    def checkPose_JSD(self, image, results, MID_Y, draw=True):
        # Vertical posture based on midpoint shoulders y coordinate vs MID_Y calibration
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        diff = mid_shoulder_y - MID_Y
        posture = "Standing"
        if diff < -0.05:
            posture = "Jumping"
        elif diff > 0.05:
            posture = "Crouching"

        if draw:
            cv2.putText(image, f"Vertical: {posture}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        return image, posture


class myGame():
    def __init__(self):
        self.pose = myPose()
        self.game_started = True  # Start game immediately (hand join removed)
        self.x_pos_index = 1  # 0=Left,1=Center-Left,2=Center-Right,3=Right
        self.y_pos_index = 1  # 0=Crouch,1=Standing,2=Jump
        self.MID_Y = None
        self.time1 = 0

    def move_LRC(self, LRC):
        # Map grid positions to indices for movement
        target_index = {"Left": 0, "Center-Left": 1, "Center-Right": 2, "Right": 3}[LRC]

        if target_index > self.x_pos_index:
            for _ in range(target_index - self.x_pos_index):
                pyautogui.press('right')
            self.x_pos_index = target_index
        elif target_index < self.x_pos_index:
            for _ in range(self.x_pos_index - target_index):
                pyautogui.press('left')
            self.x_pos_index = target_index

    def move_JSD(self, JSD):
        # Jump
        if JSD == 'Jumping' and self.y_pos_index != 2:
            pyautogui.press('up')
            self.y_pos_index = 2
        # Crouch
        elif JSD == 'Crouching' and self.y_pos_index != 0:
            pyautogui.press('down')
            self.y_pos_index = 0
        # Standing (reset)
        elif JSD == 'Standing' and self.y_pos_index != 1:
            self.y_pos_index = 1

    def calibrate_MID_Y(self, results):
        # Calibrate MID_Y once at start based on shoulder midpoint y
        left_sh = results.pose_landmarks.landmark[self.pose.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_sh = results.pose_landmarks.landmark[self.pose.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        self.MID_Y = (left_sh.y + right_sh.y) / 2

    def play(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 960)

        cv2.namedWindow('Pose Game', cv2.WINDOW_NORMAL)

        calibrated = False

        while True:
            ret, image = cap.read()
            if not ret:
                continue

            image = cv2.flip(image, 1)
            image_height, image_width, _ = image.shape

            image, results = self.pose.detectPose(image, self.pose.pose_video, draw=self.game_started)

            if results.pose_landmarks:
                if not calibrated:
                    self.calibrate_MID_Y(results)
                    calibrated = True
                    print("MID_Y calibrated:", self.MID_Y)

                if self.game_started and calibrated:
                    # Horizontal grid move
                    image, LRC = self.pose.checkPose_LRC(image, results, draw=True)
                    self.move_LRC(LRC)

                    # Vertical jump/crouch move
                    image, JSD = self.pose.checkPose_JSD(image, results, self.MID_Y, draw=True)
                    self.move_JSD(JSD)

            # FPS display
            time2 = time()
            if (time2 - self.time1) > 0:
                fps = 1.0 / (time2 - self.time1)
                cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            self.time1 = time2

            cv2.imshow("Pose Game", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    game = myGame()
    game.play()

