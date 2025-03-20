import mediapipe as mp
import cv2
from math import hypot


class myPose():
    def __init__(self):
        # self.mp_pose = mp.solutions.pose
        # self.pose = self.mp_pose.Pose()
        # self.mp_drawing = mp.solutions.drawing_utils
        # self.shoudler_line_y = 0  # Luu lai cai vi tri 2 vai cua nguoi dung khi vo tay bat dau game

        # Initialize mediapipe pose class.
        self.mp_pose = mp.solutions.pose

        # Setup the Pose function for images.
        self.pose_image = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

        # Setup the Pose function for videos.
        self.pose_video = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                                            min_tracking_confidence=0.7)

        # Initialize mediapipe drawing class.
        self.mp_drawing = mp.solutions.drawing_utils

    def detectPose(self, image, pose, draw=False, display=False):
        """
        This function performs the pose detection on the most prominent person in an image.
        Args:
            image: The input image with a prominent person whose pose landmarks needs to be detected.
            pose: The pose function required to perform the pose detection.
            draw: A boolean value that is if set to true the function draw pose landmarks on the output image.
            display: A boolean value that is if set to true the function displays the original input image, and the
                 resultant image and returns nothing.

        Returns:
            output_image: The input image with the detected pose landmarks drawn if it was specified.
            results:      The output of the pose landmarks detection on the input image.

        """

        # Create a copy of the input image.
        output_image = image.copy()

        # Convert the image from BGR into RGB format.
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform the Pose Detection.
        results = pose.process(imageRGB)

        # Check if any landmarks are detected and are specified to be drawn.
        if results.pose_landmarks and draw:
            # Draw Pose Landmarks on the output image.
            self.mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                      connections=self.mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                   thickness=3, circle_radius=3),
                                      connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(49, 125, 237),
                                                                                     thickness=2, circle_radius=2))

        # Check if the original input image and the resultant image are specified to be displayed.
        if display:
            # Display the original input image and the resultant image.
            plt.figure(figsize=[22, 22])
            plt.subplot(121);
            plt.imshow(image[:, :, ::-1]);
            plt.title("Original Image");
            plt.axis('off');
            plt.subplot(122);
            plt.imshow(output_image[:, :, ::-1]);
            plt.title("Output Image");
            plt.axis('off');

        # Otherwise
        else:
            # Return the output image and the results of pose landmarks detection.
            return output_image, results

    def checkPose_LRC(self, image, results, draw=False, display=False):
        """
        This function finds the horizontal position (left, center, right) of the person in an image.
        Args:
            image:   The input image with a prominent person whose the horizontal position needs to be found.
            results: The output of the pose landmarks detection on the input image.
            draw:    A boolean value that is if set to true the function writes the horizontal position on the output image.
            display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
        Returns:
            output_image:         The same input image but with the horizontal position written, if it was specified.
            horizontal_position:  The horizontal position (left, center, right) of the person in the input image.

        """

        # Declare a variable to store the horizontal position (left, center, right) of the person.
        horizontal_position = None

        # Get the height and width of the image.
        height, width, _ = image.shape

        # Create a copy of the input image to write the horizontal position on.
        output_image = image.copy()

        # Retreive the x-coordinate of the left shoulder landmark.
        left_x = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)

        # Retreive the x-corrdinate of the right shoulder landmark.
        right_x = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)

        # Check if the person is at left that is when both shoulder landmarks x-corrdinates
        # are less than or equal to the x-corrdinate of the center of the image.
        if (right_x <= width // 2 and left_x <= width // 2):

            # Set the person's position to left.
            horizontal_position = 'Left'

        # Check if the person is at right that is when both shoulder landmarks x-corrdinates
        # are greater than or equal to the x-corrdinate of the center of the image.
        elif (right_x >= width // 2 and left_x >= width // 2):

            # Set the person's position to right.
            horizontal_position = 'Right'

        # Check if the person is at center that is when right shoulder landmark x-corrdinate is greater than or equal to
        # and left shoulder landmark x-corrdinate is less than or equal to the x-corrdinate of the center of the image.
        elif (right_x >= width // 2 and left_x <= width // 2):

            # Set the person's position to center.
            horizontal_position = 'Center'

        # Check if the person's horizontal position and a line at the center of the image is specified to be drawn.
        if draw:
            # Write the horizontal position of the person on the image.
            cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),
                        3)

            # Draw a line at the center of the image.
            cv2.line(output_image, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

        # Check if the output image is specified to be displayed.
        if display:

            # Display the output image.
            plt.figure(figsize=[10, 10])
            plt.imshow(output_image[:, :, ::-1]);
            plt.title("Output Image");
            plt.axis('off');

        # Otherwise
        else:

            # Return the output image and the person's horizontal position.
            return output_image, horizontal_position

    def checkPose_JSD(self, image, results, MID_Y=250, draw=False, display=False):
        """
        This function checks the posture (Jumping, Crouching or Standing) of the person in an image.
        Args:
            image:   The input image with a prominent person whose the posture needs to be checked.
            results: The output of the pose landmarks detection on the input image.
            MID_Y:   The intial center y-coordinate of both shoulders landmarks of the person recorded during starting
                 the game. This will give the idea of the person's height when he is standing straight.
            draw:    A boolean value that is if set to true the function writes the posture on the output image.
            display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
        Returns:
            output_image: The input image with the person's posture written, if it was specified.
            posture:      The posture (Jumping, Crouching or Standing) of the person in an image.

        """
        # Get the height and width of the image.
        height, width, _ = image.shape

        # Create a copy of the input image to write the posture label on.
        output_image = image.copy()

        # Retreive the y-coordinate of the left shoulder landmark.
        left_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)

        # Retreive the y-coordinate of the right shoulder landmark.
        right_y = int(results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

        # Calculate the y-coordinate of the mid-point of both shoulders.
        actual_mid_y = abs(right_y + left_y) // 2

        # Calculate the upper and lower bounds of the threshold.
        lower_bound = MID_Y - 15
        upper_bound = MID_Y + 100

        # Check if the person has jumped that is when the y-coordinate of the mid-point
        # of both shoulders is less than the lower bound.
        if (actual_mid_y < lower_bound):

            # Set the posture to jumping.
            posture = 'Jumping'

        # Check if the person has crouched that is when the y-coordinate of the mid-point
        # of both shoulders is greater than the upper bound.
        elif (actual_mid_y > upper_bound):

            # Set the posture to crouching.
            posture = 'Crouching'

        # Otherwise the person is standing and the y-coordinate of the mid-point
        # of both shoulders is between the upper and lower bounds.
        else:

            # Set the posture to Standing straight.
            posture = 'Standing'

        # Check if the posture and a horizontal line at the threshold is specified to be drawn.
        if draw:
            # Write the posture of the person on the image.
            cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

            # Draw a line at the intial center y-coordinate of the person (threshold).
            cv2.line(output_image, (0, MID_Y), (width, MID_Y), (255, 255, 255), 2)

        # Check if the output image is specified to be displayed.
        if display:

            # Display the output image.
            plt.figure(figsize=[10, 10])
            plt.imshow(output_image[:, :, ::-1]);
            plt.title("Output Image");
            plt.axis('off');

        # Otherwise
        else:

            # Return the output image and posture indicating whether the person is standing straight or has jumped, or crouched.
            return output_image, posture

    def checkHandsJoined(self, image, results, draw=False, display=False):
        """
        This function checks whether the hands of the person are joined or not in an image.
        Args:
            image:   The input image with a prominent person whose hands status (joined or not) needs to be classified.
            results: The output of the pose landmarks detection on the input image.
            draw:    A boolean value that is if set to true the function writes the hands status &amp; distance on the output image.
            display: A boolean value that is if set to true the function displays the resultant image and returns nothing.

        Returns:
            output_image: The same input image but with the classified hands status written, if it was specified.
            hand_status:  The classified status of the hands whether they are joined or not.

        """

        # Get the height and width of the input image.
        height, width, _ = image.shape

        # Create a copy of the input image to write the hands status label on.
        output_image = image.copy()

        # Get the left wrist landmark x and y coordinates.
        left_wrist_landmark = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                               results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y * height)

        # Get the right wrist landmark x and y coordinates.
        right_wrist_landmark = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                                results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y * height)

        # Calculate the euclidean distance between the left and right wrist.
        euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                       left_wrist_landmark[1] - right_wrist_landmark[1]))

        # Compare the distance between the wrists with a appropriate threshold to check if both hands are joined.
        if euclidean_distance < 100:

            # Set the hands status to joined.
            hand_status = 'Hands Joined'

            # Set the color value to green.
            color = (0, 255, 0)

        # Otherwise.
        else:

            # Set the hands status to not joined.
            hand_status = 'Hands Not Joined'

            # Set the color value to red.
            color = (0, 0, 255)

        # Check if the Hands Joined status and hands distance are specified to be written on the output image.
        if draw:
            # Write the classified hands status on the image.
            cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

            # Write the the distance between the wrists on the image.
            cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

        # Check if the output image is specified to be displayed.
        if display:

            # Display the output image.
            plt.figure(figsize=[10, 10])
            plt.imshow(output_image[:, :, ::-1]);
            plt.title("Output Image");
            plt.axis('off');

        # Otherwise
        else:

            # Return the output image and the classified hands status indicating whether the hands are joined or not.
            return output_image, hand_status

