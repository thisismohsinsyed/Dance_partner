import cv2 as cv
import mediapipe as mp
import numpy as np

class PoseDetector:
    """
    A wrapper for MediaPipe's Pose model for pose estimation using OpenCV.
    """

    def __init__(self, static_mode=False, smooth_landmarks=True, detection_confidence=0.7, tracking_confidence=0.5):
        self.static_mode = static_mode
        self.smooth_landmarks = smooth_landmarks
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_mode,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.results = None

    def find_pose(self, draw_image, process_image, draw=True):
        """
        Detects human pose in the given image using MediaPipe.

        Args:
            draw_image (np.ndarray): The image to draw landmarks on.
            process_image (np.ndarray): The image for processing (e.g., flipped).
            draw (bool): Whether to draw landmarks.

        Returns:
            np.ndarray: Image with pose landmarks drawn.
        """
        img_rgb = cv.cvtColor(process_image, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            landmark_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=0)
            connection_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
            self.mp_draw.draw_landmarks(
                draw_image, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec
            )
        return draw_image

    def find_positions(self, img):
        """
        Finds pose landmark positions in the processed results.

        Args:
            img (np.ndarray): The image to map landmark coordinates.

        Returns:
            list of tuple: List of (x, y) landmark coordinates.
        """
        lmlist = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for lm in self.results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append((cx, cy))
        return lmlist

    @staticmethod
    def create_pose_mask(img, landmarks, radius=30):
        """
        Creates a binary mask with circles at landmark locations.

        Args:
            img (np.ndarray): Reference image for shape.
            landmarks (list): List of (x, y) coordinates.
            radius (int): Radius for each landmark mask.

        Returns:
            np.ndarray: Binary mask.
        """
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for (x, y) in landmarks:
            cv.circle(mask, (x, y), radius, 255, -1)
        return mask