import cv2 as cv
import time
from pose_detection import PoseDetector

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = PoseDetector()
    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame from webcam.")
            break

        flipped_frame = cv.flip(frame, 1)
        frame_with_pose = detector.find_pose(frame, flipped_frame, draw=True)
        landmarks = detector.find_positions(frame_with_pose)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        cv.putText(frame_with_pose, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv.imshow("Partner Pose Detection", frame_with_pose)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
