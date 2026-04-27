"""
Module for real time face detection using mediapipe
"""

import time
import cv2
import mediapipe as mp

mp.tasks

mpFaceDetection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap = cv2
    # tracking = HandTracking()

    while True:
        success, img = cap.read()
        # img = tracking.find_hands(img=img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img=img,
                    text=f"{str(int(fps))} FPS",
                    org=(10,70),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=3,
                    color=(0, 255, 0),
                    thickness=4)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
