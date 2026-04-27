"""
Module for real time face detection
"""

import threading
import time
import cv2

def detect_face():
    pass

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)#, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

    counter = 0

    face_detection = False


    while True:
        success, frame = cap.read()

        if success:
            pass

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime

            cv2.putText(img=frame,
                        text=f"{str(int(fps))} FPS",
                        org=(10,70),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=3,
                        color=(200, 250, 100),
                        thickness=4)

            cv2.imshow("Image", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
