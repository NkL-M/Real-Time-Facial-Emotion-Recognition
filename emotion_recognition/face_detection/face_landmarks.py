"""
Module for real time facial landmarks detection using mediapipe
"""

import time
import cv2
import mediapipe as mp
from pathlib import Path
from emotion_recognition.params import *

class FacialTracker():
    def __init__(self,
                 static_mode=False,
                 max_faces=2,
                 detect_conf=0.5,
                 track_conf=0.5):

        self.static_mode=static_mode
        self.max_faces=max_faces
        self.detect_conf=detect_conf
        self.track_conf=track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.static_mode,
                                                 max_num_faces=self.max_faces,
                                                 min_detection_confidence=self.detect_conf,
                                                 min_tracking_confidence=self.track_conf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,
                                                color=(255, 0, 255),
                                                circle_radius=1)
        # self.landmark_spec = self.mpDraw.DrawingSpec(color=(255, 0, 0), circle_radius=1)
        # self.connection_spec = self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=1)


    def find_face_mesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,
                                            face_lms,
                                            self.mpFaceMesh.FACEMESH_CONTOURS,
                                            self.drawSpec,
                                            self.drawSpec)

                    # self.mpDraw.draw_landmarks(img,
                    #                            face_lms,
                    #                            self.mpFaceMesh.FACEMESH_TESSELATION,
                    #                            self.landmark_spec,   # dots
                    #                            self.connection_spec) # lines

                # for id, lm in enumerate(face_lms.landmark):
                #     h, w, c = img.shape
                #     x, y = int(lm.x * w), int(lm.y * h)

        return img

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(str(DATA_DIR/'videos'/'video_happy_01.mp4'))
    # cap = cv2.ImageCapture(str(DATA_DIR/'train'/'angry'/'Training_3908.jpg'))
    tracker = FacialTracker()

    while True:
        success, img = cap.read()
        img = tracker.find_face_mesh(img, draw=True)

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

        key = cv2.waitKey(30)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
