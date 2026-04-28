"""
Module for real time face detection using OpenCV HaarCascade
"""

import numpy as np
import time
import cv2


def rescale_frame(frame, scale=0.75):
    """
    Resize the image
    """
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)
    dimensions = (width, heigth)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def detect_face(colored_img, scale_factor=1.1, min_neighbors=5):
    """
    Function that use OpenCV's haar cascade algorithm to draw bounding boxes around faces
    """
    img_copy = np.copy(colored_img)

    # convert colored image to gray image as expected by opencv face detector
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Train haar cascade on xml file
    haar_cascade = cv2.CascadeClassifier('haar_faces.xml')

    # detect multiscale (some images may be closer to camera than others) images
    faces_boxes = haar_cascade.detectMultiScale(image=gray,
                                                scaleFactor=scale_factor,
                                                minNeighbors=min_neighbors)

    # Go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces_boxes:
        cv2.rectangle(img=img_copy,
                      pt1=(x, y),
                      pt2=(x + w, y + h),
                      color=(0, 255, 0),
                      thickness=1)

    return img_copy


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('../data/videos/video_happy_01.mp4') # Local video version

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1040)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while(cap.isOpened()):
        success, frame = cap.read()

        frame = rescale_frame(frame=frame, scale=0.4)

        if success:
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

            frame = detect_face(frame, min_neighbors=8)

            cv2.imshow("Image", frame)


        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
