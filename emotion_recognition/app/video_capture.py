import cv2
import threading
from emotion_recognition.params import CAP_RESOLUTION, CAP_FPS


class VideoStream():
    """
    Captures webcam frames in a background thread so the main loop
    never blocks waiting for the camera.

    Exemple:
        stream = VideoStream(camera_idx=0)
        success, img = stream.read()
        stream.stop()
    """
    def __init__(self, camera_idx: int = 0):
        self.cap = cv2.VideoCapture(camera_idx)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_RESOLUTION[1])
        self.cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self) -> None:
        """
        Runs in background and continuously reads the latest frame.
        """
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self) -> tuple:
        """
        Returns the latest frame instantly.

        Returns
        ----
        ret : bool
            - False if the camera is unavailable.
        frame : np.ndarray
            - Latest BGR frame from the camera.
        """
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def stop(self) -> None:
        """
        Stops the background thread and releases the camera.
        """
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()
