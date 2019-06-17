from deeplab_parser import Parser
import cv2


class Streamer():
    def __init__(self):
        self.parser = Parser()

        self.capture = cv2.VideoCapture(0)

    def __del__(self):
        return
        # self.capture.release()

    def get_frame(self):
        ok, frame = self.capture.read()
        if ok:
            return frame
        else:
            return FileNotFoundError
