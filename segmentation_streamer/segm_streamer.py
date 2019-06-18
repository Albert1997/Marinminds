from deeplab_parser import Parser
import time
import math
import cv2


class Streamer():
    def __init__(self, framerate, camera):
        self.parser = Parser()
        self.time = 0
        self.framerate = framerate
        self.current_frame = 0

        self.capture = cv2.VideoCapture(camera)
        self.counter = 0
        self.total_frames = self.capture.get(7)

    def __del__(self):
        # return
        self.capture.release()

    def get_frame(self):
        if self.time == 0:
            self.current_frame = self.current_frame + \
                (int(time.time() - self.time) * int(self.framerate))

            if self.total_frames >= self.current_frame:
                self.capture.set(1, self.current_frame)

        frame = self.capture.read()[1]
        self.time = time.time()
        # self.counter += 1
        # if self.counter == 5:
        #     self.counter = 0
        # cv2.imencode('.jpg', frame)[1].tobytes()
        # return cv2.imread('../datasets/deeplab/images/' + self.frames[self.counter] + '.jpg')
        return frame
