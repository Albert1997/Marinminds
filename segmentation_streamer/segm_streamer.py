from deeplab_parser import Parser
import time
import math
import cv2


class Streamer():
    def __init__(self, framerate, file):
        self.parser = Parser()
        self.time = time.time()
        self.framerate = framerate
        self.current_frame = 0
        self.accumulator = 0.0

        if file != '':
            file = file.replace("\\", "/")
            self.capture = cv2.VideoCapture(file)
        else:
            self.capture = cv2.VideoCapture("rtsp://root:marinminds@172.16.2.12:554/axis-media/media.amp")
        
        self.counter = 0
        self.total_frames = self.capture.get(7)

    def __del__(self):
        # return
        self.capture.release()

    def get_frame(self):
        frames = (time.time() - self.time) * self.framerate
        self.accumulator += frames - int(frames)
        frames = int(frames)

        if self.accumulator >= 1:
            frames += 1
            self.accumulator -= 1

        for i in range(frames):
            self.capture.read()

        frame = self.capture.read()[1]
        print("fps = {0}".format(1/(time.time()-self.time)))
        self.time = time.time()
        # self.counter += 1
        # if self.counter == 5:
        #     self.counter = 0
        # cv2.imencode('.jpg', frame)[1].tobytes()
        # return cv2.imread('../datasets/deeplab/images/' + self.frames[self.counter] + '.jpg')
        return frame
