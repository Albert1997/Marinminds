from segm_streamer import Streamer
from deeplab_parser import Parser
from inference.common_inference import construct_model
from inference.inference import inference
import cv2
import time
import pdb


def stream(streamer):
    session, inputs, logits = construct_model(
        "../models/deeplab_v3_plus_53_19e_0095", (513, 513), 5)
    parser = Parser()

    cap = cv2.VideoCapture("rtsp://127.0.0.1:5000")

    framerate = 12.0

    out = cv2.VideoWriter('appsrc ! videoconvert ! '
                          'x264enc noise-reduction=10000 speed-preset=ultrafast tune=zerolatency ! '
                          'rtph264pay config-interval=1 pt=96 !'
                          'tcpserversink host=127.0.0.1 port=5000 sync=false',
                          0, framerate, (1920, 1080))

    counter = 0
    while cap.isOpened():
        frame = streamer.get_frame()
        pred = inference(frame, session, logits, inputs)
        result = parser.parse(pred, frame)

        # result = cv2.imencode('.jpg', result)[1].tobytes()

        out.write(result)

    cap.release()
    out.release()


stream(Streamer(24))
