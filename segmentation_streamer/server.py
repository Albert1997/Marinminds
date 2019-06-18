from flask import Flask, render_template, Response
from segm_streamer import Streamer
from deeplab_parser import Parser
from inference.common_inference import construct_model
from inference.inference import inference
import cv2
import time
import pdb

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('./index.html')


def gen(streamer):
    session, inputs, logits = construct_model(
        "../models/deeplab_v3_plus_53_19e_0095", (513, 513), 5)
    parser = Parser()

    while True:
        frame = streamer.get_frame()
        pred = inference(frame, session, logits, inputs)
        result = parser.parse(pred, frame)

        result = cv2.imencode('.jpg', result)[1].tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + result + b'\r\n\r\n')
        # time.sleep(1/5)


@app.route('/video')
def video():
    return Response(gen(Streamer(24)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False)
