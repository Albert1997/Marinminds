import cv2
import numpy as np
from inference.common_inference import construct_model

KEEP_ASPECT_RATIO = -1


def restore_proportional_size(in_size: tuple, out_size: tuple = None,
                              frow: float = None, fcol: float = None, f: float = None):
    if out_size is not None and (frow is not None or fcol is not None) and f is None:
        raise ValueError(
            'Must be specified output size or scale factors not both of them.')

    if out_size is not None:
        if out_size[0] == KEEP_ASPECT_RATIO and out_size[1] == KEEP_ASPECT_RATIO:
            raise ValueError('Must be specified at least 1 dimension of size!')

        if (out_size[0] <= 0 and out_size[0] != KEEP_ASPECT_RATIO) or \
                (out_size[1] <= 0 and out_size[1] != KEEP_ASPECT_RATIO):
            raise ValueError('Size dimensions must be greater than 0.')

        result_row = out_size[0] if out_size[0] > 0 else max(
            1, round(out_size[1] / in_size[1] * in_size[0]))
        result_col = out_size[1] if out_size[1] > 0 else max(
            1, round(out_size[0] / in_size[0] * in_size[1]))
    else:
        if f is not None:
            if f < 0:
                raise ValueError('"f" argument must be positive!')
            frow = fcol = f

        if (frow < 0 or fcol < 0) or (frow is None or fcol is None):
            raise ValueError('Specify "f" argument for single scale!')

        result_col = round(fcol * in_size[1])
        result_row = round(frow * in_size[0])
    return result_row, result_col


def inference(img, session, logits, inputs):
    h, w = img.shape[:2]

    result_height, result_width = restore_proportional_size(
        img.shape[:2], (513, 513))

    img = cv2.resize(img, (result_width, result_height),
                     interpolation=cv2.INTER_CUBIC)

    input_image_var = np.expand_dims(img.astype(np.float32), 0)
    raw_pixelwise_probas_array = session.run(
        logits, feed_dict={inputs: input_image_var})
    # Resize back to the original
    pixelwise_probas_array = cv2.resize(np.squeeze(
        raw_pixelwise_probas_array[0]), (w, h), cv2.INTER_LINEAR)

    return pixelwise_probas_array
