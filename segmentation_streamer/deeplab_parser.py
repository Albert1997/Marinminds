import json

import cv2
import numpy as np


class Parser():
    def __init__(self):
        with open('../datasets/deeplab/meta.json', 'r') as meta_json:
            meta = json.load(meta_json)

        with open('../datasets/deeplab/obj_class_to_machine_color.json', 'r') as obj_map_json:
            obj_map = json.load(obj_map_json)

        self.img_classes = {}

        for obj_class, obj_val in obj_map.items():
            self.img_classes[obj_class] = {'value': obj_val[0]}

        for img_class in meta['classes']:
            for obj_val, obj_class in self.img_classes.items():
                if obj_val == img_class['title']:
                    hex_color = img_class['color'].lstrip('#')
                    self.img_classes[obj_val]['color'] = tuple(
                        int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def parse(self, arr, img):
        arr = np.argmax(arr, axis=2)
        arr = np.expand_dims(arr, axis=2)
        img_class = arr.T

        for _, img_val in self.img_classes.items():
            obj_group = img_class == img_val['value']
            if obj_group.any():
                arr = np.where(obj_group.T, arr, img_val['color'])

        arr = arr.astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        result_img = cv2.addWeighted(
            arr, 0.5, img, 0.5, 0)

        return result_img