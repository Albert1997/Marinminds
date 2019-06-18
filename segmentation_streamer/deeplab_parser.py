import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import time


class Parser():
    def __init__(self):
        with open('../datasets/meta.json', 'r') as meta_json:
            meta = json.load(meta_json)

        with open('../datasets/obj_class_to_machine_color.json', 'r') as obj_map_json:
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
        arr = np.pad(arr, pad_width=((0, 0), (0, 0), (0, 2)), mode='constant', constant_values=0)
        img_class, _, _ = arr.T

        for _, img_val in self.img_classes.items():
            obj_group = img_class == img_val['value']
            arr[...][obj_group.T] = img_val['color']

        arr = arr.astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        arr = cv2.resize(arr, (1920, 1080), cv2.INTER_LINEAR)

        result_img = cv2.addWeighted(
            arr, 0.5, img, 0.5, 0)
            
        return result_img
