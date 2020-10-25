import numpy as np
import os
import json
from PIL import Image
import base64
import io
from imageio import imwrite
import time
import random
import cv2


class LabelMePolygon:
    """
    self.label = (String) label name.
    self.points = (ndarray) points array.
    """

    def __init__(self, label, points):
        self.label = label
        self.points = np.array(points)


class LabelMeItem:

    def __init__(self, name, image, label_me_polygon_list):
        self.name = name
        self.image = image
        self.label_me_polygon_list = label_me_polygon_list
        self.img_size = image.shape


class LabelMap:
    """
    'LabelMap' is object which contains 'labelMe' parsing data.
    If you are interesting about LabelMe, please check below link.

    https://github.com/wkentaro/labelme/tree/master/examples/tutorial#convert-to-dataset

    """

    def __init__(self, json_path):
        file_names = os.listdir(json_path)

        assert len(file_names) > 0, "Path is empty. please check a path."

        # extract only json file.
        paths = [os.path.join(json_path, i) for i in file_names if i.endswith(".json")]

        # all images and polygon, label data.
        self.label_me_item_list = []

        # all labels
        all_label = []

        for file in paths:
            with open(file) as json_file:
                parsed_data = json.load(json_file, encoding='UTF-8')

                f = io.BytesIO()
                f.write(base64.b64decode(parsed_data['imageData']))
                np_image = np.array(Image.open(f))

                polygon_shapes = parsed_data['shapes']

                polygon_list = []

                for polygon_data in polygon_shapes:
                    polygon_list.append(LabelMePolygon(polygon_data['label'], polygon_data['points']))
                    all_label.append(polygon_data['label'])

            self.label_me_item_list.append(LabelMeItem(str.split(str.split(file, os.path.sep)[-1], ".")[0], np_image, polygon_list))

        # unique labels.
        self.labels = list(set(all_label))

        # color palate for instance label.
        self.color_list = [(random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256)) for c in self.labels]

        # prevent duplication.
        while len(set(self.color_list)) < len(self.labels):
            self.color_list = [(random.randrange(0, 256), random.randrange(0, 256), random.randrange(0, 256)) for c in self.labels]

        self.origin_img_path = None
        self.semantic_saved_path = None
        self.instance_saved_path = None

    def set_color_palate(self, color_list):
        assert len(color_list) == len(self.labels), "Colors should be prepared size of labels, label size is : {}".format(len(self.labels))
        self.color_list = color_list

    def save_original_img(self, target_path):
        """
        Extract origin image.
        """
        if os.path.isfile(target_path):
            raise RuntimeError("'target_path' should be path, not a file.")
        elif not os.path.isdir(target_path):
            os.mkdir(target_path)

        for idx, label_me_item in enumerate(self.label_me_item_list):
            imwrite(os.path.join(target_path, label_me_item.name + ".png"), label_me_item.image)

            print("PROCESSING save origin images...[{}/{}]".format(idx + 1, len(self.label_me_item_list)), end="\r", flush=True)

        self.origin_img_path = target_path
        print("All origin data is saved as image at : {}".format(self.origin_img_path))

    def save_semantic_label(self, target_path):
        """
        Extract instance label from label_me_item
        """
        if os.path.isfile(target_path):
            raise RuntimeError("'target_path' should be path, not a file.")
        elif not os.path.isdir(target_path):
            os.mkdir(target_path)

        for idx, label_me_item in enumerate(self.label_me_item_list):
            img_blank = np.zeros(label_me_item.img_size, np.uint8)

            # make polygon on blank image.
            for idx_poly, poly in enumerate(label_me_item.label_me_polygon_list):
                cv2.fillPoly(img_blank, [np.array(poly.points, np.int32)], color=(255, 255, 255))

            imwrite(os.path.join(target_path, label_me_item.name + ".png"), img_blank)
            print("PROCESSING save semantic label images...[{}/{}]".format(idx + 1, len(self.label_me_item_list)), end="\r", flush=True)

        self.semantic_saved_path = target_path
        print("All semantic label data is saved as image at : {}".format(self.semantic_saved_path))

    def save_instance_label(self, target_path):
        """
        Extract instance label from label_me_item
        """
        if os.path.isfile(target_path):
            raise RuntimeError("'target_path' should be path, not a file.")
        elif not os.path.isdir(target_path):
            os.mkdir(target_path)

        for idx, label_me_item in enumerate(self.label_me_item_list):
            img_blank = np.zeros(label_me_item.img_size, np.uint8)

            # make polygon on blank image.
            for idx_poly, poly in enumerate(label_me_item.label_me_polygon_list):
                cv2.fillPoly(img_blank, [np.array(poly.points, np.int32)], color=self.color_list[self.labels.index(poly.label)])

            imwrite(os.path.join(target_path, label_me_item.name + ".png"), img_blank)
            print("PROCESSING save instance label images...[{}/{}]".format(idx + 1, len(self.label_me_item_list)), end="\r", flush=True)

        self.instance_saved_path = target_path
        print("All instance label data is saved as image at : {}".format(self.instance_saved_path))
