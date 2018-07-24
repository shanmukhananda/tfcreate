from tfrecord import TFRecord
import os
from PIL import Image
import json

def area_rectangle(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    return dx * dy

def normalized_area_rectangle(x1, y1, x2, y2, img_width, img_height):
    area = area_rectangle(x1, y1, x2, y2)
    area /= img_width * img_height
    return area

def image_size(image_file):
    img = Image.open(image_file)
    width = img.width
    height = img.height
    img.close()
    return width, height

def bdd2tf(image_file, label_file):

    width, height = image_size(image_file)
    with open(label_file, "r") as jfile:
        bdd = json.load(jfile)

    tfr = TFRecord()
    tfr.input_fields["image/filename"] = image_file
    tfr.input_fields["image/source_id"] = bdd["name"]
    for frame in bdd["frames"]:
        for fobj in frame["objects"]:
            if "box2d" in fobj:
                area = normalized_area_rectangle(
                    fobj["box2d"]["x1"],
                    fobj["box2d"]["y1"],
                    fobj["box2d"]["x2"],
                    fobj["box2d"]["y2"],
                    width,
                    height
                )

                tfr.input_fields["image/object/area"].append(area)
                tfr.input_fields["image/object/bbox/label"].append(fobj["id"])
                tfr.input_fields["image/object/bbox/xmax"].append(fobj["box2d"]["x2"]/width)
                tfr.input_fields["image/object/bbox/xmin"].append(fobj["box2d"]["x1"]/width)
                tfr.input_fields["image/object/bbox/ymax"].append(fobj["box2d"]["y2"]/height)
                tfr.input_fields["image/object/bbox/ymin"].append(fobj["box2d"]["y1"]/height)
                tfr.input_fields["image/object/class/label"].append(fobj["id"])
                tfr.input_fields["image/object/class/text"].append(fobj["category"])
                tfr.input_fields["image/object/occluded"].append(int(fobj["attributes"]["occluded"]))
                tfr.input_fields["image/object/truncated"].append(int(fobj["attributes"]["truncated"]))

    return tfr
