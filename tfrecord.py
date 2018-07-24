from PIL import Image
import hashlib
import os
import tensorflow as tf
import io

class TFRecord(object):

    def __init__(self):
        self.input_fields = {
            "image/filename": "", # filename of image
            "image/object/area" : [], # a float array of object areas; normalized coordinates. For example, the simplest case would simply be the area of the bounding boxes. Or it could be the size of the segmentation. Normalized in this case means that the area is divided by the (image width x image height)
            "image/object/bbox/label" : [], # an integer array, specifying the index in a classification layer. The label ranges from [0, num_labels)
            "image/object/bbox/xmax" : [], # object_bbox_xmax: xmax coordinates of groundtruth box, e.g. 50, 40
            "image/object/bbox/xmin" : [], # object_bbox_xmin: xmin coordinates of groundtruth box, e.g. 10, 30
            "image/object/bbox/ymax" : [], # object_bbox_ymax: ymax coordinates of groundtruth box, e.g. 80, 70
            "image/object/bbox/ymin" : [], # object_bbox_ymin: ymin coordinates of groundtruth box, e.g. 40, 50
            "image/object/class/label" : [], # object_class_label: labels in numbers, e.g. [16, 8]
            "image/object/class/text" : [], # object_class_text: labels in text format, e.g. ["person", "cat"]
            "image/object/group_of" : [], # object_group_of: is object a single object or a group of objects
            "image/object/occluded" : [], # object_occluded: is object occluded, e.g. [true, false]
            "image/object/truncated" : [], # object_truncated: is object truncated, e.g. [true, false]
            "image/source_id" : "", # source_id: original source of the image
        }
        self.__tf_fields = {
            "image/channels": 0, # channels: number of channels of image
            "image/colorspace": "", # colorspace, e.g. "RGB"
            "image/encoded": "", # image_encoded: JPEG encoded string
            "image/filename": "", # filename
            "image/format" : "", # image format, e.g. "JPEG"
            "image/height" : 0, # height of image in pixels, e.g. 462
            "image/key/sha256" : "", # hash is calculated internally user need not have to provide
            "image/object/area" : [], #a float array of object areas; normalized coordinates. For example, the simplest case would simply be the area of the bounding boxes. Or it could be the size of the segmentation. Normalized in this case means that the area is divided by the (image width x image height)
            "image/object/bbox/label" : [], # an integer array, specifying the index in a classification layer. The label ranges from [0, num_labels)
            "image/object/bbox/xmax" : [], # object_bbox_xmax: xmax coordinates of groundtruth box, e.g. 50, 40
            "image/object/bbox/xmin" : [], # object_bbox_xmin: xmin coordinates of groundtruth box, e.g. 10, 30
            "image/object/bbox/ymax" : [], # object_bbox_ymax: ymax coordinates of groundtruth box, e.g. 80, 70
            "image/object/bbox/ymin" : [], # object_bbox_ymin: ymin coordinates of groundtruth box, e.g. 40, 50
            "image/object/class/label" : [], # object_class_label: labels in numbers, e.g. [16, 8]
            "image/object/class/text" : [], # object_class_text: labels in text format, e.g. ["person", "cat"]
            "image/object/group_of" : [], # object_group_of: is object a single object or a group of objects
            "image/object/occluded" : [], # object_occluded: is object occluded, e.g. [true, false]
            "image/object/truncated" : [], # object_truncated: is object truncated, e.g. [true, false]
            "image/source_id" : "", # source_id: original source of the image
            "image/width" : 0 # width: width of image in pixels, e.g. 581
        }

    def train_example(self, resize):
        self.__update_tf_fields(resize)
        example = tf.train.Example(features=tf.train.Features(feature=self.__tf_fields))
        return example

    def __update_image_details(self, resize):
        img_file = self.input_fields["image/filename"]
        img = Image.open(img_file)
        iformat = img.format
        channels = img.layers
        colorspace = img.mode
        width = img.width
        height = img.height

        path, ext = os.path.splitext(img_file)
        fname = os.path.basename(path)

        img_resized = img.resize(resize)
        resized_img = io.BytesIO()
        img_resized.save(resized_img, iformat)
        encoded = resized_img.getvalue()

        key = hashlib.sha256(encoded).hexdigest()

        self.__tf_fields["image/channels"] = self.__int64_feature(channels)
        self.__tf_fields["image/colorspace"] = self.__bytes_feature(colorspace.encode("utf8"))
        self.__tf_fields["image/encoded"] = self.__bytes_feature(encoded)
        self.__tf_fields["image/format"] = self.__bytes_feature(iformat.encode("utf8"))
        self.__tf_fields["image/height"] = self.__int64_feature(height)
        self.__tf_fields["image/key/sha256"] = self.__bytes_feature(key.encode("utf8"))
        self.__tf_fields["image/width"] = self.__int64_feature(width)

    def __update_tf_fields(self, resize):
        path, filename = os.path.split(self.input_fields["image/filename"])
        self.__tf_fields["image/filename"] = self.__bytes_feature(filename.encode("utf8"))
        self.__tf_fields["image/object/area"] = self.__float_list_feature(self.input_fields["image/object/area"])
        self.__tf_fields["image/object/bbox/label"] = self.__int64_list_feature(self.input_fields["image/object/bbox/label"])
        self.__tf_fields["image/object/bbox/xmax"] = self.__float_list_feature(self.input_fields["image/object/bbox/xmax"])
        self.__tf_fields["image/object/bbox/xmin"] = self.__float_list_feature(self.input_fields["image/object/bbox/xmin"])
        self.__tf_fields["image/object/bbox/ymax"] = self.__float_list_feature(self.input_fields["image/object/bbox/ymax"])
        self.__tf_fields["image/object/bbox/ymin"] = self.__float_list_feature(self.input_fields["image/object/bbox/ymin"])
        self.__tf_fields["image/object/class/label"] = self.__int64_list_feature(self.input_fields["image/object/class/label"])

        class_text = []
        for text in self.input_fields["image/object/class/text"]:
            class_text.append(text.encode("utf8"))

        self.__tf_fields["image/object/class/text"] = self.__bytes_list_feature(class_text)
        self.__tf_fields["image/object/group_of"] = self.__int64_list_feature(self.input_fields["image/object/group_of"])
        self.__tf_fields["image/object/occluded"] = self.__int64_list_feature(self.input_fields["image/object/occluded"])
        self.__tf_fields["image/object/truncated"] = self.__int64_list_feature(self.input_fields["image/object/truncated"])
        self.__tf_fields["image/source_id"] = self.__bytes_feature(self.input_fields["image/source_id"].encode("utf8"))
        self.__update_image_details(resize)

    def __int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def __int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def __bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def __bytes_list_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def __float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def __float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
