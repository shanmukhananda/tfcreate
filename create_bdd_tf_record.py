from tfrecord import TFRecord
import concurrent.futures
import fnmatch
import json
import logging
import os
import sys
import tensorflow as tf
import time
import utils

logging.basicConfig(
    filename="create_bdd.log",
    level=logging.INFO,
    format="%(levelname)s:%(asctime)s:%(processName)s:%(process)d:%(threadName)s:%(thread)d:%(filename)s:%(lineno)d:%(funcName)s:%(message)s"
)

def generate_tfrecord_from_bdd(image_file, label_file, tfrecord_file):
    logging.debug("processing label:{}".format(label_file))

    with open(label_file, "r") as jfile:
        bdd = json.load(jfile)

    tfr = utils.bdd2tf(image_file, label_file)

    width, height = utils.image_size(image_file)
    tfdata = tfr.train_example((int(width/2), int(height/2)))

    logging.debug("saving {}".format(tfrecord_file))
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    writer.write(tfdata.SerializeToString())

def bdd_source(images_dir, labels_dir, output_dir):
    assert os.path.isdir(images_dir), "{} is not a valid folder!".format(images_dir)
    assert os.path.isdir(labels_dir), "{} is not a valid folder!".format(labels_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert os.path.isdir(output_dir), "could not create folder {}".format(output_dir)

    files_jpg = fnmatch.filter(os.listdir(images_dir), "*.jpg")
    files_jpg.sort()
    files_json = fnmatch.filter(os.listdir(labels_dir), "*.json")
    files_json.sort()

    for img, label in zip(files_jpg, files_json):
        img_file, ext = os.path.splitext(img)
        json_file, ext = os.path.splitext(label)

        if os.path.basename(img_file) != os.path.basename(json_file):
            raise RuntimeError("base filenames not matching {} != {}".format(img, label))

        img_path = os.path.join(images_dir, img)
        label_path = os.path.join(labels_dir, label)

        filepath, ext = os.path.splitext(img_path)
        fname = os.path.basename(filepath)
        ouput_path = os.path.join(output_dir, fname + ".tfrecord")

        yield (img_path, label_path, ouput_path)

def main(argv):
    start = time.perf_counter()
    assert len(argv) == 5 , "usage: create_bdd_tf_record.py images_dir labels_dir output_dir num_threads"
    logging.info("{}".format(argv))

    images_dir = argv[1]
    labels_dir = argv[2]
    output_dir = argv[3]
    num_threads = int(argv[4])

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for img_path, label_path, ouput_path in bdd_source(images_dir, labels_dir, output_dir):
            executor.submit(generate_tfrecord_from_bdd, img_path, label_path, ouput_path)

    end = time.perf_counter()
    logging.info("Time taken = {}".format(end - start))

if __name__ == "__main__":
    main(sys.argv)
