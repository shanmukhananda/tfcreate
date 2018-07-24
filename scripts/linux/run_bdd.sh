#!/bin/bash

script_dir=$(realpath $(dirname $0))
project_dir=$(realpath ${script_dir}/../..)

# images_dir=/mnt/hgfs/D/DATA/Projects/BDD/bdd100k_images/bdd100k/images/100k/train
# labels_dir=/mnt/hgfs/D/DATA/Projects/BDD/bdd100k_labels/bdd100k/labels/100k/train
# output_dir=/mnt/hgfs/D/DATA/Projects/BDD/tfrecord

images_dir=${project_dir}/tests/bdd/images
labels_dir=${project_dir}/tests/bdd/labels
output_dir=${project_dir}/tests/bdd/tfrecord

num_threads=$(nproc)
python ${project_dir}/create_bdd_tf_record.py ${images_dir} ${labels_dir} ${output_dir} ${num_threads}
