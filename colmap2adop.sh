#!/bin/bash

INPUT_DIR=$1
OUTPUT_DIR=$2

mkdir -p ${OUTPUT_DIR}/images/
cp ${INPUT_DIR}/images/* ${OUTPUT_DIR}/images/

./build/bin/colmap2adop --sparse_dir ${INPUT_DIR}/sparse/0/ \
                        --image_dir ${OUTPUT_DIR}/images/ \
                        --point_cloud_file ${INPUT_DIR}/dense/fused.ply \
                        --output_path ${OUTPUT_DIR} \
                        --scale_intrinsics 1 \
                        --render_scale 1
