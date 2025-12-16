#!/bin/bash

# Take all the files in the current directory 

for file in ../CMORE/inputs/noLabels/*; do
    echo "Running $file..."

    python tools/generate_keypoints_log.py yolo11n640.general.pt "$file" output/

done