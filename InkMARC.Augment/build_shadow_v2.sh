#!/usr/bin/env bash
set -euo pipefail

g++ -std=c++17 -O3 -DNDEBUG \
  shadow_extract_v2_main.cpp \
  shadow_pipeline_v2.cpp \
  -o shadow_extract_v2 \
  -I/home/t/tburnah/miniconda3/envs/opencv_cuda/include/opencv4 \
  -L/home/t/tburnah/src/opencv/build_cuda/lib \
  -Wl,-rpath,/home/t/tburnah/src/opencv/build_cuda/lib \
  -lopencv_objdetect \
  -lopencv_videoio \
  -lopencv_imgcodecs \
  -lopencv_imgproc \
  -lopencv_calib3d \
  -lopencv_core \
  -pthread
