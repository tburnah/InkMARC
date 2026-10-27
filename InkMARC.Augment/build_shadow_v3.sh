#!/usr/bin/env bash
set -euo pipefail

g++ -std=c++17 -O3 -DNDEBUG \
  shadow_extract_v3_main.cpp \
  shadow_pipeline_v3.cpp \
  -o shadow_extract_v3 \
  -I/home/t/tburnah/miniconda3/envs/opencv_cuda/include/opencv4 \
  -I/home/t/tburnah/miniconda3/pkgs/hdf5-1.14.4-nompi_h2d575fe_105/include \
  -L/home/t/tburnah/src/opencv/build_cuda/lib \
  -L/home/t/tburnah/miniconda3/pkgs/hdf5-1.14.4-nompi_h2d575fe_105/lib \
  -L/home/t/tburnah/miniconda3/pkgs/libaec-1.1.3-h59595ed_0/lib \
  -Wl,-rpath,/home/t/tburnah/src/opencv/build_cuda/lib \
  -Wl,-rpath,/home/t/tburnah/miniconda3/pkgs/hdf5-1.14.4-nompi_h2d575fe_105/lib \
  -Wl,-rpath,/home/t/tburnah/miniconda3/pkgs/libaec-1.1.3-h59595ed_0/lib \
  -lopencv_aruco \
  -lopencv_objdetect \
  -lopencv_videoio \
  -lopencv_imgcodecs \
  -lopencv_imgproc \
  -lopencv_calib3d \
  -lopencv_core \
  -lhdf5 \
  -Wl,--no-as-needed \
  -lsz \
  -laec \
  -Wl,--as-needed \
  -pthread
