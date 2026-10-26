cmake_minimum_required(VERSION 3.22)
project(apriltag_android)

set(APRILTAG_DIR ${CMAKE_SOURCE_DIR}/third_party/apriltag)

# Grab all common/*.c that exist in your vendored copy
file(GLOB APRILTAG_COMMON_SOURCES
        "${APRILTAG_DIR}/common/*.c"
)

# Some AprilTag copies include getopt.c which can conflict on Android.
# If present, remove it.
list(FILTER APRILTAG_COMMON_SOURCES EXCLUDE REGEX ".*/getopt\\.c$")

add_library(apriltag SHARED
        jni-apriltag.cpp

        # Core
        ${APRILTAG_DIR}/apriltag.c
        ${APRILTAG_DIR}/apriltag_pose.c
        ${APRILTAG_DIR}/apriltag_quad_thresh.c

        # One tag family (adjust if you use another)
        ${APRILTAG_DIR}/tag36h11.c

        # Whatever exists in common/
        ${APRILTAG_COMMON_SOURCES}
)

target_include_directories(apriltag PRIVATE
        ${APRILTAG_DIR}
        ${APRILTAG_DIR}/common
)

find_library(log-lib log)
target_link_libraries(apriltag ${log-lib})
