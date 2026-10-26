#include <jni.h>
#include <android/log.h>
#include <vector>
#include <thread>

// AprilTag headers
extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"
#include "tag36h11.h"
#include "common/image_u8.h"
#include "common/zarray.h"
}

#define LOG_TAG "jni-apriltag"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// Global variables to hold the detector and family
static apriltag_detector_t *td = nullptr;
static apriltag_family_t *tf = nullptr;

// Cached Java class and constructor for AprilTagDetection
static jclass gDetectionClass = nullptr;
static jmethodID gDetectionCtor = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_com_example_inkmarc_1capture_MainActivity_nativeInit(JNIEnv *env, jclass clazz) {
    LOGI("Initializing AprilTag detector");
    // Create the tag family (e.g., tag36h11)
    tf = tag36h11_create();
    if (tf == nullptr) {
        LOGI("Failed to create tag family");
        return;
    }

    // Create the detector
    td = apriltag_detector_create();
    if (td == nullptr) {
        LOGI("Failed to create tag detector");
        tag36h11_destroy(tf);
        tf = nullptr;
        return;
    }
    // Add the family to the detector
    apriltag_detector_add_family(td, tf);

    // Set detector parameters (optional)
    td->quad_decimate = 1.0f;
    td->quad_sigma = 0.0f;
    unsigned int hwThreads = std::thread::hardware_concurrency();
    if (hwThreads == 0) {
        hwThreads = 4; // reasonable fallback
    }
    td->nthreads = static_cast<int>(hwThreads);
    td->debug = false;
    td->refine_edges = true;

    // Cache AprilTagDetection class and constructor for later use
    jclass localCls = env->FindClass("com/example/inkmarc_capture/AprilTagDetection");
    if (localCls == nullptr) {
        LOGI("Could not find AprilTagDetection class in nativeInit");
        return;
    }

    gDetectionClass = (jclass)env->NewGlobalRef(localCls);
    env->DeleteLocalRef(localCls);

    if (gDetectionClass == nullptr) {
        LOGI("Failed to create global ref for AprilTagDetection class");
        return;
    }

    gDetectionCtor = env->GetMethodID(gDetectionClass,
                                      "<init>",
                                      "(IIF[D[D[DII)V");
    if (gDetectionCtor == nullptr) {
        LOGI("Could not find AprilTagDetection constructor");
        // Clean up the global ref since we won't use it
        env->DeleteGlobalRef(gDetectionClass);
        gDetectionClass = nullptr;
        return;
    }

    LOGI("AprilTag detector initialized successfully");
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_inkmarc_1capture_MainActivity_nativeDestroy(JNIEnv *env, jclass clazz) {
    LOGI("Destroying AprilTag detector");
    if (td != nullptr) {
        apriltag_detector_destroy(td);
        td = nullptr;
    }
    if (tf != nullptr) {
        tag36h11_destroy(tf);
        tf = nullptr;
    }
    if (gDetectionClass != nullptr) {
        env->DeleteGlobalRef(gDetectionClass);
        gDetectionClass = nullptr;
    }
    gDetectionCtor = nullptr;
    LOGI("AprilTag detector destroyed");
}

// NOTE: This function assumes you have a Kotlin data class like this:
// data class AprilTagDetection(
//     val id: Int,
//     val hamming: Int,
//     val decisionMargin: Float,
//     val center: DoubleArray,
//     val corners: DoubleArray,
//     val homography: DoubleArray
// )
extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_inkmarc_1capture_MainActivity_nativeDetect(JNIEnv *env, jclass clazz, jint width, jint height, jint stride, jobject y_byte_buffer) {
    if (td == nullptr) {
        LOGI("Detector not initialized");
        return nullptr;
    }

    // Get direct access to the image data from the ByteBuffer
    auto *y_data = (uint8_t *) env->GetDirectBufferAddress(y_byte_buffer);
    if (y_data == nullptr) {
        LOGI("Failed to get direct buffer address");
        return nullptr;
    }

    // Create an AprilTag image structure
    image_u8_t im = {
            .width = width,
            .height = height,
            .stride = stride,
            .buf = y_data
    };

    // Run the detector
    zarray_t *detections = apriltag_detector_detect(td, &im);

    // Find the Java class and constructor for the detection result object
    // Use cached Java class and constructor
    if (gDetectionClass == nullptr || gDetectionCtor == nullptr) {
        LOGI("Detection class/constructor not initialized");
        apriltag_detections_destroy(detections);
        return nullptr;
    }

    int num_detections = zarray_size(detections);
    jobjectArray result_array = env->NewObjectArray(num_detections, gDetectionClass, nullptr);

    for (int i = 0; i < num_detections; i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);

        // --- Create center array ---
        jdoubleArray center_array = env->NewDoubleArray(2);
        env->SetDoubleArrayRegion(center_array, 0, 2, det->c);

        // --- Create corners array ---
        jdoubleArray corners_array = env->NewDoubleArray(8);
        jdouble corners_data[8] = {
            det->p[0][0], det->p[0][1],
            det->p[1][0], det->p[1][1],
            det->p[2][0], det->p[2][1],
            det->p[3][0], det->p[3][1]
        };
        env->SetDoubleArrayRegion(corners_array, 0, 8, corners_data);

        // --- Create homography array ---
        jdoubleArray homography_array = env->NewDoubleArray(9);
        jdouble homography_data[9];
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                homography_data[r * 3 + c] = matd_get(det->H, r, c);
            }
        }
        env->SetDoubleArrayRegion(homography_array, 0, 9, homography_data);

        jvalue args[8];
        args[0].i = (jint)det->id;
        args[1].i = (jint)det->hamming;
        args[2].f = (jfloat)det->decision_margin;
        args[3].l = center_array;
        args[4].l = corners_array;
        args[5].l = homography_array;
        args[6].i = (jint)width;
        args[7].i = (jint)height;

        // --- Create Java detection object ---
        jobject j_detection = env->NewObjectA(gDetectionClass, gDetectionCtor, args);
        if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); return nullptr; }

        env->SetObjectArrayElement(result_array, i, j_detection);

        // Clean up local references
        env->DeleteLocalRef(j_detection);
        env->DeleteLocalRef(center_array);
        env->DeleteLocalRef(corners_array);
        env->DeleteLocalRef(homography_array);
    }

    // Clean up AprilTag detections
    apriltag_detections_destroy(detections);

    return result_array;
}
