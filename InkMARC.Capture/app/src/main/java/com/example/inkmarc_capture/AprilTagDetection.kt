package com.example.inkmarc_capture

/**
 * Represents a detected AprilTag.
 *
 * @property id The decoded ID of the tag.
 * @property hamming The number of bit errors corrected.
 * @property decisionMargin A measure of the quality of the detection.
 * @property center The center of the tag in image coordinates.
 * @property corners The corners of the tag in image coordinates.
 * @property homography The homography matrix that transforms tag coordinates to image coordinates.
 * @property width The width of the camera image.
 * @property height The height of the camera image.
 */
data class AprilTagDetection(
    val id: Int,
    val hamming: Int,
    val decisionMargin: Float,
    val center: DoubleArray,
    val corners: DoubleArray,
    val homography: DoubleArray,
    val width: Int,
    val height: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as AprilTagDetection

        if (id != other.id) return false
        if (hamming != other.hamming) return false
        if (decisionMargin != other.decisionMargin) return false
        if (width != other.width) return false
        if (height != other.height) return false
        if (!center.contentEquals(other.center)) return false
        if (!corners.contentEquals(other.corners)) return false
        if (!homography.contentEquals(other.homography)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = id
        result = 31 * result + hamming
        result = 31 * result + decisionMargin.hashCode()
        result = 31 * result + width
        result = 31 * result + height
        result = 31 * result + center.contentHashCode()
        result = 31 * result + corners.contentHashCode()
        result = 31 * result + homography.contentHashCode()
        return result
    }
}
