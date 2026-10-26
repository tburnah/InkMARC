package com.inkmarc.fallback

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Frame-rate sparse LK tracking with periodic correction from a bounded SIFT keyframe gallery.
 *
 * Descriptor work runs at half resolution and is rate limited. The immutable origin is retained,
 * while verified intermediate views are appended and the oldest non-origin view is evicted.
 */
class GallerySurfaceTrackerCore(
  private val sparse: SurfaceTrackerCore = SurfaceTrackerCore(),
  private val config: Config = Config(),
) : AutoCloseable {
  data class Config(
    val featureScale: Double = 0.5,
    val descriptorIntervalFrames: Int = 45,
    val recoveryRetryIntervalFrames: Int = 3,
    val triggerConfidence: Float = 0.62f,
    val triggerInlierRatio: Float = 0.52f,
    val triggerInlierCount: Int = 12,
    val maxHealthyCorrectionRatio: Double = 0.035,
    val maxRecoveryCorrectionRatio: Double = 0.18,
    val maxHealthyAreaScaleChange: Double = 0.18,
    val promotionIntervalFrames: Int = 8,
    val minStableFramesForPromotion: Int = 8,
    val minPromotionViewChangeRatio: Double = 0.055,
    val originIntervalFrames: Int = 15,
    val maxReferences: Int = 5,
    val maxSiftFeatures: Int = 500,
    val localSearchScale: Double = 1.65,
    val originSearchScale: Double = 2.20,
    val ratioTest: Float = 0.78f,
    val minMatches: Int = 6,
    val minInliers: Int = 6,
    val minCoverage: Double = 0.16,
    val minPatchSimilarity: Double = 0.10,
    val maxReprojectionErrorPx: Double = 2.5,
    val minPromotionSimilarity: Double = 0.12,
    val minPromotionInlierRatio: Double = 0.45,
    val maxPromotionErrorPx: Double = 1.8,
  )

  private data class Reference(
    val label: String,
    val quad: List<Point>,
    val keypoints: Array<KeyPoint>,
    val descriptors: Mat,
    val patch: Mat,
    val createdFrame: Int,
    var lastMatchedFrame: Int,
  ) {
    fun release() {
      descriptors.release()
      patch.release()
    }
  }

  private data class Features(
    val keypoints: Array<KeyPoint>,
    val descriptors: Mat,
  ) : AutoCloseable {
    override fun close() = descriptors.release()
  }

  private data class GalleryMatch(
    val reference: Reference,
    val quad: List<Point>,
    val homography: DoubleArray,
    val support: Int,
    val errorPx: Double,
    val similarity: Double,
    val inlierRatio: Double,
    val referenceCoverage: Double,
    val currentCoverage: Double,
    val quality: Double,
    val predictionErrorRatio: Double,
    val predictedAreaScale: Double,
  )

  private val sift =
    SIFT.create(config.maxSiftFeatures, 3, 0.008, 14.0, 1.6)
  private val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE)
  private val references = ArrayList<Reference>(config.maxReferences)
  private var frameCounter = 0
  private var lastDescriptorFrame = 0
  private var lastPromotionFrame = 0
  private var consecutiveStableFrames = 0
  private var nextReferenceIndex = 1
  private var lastQuad: List<Point>? = null

  fun seedReference(
    grayFrame: Mat,
    referenceQuad: List<Point>,
    exclusionMask: Mat? = null,
  ): SurfaceTrackerCore.SeedResult {
    val result = sparse.seedReference(grayFrame, referenceQuad, exclusionMask)
    clearReferences()
    frameCounter = 0
    lastDescriptorFrame = 0
    lastPromotionFrame = 0
    consecutiveStableFrames = 0
    nextReferenceIndex = 1
    lastQuad = result.mappedQuad?.map(::copyPoint)
    if (!result.success) return result

    val scaledGray = scaledFrame(grayFrame)
    val scaledExclusion = scaledMask(exclusionMask, scaledGray.size())
    try {
      captureReference(scaledGray, scaleQuad(referenceQuad, config.featureScale), "origin", scaledExclusion)
        ?.let(references::add)
    } finally {
      scaledExclusion?.release()
      scaledGray.release()
    }
    return result.copy(message = "${result.message}; gallery=${references.size}")
  }

  fun update(
    grayFrame: Mat,
    exclusionMask: Mat? = null,
    initialFlowTransform: SurfaceTrackerCore.SurfaceTransform? = null,
  ): SurfaceTrackerCore.TrackerResult {
    frameCounter += 1
    val sparseResult = sparse.update(grayFrame, exclusionMask, initialFlowTransform)
    sparseResult.mappedQuad?.let { lastQuad = it.map(::copyPoint) }
    val predicted = sparseResult.mappedQuad ?: lastQuad
    val stableTracking =
      sparseResult.state == SurfaceTrackerCore.State.TRACKING &&
        !sparseResult.heldLastGoodTransform &&
        sparseResult.confidence >= config.triggerConfidence &&
        sparseResult.inlierRatio >= config.triggerInlierRatio &&
        sparseResult.inlierCount >= config.triggerInlierCount
    consecutiveStableFrames = if (stableTracking) consecutiveStableFrames + 1 else 0
    val matchInterval =
      if (stableTracking) config.descriptorIntervalFrames else config.recoveryRetryIntervalFrames
    val shouldMatch =
      frameCounter - lastDescriptorFrame >= max(1, matchInterval)
    if (!shouldMatch || predicted == null || references.isEmpty()) return sparseResult
    lastDescriptorFrame = frameCounter

    val scaledGray = scaledFrame(grayFrame)
    val scaledExclusion = scaledMask(exclusionMask, scaledGray.size())
    val predictedScaled = scaleQuad(predicted, config.featureScale)
    val matches = ArrayList<GalleryMatch>(references.size + 1)
    val failures = ArrayList<String>(references.size + 1)
    try {
      detectFeatures(scaledGray, predictedScaled, config.localSearchScale, scaledExclusion)?.use { features ->
        for (reference in references) {
          val (match, reason) = matchReference(scaledGray, predictedScaled, features, reference)
          if (match != null) matches.add(match) else failures.add("${reference.label}:$reason")
        }
      } ?: failures.add("local:no_descriptors")

      val shouldCloseOrigin =
        frameCounter % max(1, config.originIntervalFrames) == 0 || matches.isEmpty()
      val origin = references.firstOrNull { it.label == "origin" }
      if (shouldCloseOrigin && origin != null) {
        detectFeatures(scaledGray, predictedScaled, config.originSearchScale, scaledExclusion)?.use { features ->
          val (match, reason) = matchReference(scaledGray, predictedScaled, features, origin)
          if (match != null) matches.add(match) else failures.add("origin_wide:$reason")
        } ?: failures.add("origin_wide:no_descriptors")
      }

      val allowedMatches = matches.filter { isCorrectionAllowed(it, stableTracking) }
      if (allowedMatches.isEmpty() && matches.isNotEmpty()) {
        failures.add("innovation")
      }
      val correction = allowedMatches.maxByOrNull { it.quality }
        ?: return sparseResult.copy(
          rejectionReason = combineReasons(sparseResult.rejectionReason, galleryFailure(failures)),
        )
      val correctedQuad = scaleQuad(correction.quad, 1.0 / config.featureScale)
      val reseed = sparse.seedReference(grayFrame, correctedQuad, exclusionMask)
      if (!reseed.success) {
        sparseResult.mappedQuad?.let { previousQuad ->
          sparse.seedReference(grayFrame, previousQuad, exclusionMask)
        }
        return sparseResult.copy(
          rejectionReason = combineReasons(sparseResult.rejectionReason, "gallery:reseed_failed"),
        )
      }
      lastQuad = correctedQuad.map(::copyPoint)
      correction.reference.lastMatchedFrame = frameCounter

      var promoted = false
      if (shouldPromote(correction)) {
        val label = "k${nextReferenceIndex}"
        val reference = captureReference(scaledGray, correction.quad, label, scaledExclusion)
        if (reference != null) {
          nextReferenceIndex += 1
          references.add(reference)
          lastPromotionFrame = frameCounter
          while (references.size > max(2, config.maxReferences)) {
            val evictionIndex =
              (1 until references.size).minByOrNull { index -> references[index].lastMatchedFrame }
                ?: 1
            references.removeAt(evictionIndex).release()
          }
          promoted = true
        }
      }

      val action = if (promoted) "promote" else "correct"
      val diagnostic =
        "gallery:$action:${correction.reference.label}:refs=${references.size}:" +
          "sim=${format(correction.similarity)}:ratio=${format(correction.inlierRatio)}:" +
          "quality=${format(correction.quality)}:prediction_ratio=" +
          "${format(correction.predictionErrorRatio)}:area_scale=" +
          format(correction.predictedAreaScale)
      return SurfaceTrackerCore.TrackerResult(
        state = SurfaceTrackerCore.State.TRACKING,
        activeTrackCount = reseed.trackCount,
        candidateTrackCount = correction.support,
        stableTrackCount = 0,
        inlierCount = correction.support,
        inlierRatio = correction.inlierRatio.toFloat(),
        confidence = correction.quality.coerceIn(0.0, 1.0).toFloat(),
        transform = SurfaceTrackerCore.HomographyTransform(fullResolutionHomography(correction.homography)),
        mappedQuad = correctedQuad.map(::copyPoint),
        meanReprojectionErrorPx = correction.errorPx / config.featureScale,
        referenceFeatureCoverage = correction.referenceCoverage.toFloat(),
        currentFeatureCoverage = correction.currentCoverage.toFloat(),
        transformCondition = null,
        heldLastGoodTransform = false,
        rejectionReason = diagnostic,
        referenceWasReseeded = true,
      )
    } finally {
      scaledExclusion?.release()
      scaledGray.release()
    }
  }

  fun reset() {
    sparse.reset()
    clearReferences()
    frameCounter = 0
    lastDescriptorFrame = 0
    lastPromotionFrame = 0
    consecutiveStableFrames = 0
    nextReferenceIndex = 1
    lastQuad = null
  }

  override fun close() {
    reset()
    sift.clear()
    matcher.clear()
    sparse.close()
  }

  private fun detectFeatures(
    gray: Mat,
    quad: List<Point>,
    expansion: Double,
    exclusionMask: Mat?,
  ): Features? {
    val mask = buildMask(gray.rows(), gray.cols(), expandQuad(quad, expansion), exclusionMask)
    val keypointMat = MatOfKeyPoint()
    val descriptors = Mat()
    return try {
      sift.detectAndCompute(gray, mask, keypointMat, descriptors)
      val keypoints = keypointMat.toArray()
      if (descriptors.empty() || keypoints.size < config.minMatches) {
        descriptors.release()
        null
      } else {
        Features(keypoints, descriptors)
      }
    } finally {
      mask.release()
      keypointMat.release()
    }
  }

  private fun captureReference(
    gray: Mat,
    quad: List<Point>,
    label: String,
    exclusionMask: Mat?,
  ): Reference? {
    val mask = buildMask(gray.rows(), gray.cols(), quad, exclusionMask)
    val keypointMat = MatOfKeyPoint()
    val descriptors = Mat()
    return try {
      sift.detectAndCompute(gray, mask, keypointMat, descriptors)
      val keypoints = keypointMat.toArray()
      if (descriptors.empty() || keypoints.size < config.minMatches) {
        descriptors.release()
        null
      } else {
        Reference(
          label = label,
          quad = quad.map(::copyPoint),
          keypoints = keypoints,
          descriptors = descriptors,
          patch = canonicalPatch(gray, quad),
          createdFrame = frameCounter,
          lastMatchedFrame = frameCounter,
        )
      }
    } finally {
      mask.release()
      keypointMat.release()
    }
  }

  private fun matchReference(
    gray: Mat,
    predictedQuad: List<Point>,
    current: Features,
    reference: Reference,
  ): Pair<GalleryMatch?, String> {
    val matchGroups = ArrayList<MatOfDMatch>()
    try {
      matcher.knnMatch(reference.descriptors, current.descriptors, matchGroups, 2)
      val good = ArrayList<DMatch>()
      for (group in matchGroups) {
        val pair = group.toArray()
        if (pair.size == 2 && pair[0].distance < config.ratioTest * pair[1].distance) {
          good.add(pair[0])
        }
      }
      if (good.size < config.minMatches) return null to "matches=${good.size}"

      val referencePoints = good.map { reference.keypoints[it.queryIdx].pt }
      val currentPoints = good.map { current.keypoints[it.trainIdx].pt }
      val source = MatOfPoint2f(*referencePoints.toTypedArray())
      val destination = MatOfPoint2f(*currentPoints.toTypedArray())
      val inlierMask = Mat()
      val homography =
        try {
          Calib3d.findHomography(
            source,
            destination,
            Calib3d.USAC_MAGSAC,
            3.0,
            inlierMask,
          )
        } catch (_: Exception) {
          Mat()
        }
      try {
        if (homography.empty() || inlierMask.empty()) return null to "homography"
        val inlierIndices = inlierIndices(inlierMask, good.size)
        if (inlierIndices.size < config.minInliers) return null to "inliers=${inlierIndices.size}"
        val inlierReference = inlierIndices.map { referencePoints[it] }
        val inlierCurrent = inlierIndices.map { currentPoints[it] }
        val referenceCoverage = featureCoverage(inlierReference, reference.quad)
        if (referenceCoverage < config.minCoverage) {
          return null to "coverage=${format(referenceCoverage)}"
        }
        val mappedQuad = perspectiveTransform(reference.quad, homography)
          ?: return null to "quad_mapping"
        validateQuad(mappedQuad, lastQuad?.let { scaleQuad(it, config.featureScale) }, gray.cols(), gray.rows())
          ?.let { return null to it }
        val similarity = patchSimilarity(reference.patch, gray, mappedQuad)
          ?: return null to "patch"
        if (similarity < config.minPatchSimilarity) return null to "similarity=${format(similarity)}"
        val projected = perspectiveTransform(inlierReference, homography)
          ?: return null to "reprojection"
        val error =
          projected.indices.sumOf { index -> pointDistance(projected[index], inlierCurrent[index]) } /
            projected.size.toDouble()
        if (error > config.maxReprojectionErrorPx) return null to "error=${format(error)}"
        val inlierRatio = inlierIndices.size.toDouble() / good.size.toDouble()
        val currentCoverage = featureCoverage(inlierCurrent, mappedQuad)
        val predictionError =
          mappedQuad.indices.sumOf { pointDistance(mappedQuad[it], predictedQuad[it]) } /
            mappedQuad.size.toDouble()
        val diagonal = hypot(gray.cols().toDouble(), gray.rows().toDouble())
        val predictionErrorRatio = predictionError / diagonal
        val predictedAreaScale = polygonArea(mappedQuad) / max(1.0, polygonArea(predictedQuad))
        val quality =
          similarity +
            0.25 * min(1.0, inlierIndices.size / 20.0) +
            0.15 * inlierRatio -
            0.05 * min(error, 3.0) -
            0.10 * min(1.0, predictionError / (0.10 * diagonal))
        val coefficients = DoubleArray(9)
        homography.get(0, 0, coefficients)
        return GalleryMatch(
          reference = reference,
          quad = mappedQuad,
          homography = coefficients,
          support = inlierIndices.size,
          errorPx = error,
          similarity = similarity,
          inlierRatio = inlierRatio,
          referenceCoverage = referenceCoverage,
          currentCoverage = currentCoverage,
          quality = quality,
          predictionErrorRatio = predictionErrorRatio,
          predictedAreaScale = predictedAreaScale,
        ) to ""
      } finally {
        source.release()
        destination.release()
        inlierMask.release()
        homography.release()
      }
    } finally {
      matchGroups.forEach(MatOfDMatch::release)
    }
  }

  private fun shouldPromote(match: GalleryMatch): Boolean =
    frameCounter - lastPromotionFrame >= max(1, config.promotionIntervalFrames) &&
      consecutiveStableFrames >= config.minStableFramesForPromotion &&
      match.support >= 8 &&
      match.errorPx <= config.maxPromotionErrorPx &&
      match.similarity >= config.minPromotionSimilarity &&
      match.inlierRatio >= config.minPromotionInlierRatio &&
      references.none { reference ->
        normalizedQuadDistance(reference.quad, match.quad) < config.minPromotionViewChangeRatio
      }

  private fun isCorrectionAllowed(match: GalleryMatch, stableTracking: Boolean): Boolean {
    val maxPredictionError =
      if (stableTracking) config.maxHealthyCorrectionRatio else config.maxRecoveryCorrectionRatio
    if (match.predictionErrorRatio > maxPredictionError) return false
    if (stableTracking && abs(match.predictedAreaScale - 1.0) > config.maxHealthyAreaScaleChange) {
      return false
    }
    return true
  }

  private fun normalizedQuadDistance(first: List<Point>, second: List<Point>): Double {
    if (first.size != second.size || first.isEmpty()) return Double.POSITIVE_INFINITY
    val meanDistance =
      first.indices.sumOf { index -> pointDistance(first[index], second[index]) } / first.size
    val xs = first.map(Point::x)
    val ys = first.map(Point::y)
    val diagonal = hypot((xs.maxOrNull() ?: 0.0) - (xs.minOrNull() ?: 0.0), (ys.maxOrNull() ?: 0.0) - (ys.minOrNull() ?: 0.0))
    return if (diagonal > 1e-6) meanDistance / diagonal else Double.POSITIVE_INFINITY
  }

  private fun scaledFrame(grayFrame: Mat): Mat =
    Mat().also {
      Imgproc.resize(
        grayFrame,
        it,
        Size(),
        config.featureScale,
        config.featureScale,
        Imgproc.INTER_AREA,
      )
    }

  private fun scaledMask(mask: Mat?, targetSize: Size): Mat? {
    if (mask == null || mask.empty()) return null
    return Mat().also { Imgproc.resize(mask, it, targetSize, 0.0, 0.0, Imgproc.INTER_NEAREST) }
  }

  private fun buildMask(
    rows: Int,
    cols: Int,
    quad: List<Point>,
    exclusionMask: Mat?,
  ): Mat {
    val mask = Mat.zeros(rows, cols, CvType.CV_8UC1)
    val polygon = MatOfPoint(*quad.map { Point(it.x.coerceIn(0.0, cols - 1.0), it.y.coerceIn(0.0, rows - 1.0)) }.toTypedArray())
    try {
      Imgproc.fillConvexPoly(mask, polygon, Scalar(255.0))
      if (exclusionMask != null && !exclusionMask.empty()) {
        val allowed = Mat()
        try {
          Core.bitwise_not(exclusionMask, allowed)
          Core.bitwise_and(mask, allowed, mask)
        } finally {
          allowed.release()
        }
      }
      return mask
    } finally {
      polygon.release()
    }
  }

  private fun canonicalPatch(gray: Mat, quad: List<Point>): Mat {
    val width = 90
    val height = 180
    val source = MatOfPoint2f(*quad.toTypedArray())
    val target =
      MatOfPoint2f(
        Point(0.0, 0.0),
        Point(width - 1.0, 0.0),
        Point(width - 1.0, height - 1.0),
        Point(0.0, height - 1.0),
      )
    val transform = Imgproc.getPerspectiveTransform(source, target)
    val patch = Mat()
    try {
      Imgproc.warpPerspective(gray, patch, transform, Size(width.toDouble(), height.toDouble()))
      return patch
    } finally {
      source.release()
      target.release()
      transform.release()
    }
  }

  private fun patchSimilarity(referencePatch: Mat, gray: Mat, quad: List<Point>): Double? {
    val patch = canonicalPatch(gray, quad)
    val referenceGradient = Mat()
    val currentGradient = Mat()
    val result = Mat()
    var referenceRoi: Mat? = null
    var currentRoi: Mat? = null
    return try {
      Imgproc.Laplacian(referencePatch, referenceGradient, CvType.CV_32F)
      Imgproc.Laplacian(patch, currentGradient, CvType.CV_32F)
      val marginX = max(2, referencePatch.cols() / 20)
      val marginY = max(2, referencePatch.rows() / 20)
      val roi = Rect(marginX, marginY, referencePatch.cols() - 2 * marginX, referencePatch.rows() - 2 * marginY)
      referenceRoi = referenceGradient.submat(roi)
      currentRoi = currentGradient.submat(roi)
      Imgproc.matchTemplate(referenceRoi, currentRoi, result, Imgproc.TM_CCOEFF_NORMED)
      result.get(0, 0)?.firstOrNull()?.takeIf(Double::isFinite)
    } finally {
      referenceRoi?.release()
      currentRoi?.release()
      patch.release()
      referenceGradient.release()
      currentGradient.release()
      result.release()
    }
  }

  private fun perspectiveTransform(points: List<Point>, homography: Mat): List<Point>? {
    val source = MatOfPoint2f(*points.toTypedArray())
    val destination = MatOfPoint2f()
    return try {
      Core.perspectiveTransform(source, destination, homography)
      destination.toArray().takeIf { it.size == points.size }?.map(::copyPoint)
    } catch (_: Exception) {
      null
    } finally {
      source.release()
      destination.release()
    }
  }

  private fun inlierIndices(mask: Mat, maximum: Int): List<Int> {
    val bytes = ByteArray(mask.total().toInt())
    mask.get(0, 0, bytes)
    return (0 until min(maximum, bytes.size)).filter { bytes[it].toInt() != 0 }
  }

  private fun featureCoverage(points: List<Point>, quad: List<Point>): Double {
    if (points.isEmpty() || quad.size != 4) return 0.0
    val minX = quad.minOf(Point::x)
    val maxX = quad.maxOf(Point::x)
    val minY = quad.minOf(Point::y)
    val maxY = quad.maxOf(Point::y)
    val width = maxX - minX
    val height = maxY - minY
    if (width <= 1e-6 || height <= 1e-6) return 0.0
    val cells = HashSet<Int>()
    for (point in points) {
      if (!isInsideConvexQuad(point, quad)) continue
      val column = (((point.x - minX) / width) * 4.0).toInt().coerceIn(0, 3)
      val row = (((point.y - minY) / height) * 3.0).toInt().coerceIn(0, 2)
      cells.add(row * 4 + column)
    }
    return cells.size / 12.0
  }

  private fun validateQuad(
    quad: List<Point>,
    previous: List<Point>?,
    width: Int,
    height: Int,
  ): String? {
    if (quad.size != 4 || quad.any { !it.x.isFinite() || !it.y.isFinite() }) return "non_finite"
    if (!isConvex(quad)) return "non_convex"
    val area = polygonArea(quad)
    val frameArea = width.toDouble() * height.toDouble()
    if (area < frameArea * 0.0015) return "too_small"
    if (area > frameArea * 2.0) return "too_large"
    if (quad.indices.minOf { pointDistance(quad[it], quad[(it + 1) % quad.size]) } < 4.0) return "short_edge"
    val marginX = width * 0.35
    val marginY = height * 0.35
    if (quad.any { it.x < -marginX || it.x > width + marginX || it.y < -marginY || it.y > height + marginY }) {
      return "outside"
    }
    if (previous != null && previous.size == 4) {
      val previousArea = max(1.0, polygonArea(previous))
      val areaScale = area / previousArea
      if (areaScale !in 0.40..2.50) return "area_jump"
      val maxMotion = quad.indices.maxOf { pointDistance(quad[it], previous[it]) }
      if (maxMotion > hypot(width.toDouble(), height.toDouble()) * 0.42) return "corner_jump"
    }
    return null
  }

  private fun isConvex(quad: List<Point>): Boolean {
    var direction = 0
    for (index in quad.indices) {
      val a = quad[index]
      val b = quad[(index + 1) % quad.size]
      val c = quad[(index + 2) % quad.size]
      val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
      if (abs(cross) < 1e-6) continue
      val nextDirection = if (cross > 0.0) 1 else -1
      if (direction != 0 && direction != nextDirection) return false
      direction = nextDirection
    }
    return direction != 0
  }

  private fun isInsideConvexQuad(point: Point, quad: List<Point>): Boolean {
    var direction = 0
    for (index in quad.indices) {
      val a = quad[index]
      val b = quad[(index + 1) % quad.size]
      val cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)
      if (abs(cross) < 1e-6) continue
      val nextDirection = if (cross > 0.0) 1 else -1
      if (direction != 0 && direction != nextDirection) return false
      direction = nextDirection
    }
    return true
  }

  private fun polygonArea(points: List<Point>): Double {
    var twiceArea = 0.0
    for (index in points.indices) {
      val next = points[(index + 1) % points.size]
      twiceArea += points[index].x * next.y - next.x * points[index].y
    }
    return abs(twiceArea) * 0.5
  }

  private fun expandQuad(quad: List<Point>, scale: Double): List<Point> {
    val centerX = quad.sumOf(Point::x) / quad.size.toDouble()
    val centerY = quad.sumOf(Point::y) / quad.size.toDouble()
    return quad.map { Point(centerX + (it.x - centerX) * scale, centerY + (it.y - centerY) * scale) }
  }

  private fun scaleQuad(quad: List<Point>, scale: Double): List<Point> =
    quad.map { Point(it.x * scale, it.y * scale) }

  private fun fullResolutionHomography(scaled: DoubleArray): DoubleArray {
    val scale = config.featureScale
    return doubleArrayOf(
      scaled[0], scaled[1], scaled[2] / scale,
      scaled[3], scaled[4], scaled[5] / scale,
      scaled[6] * scale, scaled[7] * scale, scaled[8],
    )
  }

  private fun galleryFailure(failures: List<String>): String {
    val detail = failures.joinToString("|").take(180)
    return if (detail.isEmpty()) "gallery:no_match" else "gallery:no_match:$detail"
  }

  private fun combineReasons(first: String?, second: String): String =
    if (first.isNullOrBlank()) second else "$first|$second"

  private fun clearReferences() {
    references.forEach(Reference::release)
    references.clear()
  }

  private fun pointDistance(first: Point, second: Point): Double =
    hypot(first.x - second.x, first.y - second.y)

  private fun copyPoint(point: Point): Point = Point(point.x, point.y)

  private fun format(value: Double): String = String.format(java.util.Locale.US, "%.3f", value)

}
