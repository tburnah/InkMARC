package com.inkmarc

import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sqrt

/** Geometry for a horizontal plane observed by a calibrated moving camera. */
internal object PlaneMotionGeometry {
  private const val MIN_PLANE_RAY_DOT = 1e-4
  private const val MIN_SOLVER_PIVOT = 1e-9
  private const val MIN_ROBUST_THRESHOLD_PX = 2.5
  private const val MAX_ROBUST_THRESHOLD_PX = 18.0
  private const val CONSENSUS_THRESHOLD_PX = 10.0
  private const val MAX_CONSENSUS_SAMPLES = 12
  private const val COMPARABLE_SUPPORT_DIFFERENCE = 2
  private const val PRIOR_XY_SCALE = 0.08
  private const val PRIOR_DEPTH_SCALE = 0.05
  private const val PRIOR_ERROR_WEIGHT_PX = 1.5

  data class Estimate(
    val planeTerm: DoubleArray,
    val rmsePx: Double,
    val inlierCount: Int,
    val sampleCount: Int,
    val inlierIndices: List<Int>,
    val reprojectionErrorsPx: List<Double>,
    val effectiveInlierCount: Double,
    val effectiveSampleCount: Double,
  )

  /**
   * Fits the three observable translation-over-plane-distance parameters in
   * H = K (R + u n^T) K^-1, with R and the reference-frame plane normal fixed.
   */
  fun fitPlaneTerm(
    referencePoints: List<Point>,
    currentPoints: List<Point>,
    relativeRotation: DoubleArray,
    referencePlaneNormal: DoubleArray,
    intrinsics: CameraIntrinsics,
    priorPlaneTerm: DoubleArray? = null,
    perPointRelativeRotations: List<DoubleArray>? = null,
    pointWeights: List<Double>? = null,
  ): Estimate? {
    if (
      referencePoints.size != currentPoints.size ||
        referencePoints.size < 2 ||
        relativeRotation.size != 9 ||
        referencePlaneNormal.size != 3 ||
        (pointWeights != null && pointWeights.size != referencePoints.size) ||
        (perPointRelativeRotations != null &&
          (perPointRelativeRotations.size != referencePoints.size ||
            perPointRelativeRotations.any { it.size != 9 }))
    ) {
      return null
    }
    val samples =
      referencePoints.indices.mapNotNull { index ->
        buildSample(
          index,
          referencePoints[index],
          currentPoints[index],
          perPointRelativeRotations?.get(index) ?: relativeRotation,
          referencePlaneNormal,
          intrinsics,
          pointWeights?.get(index) ?: 1.0,
        )
      }
    if (samples.size < 2) return null

    var retained =
      initialConsensus(
        samples,
        referencePlaneNormal,
        intrinsics,
        priorPlaneTerm,
      )
    var term = solve(retained) ?: return null
    repeat(2) {
      val errors =
        reprojectionErrors(retained, term, referencePlaneNormal, intrinsics)
      val finiteErrors = errors.filter(Double::isFinite)
      if (finiteErrors.size < 2) return null
      val median = median(finiteErrors)
      val threshold = (median * 2.5 + 1.0).coerceIn(MIN_ROBUST_THRESHOLD_PX, MAX_ROBUST_THRESHOLD_PX)
      val filtered =
        retained.filterIndexed { index, _ ->
          val error = errors[index]
          error.isFinite() && error <= threshold
        }
      if (filtered.size < 2 || filtered.size == retained.size) return@repeat
      retained = filtered
      term = solve(retained) ?: return null
    }

    val squaredError =
      reprojectionErrors(retained, term, referencePlaneNormal, intrinsics)
        .sumOf { error -> error * error }
    val rmse = sqrt(squaredError / retained.size)
    if (!rmse.isFinite()) return null
    val allSampleErrors =
      reprojectionErrors(samples, term, referencePlaneNormal, intrinsics)
    val inputErrors = MutableList(referencePoints.size) { Double.POSITIVE_INFINITY }
    samples.forEachIndexed { sampleIndex, sample ->
      inputErrors[sample.sourceIndex] = allSampleErrors[sampleIndex]
    }
    return Estimate(
      planeTerm = term.copyOf(),
      rmsePx = rmse,
      inlierCount = retained.size,
      sampleCount = samples.size,
      inlierIndices = retained.map(LinearSample::sourceIndex).sorted(),
      reprojectionErrorsPx = inputErrors,
      effectiveInlierCount = retained.sumOf(LinearSample::baseWeight),
      effectiveSampleCount = samples.sumOf(LinearSample::baseWeight),
    )
  }

  fun planeReprojectionErrors(
    referencePoints: List<Point>,
    currentPoints: List<Point>,
    relativeRotation: DoubleArray,
    referencePlaneNormal: DoubleArray,
    planeTerm: DoubleArray,
    intrinsics: CameraIntrinsics,
    perPointRelativeRotations: List<DoubleArray>? = null,
  ): List<Double> {
    if (
      referencePoints.size != currentPoints.size ||
        (perPointRelativeRotations != null && perPointRelativeRotations.size != referencePoints.size)
    ) return emptyList()
    return referencePoints.indices.map { index ->
      val rotation = perPointRelativeRotations?.get(index) ?: relativeRotation
      val homography =
        planeHomography(rotation, referencePlaneNormal, planeTerm, intrinsics)
          ?: return@map Double.POSITIVE_INFINITY
      val projected = mapPoint(referencePoints[index], homography)
      if (projected == null) {
        Double.POSITIVE_INFINITY
      } else {
        hypot(projected.x - currentPoints[index].x, projected.y - currentPoints[index].y)
      }
    }
  }

  fun planeHomography(
    relativeRotation: DoubleArray,
    referencePlaneNormal: DoubleArray,
    planeTerm: DoubleArray,
    intrinsics: CameraIntrinsics,
  ): DoubleArray? {
    if (
      relativeRotation.size != 9 ||
        referencePlaneNormal.size != 3 ||
        planeTerm.size != 3 ||
        listOf(intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy).any { !it.isFinite() } ||
        intrinsics.fx <= 0.0 ||
        intrinsics.fy <= 0.0
    ) {
      return null
    }
    val normalizedHomography = relativeRotation.copyOf()
    for (row in 0..2) {
      for (column in 0..2) {
        normalizedHomography[row * 3 + column] +=
          planeTerm[row] * referencePlaneNormal[column]
      }
    }
    val cameraMatrix =
      doubleArrayOf(
        intrinsics.fx, 0.0, intrinsics.cx,
        0.0, intrinsics.fy, intrinsics.cy,
        0.0, 0.0, 1.0,
      )
    val imageHomography =
      multiply3x3(
        cameraMatrix,
        multiply3x3(normalizedHomography, invertCameraMatrix(intrinsics)),
      )
    if (imageHomography.any { !it.isFinite() }) return null
    val scale = imageHomography[8]
    if (abs(scale) > 1e-9) {
      for (index in imageHomography.indices) imageHomography[index] /= scale
    }
    return imageHomography
  }

  fun mapPoints(points: List<Point>, homography: DoubleArray): List<Point>? {
    if (homography.size != 9) return null
    val mapped = ArrayList<Point>(points.size)
    for (point in points) {
      val w = homography[6] * point.x + homography[7] * point.y + homography[8]
      if (!w.isFinite() || abs(w) <= 1e-8) return null
      val x = (homography[0] * point.x + homography[1] * point.y + homography[2]) / w
      val y = (homography[3] * point.x + homography[4] * point.y + homography[5]) / w
      if (!x.isFinite() || !y.isFinite()) return null
      mapped.add(Point(x, y))
    }
    return mapped
  }

  fun invert3x3(matrix: DoubleArray): DoubleArray? {
    if (matrix.size != 9) return null
    val a = matrix[0]
    val b = matrix[1]
    val c = matrix[2]
    val d = matrix[3]
    val e = matrix[4]
    val f = matrix[5]
    val g = matrix[6]
    val h = matrix[7]
    val i = matrix[8]
    val determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if (!determinant.isFinite() || abs(determinant) <= 1e-10) return null
    val inverseDeterminant = 1.0 / determinant
    return doubleArrayOf(
      (e * i - f * h) * inverseDeterminant,
      (c * h - b * i) * inverseDeterminant,
      (b * f - c * e) * inverseDeterminant,
      (f * g - d * i) * inverseDeterminant,
      (a * i - c * g) * inverseDeterminant,
      (c * d - a * f) * inverseDeterminant,
      (d * h - e * g) * inverseDeterminant,
      (b * g - a * h) * inverseDeterminant,
      (a * e - b * d) * inverseDeterminant,
    )
  }

  fun multiply3x3(first: DoubleArray, second: DoubleArray): DoubleArray {
    require(first.size == 9 && second.size == 9)
    val result = DoubleArray(9)
    for (row in 0..2) {
      for (column in 0..2) {
        var value = 0.0
        for (index in 0..2) value += first[row * 3 + index] * second[index * 3 + column]
        result[row * 3 + column] = value
      }
    }
    return result
  }

  private data class LinearSample(
    val sourceIndex: Int,
    val firstRow: DoubleArray,
    val firstValue: Double,
    val secondRow: DoubleArray,
    val secondValue: Double,
    val referencePoint: Point,
    val currentPoint: Point,
    val relativeRotation: DoubleArray,
    val baseWeight: Double,
  )

  private fun buildSample(
    sourceIndex: Int,
    referencePoint: Point,
    currentPoint: Point,
    relativeRotation: DoubleArray,
    referencePlaneNormal: DoubleArray,
    intrinsics: CameraIntrinsics,
    pointWeight: Double,
  ): LinearSample? {
    if (
      !referencePoint.x.isFinite() ||
        !referencePoint.y.isFinite() ||
        !currentPoint.x.isFinite() ||
      !currentPoint.y.isFinite() ||
      !pointWeight.isFinite() ||
      pointWeight <= 0.01
    ) {
      return null
    }
    val referenceRay =
      doubleArrayOf(
        (referencePoint.x - intrinsics.cx) / intrinsics.fx,
        (referencePoint.y - intrinsics.cy) / intrinsics.fy,
        1.0,
      )
    val currentX = (currentPoint.x - intrinsics.cx) / intrinsics.fx
    val currentY = (currentPoint.y - intrinsics.cy) / intrinsics.fy
    val planeDot = dot(referencePlaneNormal, referenceRay)
    if (!planeDot.isFinite() || abs(planeDot) < MIN_PLANE_RAY_DOT) return null
    val rotatedRay = multiply3x3Vector(relativeRotation, referenceRay)
    return LinearSample(
      sourceIndex = sourceIndex,
      firstRow = doubleArrayOf(planeDot, 0.0, -currentX * planeDot),
      firstValue = currentX * rotatedRay[2] - rotatedRay[0],
      secondRow = doubleArrayOf(0.0, planeDot, -currentY * planeDot),
      secondValue = currentY * rotatedRay[2] - rotatedRay[1],
      referencePoint = Point(referencePoint.x, referencePoint.y),
      currentPoint = Point(currentPoint.x, currentPoint.y),
      relativeRotation = relativeRotation.copyOf(),
      baseWeight = pointWeight.coerceIn(0.01, 1.0),
    )
  }

  private fun solve(samples: List<LinearSample>): DoubleArray? {
    val normalMatrix = Array(3) { DoubleArray(3) }
    val normalVector = DoubleArray(3)
    for (sample in samples) {
      accumulateNormalEquation(
        normalMatrix,
        normalVector,
        sample.firstRow,
        sample.firstValue,
        sample.baseWeight,
      )
      accumulateNormalEquation(
        normalMatrix,
        normalVector,
        sample.secondRow,
        sample.secondValue,
        sample.baseWeight,
      )
    }
    return solve3x3(normalMatrix, normalVector)
  }

  private fun initialConsensus(
    samples: List<LinearSample>,
    referencePlaneNormal: DoubleArray,
    intrinsics: CameraIntrinsics,
    priorPlaneTerm: DoubleArray?,
  ): List<LinearSample> {
    if (samples.size <= 4) return samples
    val hypothesisSamples =
      if (samples.size <= MAX_CONSENSUS_SAMPLES) {
        samples
      } else {
        List(MAX_CONSENSUS_SAMPLES) { index ->
          samples[index * (samples.lastIndex) / (MAX_CONSENSUS_SAMPLES - 1)]
        }.distinct()
      }
    var bestInliers: List<LinearSample>? = null
    var bestSupportWeight = -1.0
    var bestObjective = Double.POSITIVE_INFINITY
    for (firstIndex in 0 until hypothesisSamples.lastIndex) {
      for (secondIndex in firstIndex + 1 until hypothesisSamples.size) {
        val term = solve(listOf(hypothesisSamples[firstIndex], hypothesisSamples[secondIndex]))
          ?: continue
        val errors =
          reprojectionErrors(samples, term, referencePlaneNormal, intrinsics)
        val inliers =
          samples.filterIndexed { index, _ ->
            val error = errors[index]
            error.isFinite() && error <= CONSENSUS_THRESHOLD_PX
          }
        if (inliers.size < 2) continue
        val totalWeight = samples.sumOf(LinearSample::baseWeight).coerceAtLeast(1e-6)
        val truncatedMeanError =
          samples.indices.sumOf { index ->
            samples[index].baseWeight * min(errors[index], CONSENSUS_THRESHOLD_PX * 2.0)
          } / totalWeight
        val objective = truncatedMeanError + priorPenalty(term, priorPlaneTerm)
        val supportWeight = inliers.sumOf(LinearSample::baseWeight)
        val hasClearlyBetterSupport = supportWeight > bestSupportWeight + COMPARABLE_SUPPORT_DIFFERENCE
        val hasComparableSupport = abs(supportWeight - bestSupportWeight) <= COMPARABLE_SUPPORT_DIFFERENCE
        if (
          bestInliers == null ||
            hasClearlyBetterSupport ||
            (hasComparableSupport && objective < bestObjective)
        ) {
          bestInliers = inliers
          bestSupportWeight = supportWeight
          bestObjective = objective
        }
      }
    }
    return bestInliers?.takeIf { it.size >= 4 } ?: samples
  }

  private fun priorPenalty(planeTerm: DoubleArray, priorPlaneTerm: DoubleArray?): Double {
    if (priorPlaneTerm == null || priorPlaneTerm.size != 3 || priorPlaneTerm.any { !it.isFinite() }) {
      return 0.0
    }
    val xyInnovation =
      hypot(
        (planeTerm[0] - priorPlaneTerm[0]) / PRIOR_XY_SCALE,
        (planeTerm[1] - priorPlaneTerm[1]) / PRIOR_XY_SCALE,
      )
    val depthInnovation = abs(planeTerm[2] - priorPlaneTerm[2]) / PRIOR_DEPTH_SCALE
    return PRIOR_ERROR_WEIGHT_PX * hypot(xyInnovation, depthInnovation)
  }

  private fun accumulateNormalEquation(
    matrix: Array<DoubleArray>,
    vector: DoubleArray,
    row: DoubleArray,
    value: Double,
    weight: Double,
  ) {
    for (r in 0..2) {
      vector[r] += weight * row[r] * value
      for (c in 0..2) matrix[r][c] += weight * row[r] * row[c]
    }
  }

  private fun solve3x3(matrix: Array<DoubleArray>, vector: DoubleArray): DoubleArray? {
    val augmented = Array(3) { row -> DoubleArray(4) { column -> if (column < 3) matrix[row][column] else vector[row] } }
    for (column in 0..2) {
      var pivotRow = column
      for (row in column + 1..2) {
        if (abs(augmented[row][column]) > abs(augmented[pivotRow][column])) pivotRow = row
      }
      val pivot = augmented[pivotRow][column]
      if (!pivot.isFinite() || abs(pivot) < MIN_SOLVER_PIVOT) return null
      if (pivotRow != column) {
        val temporary = augmented[column]
        augmented[column] = augmented[pivotRow]
        augmented[pivotRow] = temporary
      }
      val pivotValue = augmented[column][column]
      for (entry in column..3) augmented[column][entry] /= pivotValue
      for (row in 0..2) {
        if (row == column) continue
        val factor = augmented[row][column]
        for (entry in column..3) augmented[row][entry] -= factor * augmented[column][entry]
      }
    }
    return DoubleArray(3) { augmented[it][3] }.takeIf { solution -> solution.all(Double::isFinite) }
  }

  private fun reprojectionErrors(
    samples: List<LinearSample>,
    planeTerm: DoubleArray,
    referencePlaneNormal: DoubleArray,
    intrinsics: CameraIntrinsics,
  ): List<Double> {
    return samples.map { sample ->
      val homography =
        planeHomography(sample.relativeRotation, referencePlaneNormal, planeTerm, intrinsics)
          ?: return@map Double.POSITIVE_INFINITY
      val point = sample.referencePoint
      val w = homography[6] * point.x + homography[7] * point.y + homography[8]
      if (!w.isFinite() || abs(w) <= 1e-8) {
        Double.POSITIVE_INFINITY
      } else {
        val x = (homography[0] * point.x + homography[1] * point.y + homography[2]) / w
        val y = (homography[3] * point.x + homography[4] * point.y + homography[5]) / w
        val dx = x - sample.currentPoint.x
        val dy = y - sample.currentPoint.y
        sqrt(dx * dx + dy * dy)
      }
    }
  }

  private fun mapPoint(point: Point, homography: DoubleArray): Point? {
    val w = homography[6] * point.x + homography[7] * point.y + homography[8]
    if (!w.isFinite() || abs(w) <= 1e-8) return null
    val x = (homography[0] * point.x + homography[1] * point.y + homography[2]) / w
    val y = (homography[3] * point.x + homography[4] * point.y + homography[5]) / w
    return if (x.isFinite() && y.isFinite()) Point(x, y) else null
  }

  private fun multiply3x3Vector(matrix: DoubleArray, vector: DoubleArray): DoubleArray =
    doubleArrayOf(
      matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
      matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
      matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    )

  private fun invertCameraMatrix(intrinsics: CameraIntrinsics): DoubleArray =
    doubleArrayOf(
      1.0 / intrinsics.fx, 0.0, -intrinsics.cx / intrinsics.fx,
      0.0, 1.0 / intrinsics.fy, -intrinsics.cy / intrinsics.fy,
      0.0, 0.0, 1.0,
    )

  private fun dot(first: DoubleArray, second: DoubleArray): Double =
    first[0] * second[0] + first[1] * second[1] + first[2] * second[2]

  private fun median(values: List<Double>): Double {
    val sorted = values.sorted()
    val middle = sorted.size / 2
    return if (sorted.size % 2 == 0) (sorted[middle - 1] + sorted[middle]) * 0.5 else sorted[middle]
  }
}
