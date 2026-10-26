package com.inkmarc

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class PlaneMotionGeometryTest {
  private val intrinsics = CameraIntrinsics(fx = 910.0, fy = 905.0, cx = 640.0, cy = 360.0)
  private val referencePoints =
    listOf(
      Point(180.0, 170.0),
      Point(420.0, 150.0),
      Point(690.0, 165.0),
      Point(980.0, 190.0),
      Point(220.0, 360.0),
      Point(490.0, 345.0),
      Point(760.0, 350.0),
      Point(1080.0, 375.0),
      Point(260.0, 570.0),
      Point(550.0, 555.0),
      Point(830.0, 545.0),
      Point(1050.0, 565.0),
    )

  @Test
  fun recoversTranslationOverHeightWithKnownRotation() {
    val rotation = rotationY(Math.toRadians(7.0))
    val normal = normalized(doubleArrayOf(0.05, 0.78, 0.62))
    val expectedTerm = doubleArrayOf(0.11, -0.045, 0.16)
    val currentPoints = project(referencePoints, rotation, normal, expectedTerm)

    val estimate =
      PlaneMotionGeometry.fitPlaneTerm(
        referencePoints,
        currentPoints,
        rotation,
        normal,
        intrinsics,
      )

    assertNotNull(estimate)
    assertEquals(referencePoints.size, estimate!!.inlierCount)
    assertTrue(
      "rmse=${estimate.rmsePx}, term=${estimate.planeTerm.joinToString()}",
      estimate.rmsePx < 1e-6,
    )
    expectedTerm.indices.forEach { index ->
      assertEquals(expectedTerm[index], estimate.planeTerm[index], 1e-7)
    }
  }

  @Test
  fun rejectsGrossPointOutlierDuringPlaneFit() {
    val rotation = rotationY(Math.toRadians(-5.0))
    val normal = normalized(doubleArrayOf(-0.04, 0.72, 0.69))
    val expectedTerm = doubleArrayOf(-0.09, 0.035, -0.08)
    val currentPoints = project(referencePoints, rotation, normal, expectedTerm).toMutableList()
    currentPoints[5] = Point(currentPoints[5].x + 310.0, currentPoints[5].y - 240.0)

    val estimate =
      PlaneMotionGeometry.fitPlaneTerm(
        referencePoints,
        currentPoints,
        rotation,
        normal,
        intrinsics,
      )

    assertNotNull(estimate)
    assertTrue(
      "inliers=${estimate!!.inlierCount}, rmse=${estimate.rmsePx}, term=${estimate.planeTerm.joinToString()}",
      estimate.inlierCount < referencePoints.size,
    )
    assertTrue(estimate.inlierCount >= referencePoints.size - 2)
    assertTrue(estimate.rmsePx < 0.1)
    assertTrue(5 !in estimate.inlierIndices)
    assertTrue(estimate.reprojectionErrorsPx[5] > 100.0)
    expectedTerm.indices.forEach { index ->
      assertEquals(expectedTerm[index], estimate.planeTerm[index], 1e-4)
    }
  }

  @Test
  fun temporalPriorResolvesComparableCompetingConsensus() {
    val rotation = rotationY(Math.toRadians(2.0))
    val normal = normalized(doubleArrayOf(0.03, 0.76, 0.65))
    val predictedTerm = doubleArrayOf(0.03, -0.01, 0.04)
    val competingTerm = doubleArrayOf(0.25, -0.15, 0.25)
    val predictedPoints = project(referencePoints, rotation, normal, predictedTerm)
    val competingPoints = project(referencePoints, rotation, normal, competingTerm)
    val ambiguousPoints =
      referencePoints.indices.map { index ->
        if (index < 7) competingPoints[index] else predictedPoints[index]
      }

    val estimate =
      PlaneMotionGeometry.fitPlaneTerm(
        referencePoints,
        ambiguousPoints,
        rotation,
        normal,
        intrinsics,
        priorPlaneTerm = predictedTerm,
      )

    assertNotNull(estimate)
    assertEquals(5, estimate!!.inlierCount)
    predictedTerm.indices.forEach { index ->
      assertEquals(predictedTerm[index], estimate.planeTerm[index], 1e-4)
    }
  }

  @Test
  fun reliabilityWeightsPreventDynamicMajorityFromWinning() {
    val rotation = rotationY(Math.toRadians(3.0))
    val normal = normalized(doubleArrayOf(0.02, 0.77, 0.64))
    val surfaceTerm = doubleArrayOf(0.045, -0.018, 0.06)
    val handTerm = doubleArrayOf(0.29, -0.16, 0.21)
    val surfacePoints = project(referencePoints, rotation, normal, surfaceTerm)
    val handPoints = project(referencePoints, rotation, normal, handTerm)
    val observations =
      referencePoints.indices.map { index ->
        if (index < 7) handPoints[index] else surfacePoints[index]
      }
    val weights = List(referencePoints.size) { index -> if (index < 7) 0.03 else 1.0 }

    val estimate =
      PlaneMotionGeometry.fitPlaneTerm(
        referencePoints,
        observations,
        rotation,
        normal,
        intrinsics,
        pointWeights = weights,
      )

    assertNotNull(estimate)
    assertEquals(5, estimate!!.inlierCount)
    assertTrue(estimate.effectiveInlierCount > 4.9)
    assertTrue(estimate.effectiveSampleCount < 5.3)
    surfaceTerm.indices.forEach { index ->
      assertEquals(surfaceTerm[index], estimate.planeTerm[index], 1e-4)
    }
  }

  @Test
  fun referenceHomographiesComposeIntoFrameToFrameMotion() {
    val normal = normalized(doubleArrayOf(0.02, 0.8, 0.6))
    val first =
      PlaneMotionGeometry.planeHomography(
        rotationY(Math.toRadians(3.0)),
        normal,
        doubleArrayOf(0.03, -0.01, 0.04),
        intrinsics,
      )!!
    val second =
      PlaneMotionGeometry.planeHomography(
        rotationY(Math.toRadians(6.0)),
        normal,
        doubleArrayOf(0.07, -0.015, 0.09),
        intrinsics,
      )!!
    val frameToFrame =
      PlaneMotionGeometry.multiply3x3(second, PlaneMotionGeometry.invert3x3(first)!!)
    val firstPoints = PlaneMotionGeometry.mapPoints(referencePoints, first)!!
    val directSecondPoints = PlaneMotionGeometry.mapPoints(referencePoints, second)!!
    val composedSecondPoints = PlaneMotionGeometry.mapPoints(firstPoints, frameToFrame)!!

    directSecondPoints.indices.forEach { index ->
      assertEquals(directSecondPoints[index].x, composedSecondPoints[index].x, 1e-6)
      assertEquals(directSecondPoints[index].y, composedSecondPoints[index].y, 1e-6)
    }
  }

  @Test
  fun recoversTranslationWithPerRowRollingShutterRotations() {
    val normal = normalized(doubleArrayOf(0.02, 0.79, 0.61))
    val expectedTerm = doubleArrayOf(0.075, -0.025, 0.11)
    val rowRotations =
      referencePoints.map { point ->
        val rowFraction = point.y / 720.0
        rotationY(Math.toRadians(-2.0 + 5.0 * rowFraction))
      }
    val currentPoints =
      referencePoints.indices.map { index ->
        project(listOf(referencePoints[index]), rowRotations[index], normal, expectedTerm).single()
      }

    val estimate =
      PlaneMotionGeometry.fitPlaneTerm(
        referencePoints,
        currentPoints,
        rotationY(Math.toRadians(0.5)),
        normal,
        intrinsics,
        perPointRelativeRotations = rowRotations,
      )

    assertNotNull(estimate)
    assertEquals(referencePoints.size, estimate!!.inlierCount)
    assertTrue(estimate.rmsePx < 1e-6)
    expectedTerm.indices.forEach { index ->
      assertEquals(expectedTerm[index], estimate.planeTerm[index], 1e-7)
    }
  }

  private fun project(
    points: List<Point>,
    rotation: DoubleArray,
    normal: DoubleArray,
    term: DoubleArray,
  ): List<Point> {
    val homography = PlaneMotionGeometry.planeHomography(rotation, normal, term, intrinsics)!!
    return PlaneMotionGeometry.mapPoints(points, homography)!!
  }

  private fun rotationY(angle: Double): DoubleArray =
    doubleArrayOf(
      cos(angle), 0.0, sin(angle),
      0.0, 1.0, 0.0,
      -sin(angle), 0.0, cos(angle),
    )

  private fun normalized(vector: DoubleArray): DoubleArray {
    val length = sqrt(vector.sumOf { it * it })
    return DoubleArray(vector.size) { vector[it] / length }
  }
}
