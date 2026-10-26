package com.inkmarc.fallback

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Minimal sparse surface tracker extracted from the app's larger SensorAnalyzer pipeline.
 *
 * Input frames must already be grayscale OpenCV Mats. Seed with a reference surface quad, then
 * call [update] on each new frame to get a fitted 2D transform and the mapped quad.
 */
class SurfaceTrackerCore(
    private val config: Config = Config()
) : AutoCloseable {

    data class Config(
        val maxFeatures: Int = 60,
        val maxDetectedCorners: Int = 240,
        val useFullFrameFeatureSupport: Boolean = false,
        val allowOffscreenMappedQuad: Boolean = false,
        val minSeedFeatures: Int = 8,
        val minFeatureSeparationPx: Double = 5.0,
        val detectQualityLevel: Double = 0.003,
        val detectMinDistancePx: Double = 6.0,
        val maxLkError: Double = 40.0,
        val maxForwardBackwardErrorPx: Double = 7.0,
        val minTrackReliability: Double = 0.12,
        val initialTrackReliability: Double = 0.65,
        val reliabilityDecayAlpha: Double = 0.48,
        val reliabilityRecoveryAlpha: Double = 0.14,
        val enableOcclusionTrackMemory: Boolean = false,
        val maxDormantTracks: Int = 32,
        val maxDormantFrames: Int = 30,
        val maxReacquisitionAttemptsPerFrame: Int = 8,
        val appearancePatchRadiusPx: Int = 7,
        val appearanceSearchRadiusPx: Int = 30,
        val minReacquisitionScore: Double = 0.74,
        val forwardBackwardCheckEveryNFrames: Int = 3,
        val stableFeatureAgeFrames: Int = 8,
        val minTracksForTransform: Int = 4,
        val minSurfaceModelTracks: Int = 8,
        val minSurfaceModelInliers: Int = 6,
        val sparseMotionMinTracks: Int = 4,
        val sparseMotionMinInliers: Int = 3,
        val sparseMotionMedianErrorPx: Double = 5.0,
        val sparseMotionInlierThresholdPx: Double = 6.0,
        val minHomographySupportTracks: Int = 10,
        val minAffineSupportTracks: Int = 6,
        val minSimilaritySupportTracks: Int = 4,
        val minHomographySpanRatio: Double = 0.18,
        val minHomographyAreaRatio: Double = 0.03,
        val minAffineSpanRatio: Double = 0.12,
        val minAffineAreaRatio: Double = 0.015,
        val minSimilaritySpanRatio: Double = 0.08,
        val minSimilarityAreaRatio: Double = 0.008,
        val transformRansacThresholdPx: Double = 3.0,
        val surfaceModelRansacThresholdPx: Double = 4.0,
        val maxMeanReprojErrorPx: Double = 6.0,
        val maxTranslationMedianErrorPx: Double = 5.0,
        val replenishEveryNFrames: Int = 2,
        val minConfidenceForReplenishment: Float = 0.72f,
        val minHighConfidenceFramesForReplenishment: Int = 6,
        val maxHeldFrames: Int = 3,
        val maxRejectedFramesBeforeLost: Int = 10,
        val featureCoverageColumns: Int = 4,
        val featureCoverageRows: Int = 3,
        val minHomographyFeatureCoverage: Double = 0.33,
        val minAffineFeatureCoverage: Double = 0.25,
        val minSimilarityFeatureCoverage: Double = 0.16,
        val minTranslationFeatureCoverage: Double = 0.08,
        val minSupportSpanWithinQuad: Double = 0.22,
        val minSimilaritySupportExtent: Double = 0.18,
        val minMappedQuadAreaRatio: Double = 0.0015,
        val maxMappedQuadAreaRatio: Double = 2.0,
        val minMappedEdgeLengthPx: Double = 12.0,
        val minCornerAngleDegrees: Double = 7.0,
        val maxOppositeEdgeRatio: Double = 8.0,
        val minFrameAreaScale: Double = 0.48,
        val maxFrameAreaScale: Double = 2.1,
        val minFrameEdgeScale: Double = 0.45,
        val maxFrameEdgeScale: Double = 2.2,
        val maxFrameCornerMotionRatio: Double = 0.38,
        val maxCornerOutsideFrameRatio: Double = 0.30,
        val maxNormalizedHomographyCondition: Double = 5_000.0
    )

    data class SeedResult(
        val success: Boolean,
        val trackCount: Int,
        val mappedQuad: List<Point>?,
        val message: String
    )

    data class TrackerResult(
        val state: State,
        val activeTrackCount: Int,
        val candidateTrackCount: Int,
        val stableTrackCount: Int,
        val inlierCount: Int,
        val inlierRatio: Float,
        val confidence: Float,
        val transform: SurfaceTransform?,
        val mappedQuad: List<Point>?,
        val meanReprojectionErrorPx: Double?,
        val referenceFeatureCoverage: Float,
        val currentFeatureCoverage: Float,
        val transformCondition: Double?,
        val heldLastGoodTransform: Boolean,
        val rejectionReason: String?,
        val referenceInlierPoints: List<Point> = emptyList(),
        val currentInlierPoints: List<Point> = emptyList(),
        val inlierTrackIds: List<Int> = emptyList(),
        val inlierTrackWeights: List<Float> = emptyList(),
        val effectivePlaneSupport: Float? = null,
        val weightedPlaneCoverage: Float? = null,
        val reacquiredTrackCount: Int = 0,
        val referenceWasReseeded: Boolean = false,
        val maskRejectedTrackCount: Int = 0
    )

    enum class State {
        IDLE,
        TRACKING,
        INSUFFICIENT_FEATURES,
        LOST
    }

    enum class MotionModel {
        HOMOGRAPHY,
        AFFINE,
        SIMILARITY,
        TRANSLATION
    }

    sealed interface SurfaceTransform {
        val model: MotionModel
        fun map(point: Point): Point?
        fun invert(): SurfaceTransform?
    }

    data class HomographyTransform(
        private val coefficients: DoubleArray
    ) : SurfaceTransform {
        init {
            require(coefficients.size == 9) { "Homography requires 9 coefficients" }
        }

        override val model: MotionModel = MotionModel.HOMOGRAPHY

        fun coefficientsCopy(): DoubleArray = coefficients.copyOf()

        override fun map(point: Point): Point? {
            val w = coefficients[6] * point.x + coefficients[7] * point.y + coefficients[8]
            if (!w.isFinite() || abs(w) < 1e-9) return null
            val x = (coefficients[0] * point.x + coefficients[1] * point.y + coefficients[2]) / w
            val y = (coefficients[3] * point.x + coefficients[4] * point.y + coefficients[5]) / w
            return if (x.isFinite() && y.isFinite()) Point(x, y) else null
        }

        override fun invert(): SurfaceTransform? {
            val src = Mat(3, 3, CvType.CV_64F)
            val inverse = Mat()
            return try {
                src.put(0, 0, *coefficients)
                if (Core.invert(src, inverse) == 0.0) return null
                val values = DoubleArray(9)
                inverse.get(0, 0, values)
                HomographyTransform(values)
            } finally {
                src.release()
                inverse.release()
            }
        }
    }

    data class AffineTransform(
        private val coefficients: DoubleArray,
        override val model: MotionModel = MotionModel.AFFINE
    ) : SurfaceTransform {
        init {
            require(coefficients.size == 6) { "Affine transform requires 6 coefficients" }
        }

        fun coefficientsCopy(): DoubleArray = coefficients.copyOf()

        override fun map(point: Point): Point? {
            val x = coefficients[0] * point.x + coefficients[1] * point.y + coefficients[2]
            val y = coefficients[3] * point.x + coefficients[4] * point.y + coefficients[5]
            return if (x.isFinite() && y.isFinite()) Point(x, y) else null
        }

        override fun invert(): SurfaceTransform? {
            val det = coefficients[0] * coefficients[4] - coefficients[1] * coefficients[3]
            if (!det.isFinite() || abs(det) < 1e-9) return null
            val inv00 = coefficients[4] / det
            val inv01 = -coefficients[1] / det
            val inv10 = -coefficients[3] / det
            val inv11 = coefficients[0] / det
            val inv02 = -(inv00 * coefficients[2] + inv01 * coefficients[5])
            val inv12 = -(inv10 * coefficients[2] + inv11 * coefficients[5])
            return AffineTransform(
                doubleArrayOf(inv00, inv01, inv02, inv10, inv11, inv12),
                model = model
            )
        }
    }

    data class TranslationTransform(
        val dx: Double,
        val dy: Double
    ) : SurfaceTransform {
        override val model: MotionModel = MotionModel.TRANSLATION

        override fun map(point: Point): Point = Point(point.x + dx, point.y + dy)

        override fun invert(): SurfaceTransform = TranslationTransform(-dx, -dy)
    }

    private data class Track(
        val id: Int,
        val referencePoint: Point,
        var currentPoint: Point,
        var age: Int = 0,
        var surfaceInlier: Boolean = false,
        var staticProbability: Double = 0.65,
        var lastPriorResidualPx: Double = 0.0,
        var appearancePatch: Mat? = null,
        var dormantFrames: Int = 0,
    )

    private data class FlowCandidate(
        val track: Track,
        val prevPoint: Point,
        val nextPoint: Point,
        val measurementReliability: Double,
    )

    private data class FlowTrackingOutcome(
        val candidates: List<FlowCandidate>,
        val maskRejectedTrackCount: Int,
        val occludedTracks: List<Track>,
    )

    private data class TransformEstimate(
        val transform: SurfaceTransform,
        val inlierTrackIds: Set<Int>,
        val meanReprojError: Double
    )

    private data class ValidatedEstimate(
        val estimate: TransformEstimate,
        val mappedQuad: List<Point>,
        val referenceCoverage: Double,
        val currentCoverage: Double,
        val transformCondition: Double?,
        val modelScore: Double
    )

    private data class EstimationOutcome(
        val accepted: ValidatedEstimate? = null,
        val referenceCoverage: Double = 0.0,
        val currentCoverage: Double = 0.0,
        val transformCondition: Double? = null,
        val rejectionReason: String? = null
    )

    private data class ValidationResult(
        val accepted: Boolean,
        val mappedQuad: List<Point>? = null,
        val referenceCoverage: Double = 0.0,
        val currentCoverage: Double = 0.0,
        val transformCondition: Double? = null,
        val reason: String? = null
    )

    private val activeTracks = ArrayList<Track>(config.maxFeatures)
    private val dormantTracks = ArrayList<Track>(config.maxDormantTracks)
    private val flowWinSize = Size(41.0, 41.0)
    private val flowTermCriteria = TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 20, 0.03)
    private var prevGray: Mat? = null
    private var referenceQuad: List<Point> = emptyList()
    private var referenceSupportQuad: List<Point> = emptyList()
    private var referenceFrameWidth = 0
    private var referenceFrameHeight = 0
    private var nextTrackId = 1
    private var frameCounter = 0
    private var consecutiveHighConfidenceFrames = 0
    private var consecutiveRejectedFrames = 0
    private var heldFrameCount = 0
    private var lastAcceptedTransform: SurfaceTransform? = null
    private var lastAcceptedMappedQuad: List<Point>? = null

    fun seedReference(grayFrame: Mat, referenceQuad: List<Point>, exclusionMask: Mat? = null): SeedResult {
        if (!isUsableGray(grayFrame)) {
            reset()
            return SeedResult(false, 0, null, "Expected non-empty single-channel grayscale Mat")
        }
        if (referenceQuad.size < 4) {
            reset()
            return SeedResult(false, 0, null, "Reference quad must contain at least 4 points")
        }

        reset()
        this.referenceQuad = referenceQuad.map(::copyPoint)
        referenceFrameWidth = grayFrame.cols()
        referenceFrameHeight = grayFrame.rows()
        referenceSupportQuad =
            if (config.useFullFrameFeatureSupport) {
                fullFrameQuad(referenceFrameWidth, referenceFrameHeight)
            } else {
                this.referenceQuad.map(::copyPoint)
            }

        val quadMask = buildQuadMask(grayFrame.rows(), grayFrame.cols(), referenceSupportQuad)
        val points = try {
            findNewPoints(
                grayFrame,
                emptyList(),
                config.maxFeatures,
                quadMask,
                exclusionMask,
                referenceSupportQuad
            )
        } finally {
            quadMask.release()
        }

        if (points.size < config.minSeedFeatures) {
            updatePrevGray(grayFrame)
            return SeedResult(
                success = false,
                trackCount = points.size,
                mappedQuad = this.referenceQuad.map(::copyPoint),
                message = "Too few seed features inside the reference quad"
            )
        }

        for (point in points) {
            activeTracks.add(
                Track(
                    id = nextTrackId++,
                    referencePoint = copyPoint(point),
                    currentPoint = copyPoint(point),
                    staticProbability = config.initialTrackReliability,
                    appearancePatch =
                        if (config.enableOcclusionTrackMemory) extractAppearancePatch(grayFrame, point) else null,
                )
            )
        }
        lastAcceptedTransform = TranslationTransform(0.0, 0.0)
        lastAcceptedMappedQuad = this.referenceQuad.map(::copyPoint)
        updatePrevGray(grayFrame)
        return SeedResult(true, activeTracks.size, this.referenceQuad.map(::copyPoint), "Tracker seeded")
    }

    fun update(
        grayFrame: Mat,
        exclusionMask: Mat? = null,
        initialFlowTransform: SurfaceTransform? = null
    ): TrackerResult {
        if (!isUsableGray(grayFrame)) {
            return emptyTrackerResult(State.IDLE, "invalid_gray_frame")
        }

        frameCounter += 1
        val previousGray = prevGray
        if (previousGray == null || referenceQuad.isEmpty()) {
            updatePrevGray(grayFrame)
            return emptyTrackerResult(State.IDLE, "tracker_not_seeded", activeTracks.size)
        }

        if (config.enableOcclusionTrackMemory) advanceDormantTracks(initialFlowTransform)
        val previousActiveTracks = activeTracks.toList()
        val flowOutcome =
            trackFrameToFrame(previousGray, grayFrame, exclusionMask, initialFlowTransform)
        val survivors = filterFrameToFrameConsistent(flowOutcome.candidates)
        if (config.enableOcclusionTrackMemory) addDormantTracks(flowOutcome.occludedTracks)
        val retainedIds =
            (survivors.map(Track::id) +
                if (config.enableOcclusionTrackMemory) flowOutcome.occludedTracks.map(Track::id) else emptyList())
                .toHashSet()
        previousActiveTracks
            .filter { track -> track.id !in retainedIds }
            .forEach(::releaseTrack)
        activeTracks.clear()
        activeTracks.addAll(survivors)
        val reacquiredTrackCount =
            if (config.enableOcclusionTrackMemory) reacquireDormantTracks(grayFrame, exclusionMask) else 0

        val candidateTrackCount = activeTracks.size
        val outcome = estimateSurfaceTransform(activeTracks)
        val accepted = outcome.accepted
        if (accepted != null) {
            val estimate = accepted.estimate
            val inlierIds = estimate.inlierTrackIds
            val inlierCount = inlierIds.size
            val inlierRatio =
                if (candidateTrackCount > 0) inlierCount.toFloat() / candidateTrackCount.toFloat() else 0f
            val confidence =
                computeConfidence(candidateTrackCount, inlierCount, estimate.meanReprojError)
            val rejectedTracks = activeTracks.filter { track -> !inlierIds.contains(track.id) }
            activeTracks.removeAll(rejectedTracks.toSet())
            rejectedTracks.forEach(::releaseTrack)
            for (track in activeTracks) {
                track.surfaceInlier = true
            }
            // Preserve the actual measurements before replenishment adds synthetic reference
            // locations. The plane-constrained estimator uses these pairs directly instead of
            // trying to recover camera translation from the rendered quad.
            val referenceInlierPoints = activeTracks.map { track -> copyPoint(track.referencePoint) }
            val currentInlierPoints = activeTracks.map { track -> copyPoint(track.currentPoint) }
            val inlierTrackIds = activeTracks.map(Track::id)
            val inlierTrackWeights = activeTracks.map { track -> track.staticProbability.toFloat() }
            consecutiveRejectedFrames = 0
            heldFrameCount = 0
            consecutiveHighConfidenceFrames =
                if (confidence >= config.minConfidenceForReplenishment) {
                    consecutiveHighConfidenceFrames + 1
                } else {
                    0
                }
            lastAcceptedTransform = estimate.transform
            lastAcceptedMappedQuad = accepted.mappedQuad.map(::copyPoint)

            if (consecutiveHighConfidenceFrames >= config.minHighConfidenceFramesForReplenishment &&
                frameCounter % config.replenishEveryNFrames == 0
            ) {
                replenishTracks(grayFrame, exclusionMask, estimate.transform)
            }

            updatePrevGray(grayFrame)
            return TrackerResult(
                state = State.TRACKING,
                activeTrackCount = activeTracks.size,
                candidateTrackCount = candidateTrackCount,
                stableTrackCount = stableTrackCount(),
                inlierCount = inlierCount,
                inlierRatio = inlierRatio,
                confidence = confidence,
                transform = estimate.transform,
                mappedQuad = accepted.mappedQuad.map(::copyPoint),
                meanReprojectionErrorPx = estimate.meanReprojError,
                referenceFeatureCoverage = accepted.referenceCoverage.toFloat(),
                currentFeatureCoverage = accepted.currentCoverage.toFloat(),
                transformCondition = accepted.transformCondition,
                heldLastGoodTransform = false,
                rejectionReason = outcome.rejectionReason,
                referenceInlierPoints = referenceInlierPoints,
                currentInlierPoints = currentInlierPoints,
                inlierTrackIds = inlierTrackIds,
                inlierTrackWeights = inlierTrackWeights,
                reacquiredTrackCount = reacquiredTrackCount,
                maskRejectedTrackCount = flowOutcome.maskRejectedTrackCount
            )
        }

        for (track in activeTracks) {
            track.surfaceInlier = false
        }
        consecutiveHighConfidenceFrames = 0
        consecutiveRejectedFrames += 1
        val canHold =
            lastAcceptedTransform != null &&
                lastAcceptedMappedQuad != null &&
                heldFrameCount < config.maxHeldFrames &&
                consecutiveRejectedFrames < config.maxRejectedFramesBeforeLost
        if (canHold) heldFrameCount += 1
        val state =
            if (activeTracks.size < config.minTracksForTransform ||
                consecutiveRejectedFrames >= config.maxRejectedFramesBeforeLost
            ) {
                State.LOST
            } else {
                State.INSUFFICIENT_FEATURES
            }
        updatePrevGray(grayFrame)
        return TrackerResult(
            state = state,
            activeTrackCount = activeTracks.size,
            candidateTrackCount = candidateTrackCount,
            stableTrackCount = stableTrackCount(),
            inlierCount = 0,
            inlierRatio = 0f,
            confidence = 0f,
            transform = if (canHold) lastAcceptedTransform else null,
            mappedQuad = if (canHold) lastAcceptedMappedQuad?.map(::copyPoint) else null,
            meanReprojectionErrorPx = null,
            referenceFeatureCoverage = outcome.referenceCoverage.toFloat(),
            currentFeatureCoverage = outcome.currentCoverage.toFloat(),
            transformCondition = outcome.transformCondition,
            heldLastGoodTransform = canHold,
            rejectionReason = outcome.rejectionReason ?: "no_valid_transform",
            reacquiredTrackCount = reacquiredTrackCount,
            maskRejectedTrackCount = flowOutcome.maskRejectedTrackCount
        )
    }

    private fun emptyTrackerResult(
        state: State,
        reason: String,
        activeTrackCount: Int = 0
    ): TrackerResult =
        TrackerResult(
            state = state,
            activeTrackCount = activeTrackCount,
            candidateTrackCount = activeTrackCount,
            stableTrackCount = stableTrackCount(),
            inlierCount = 0,
            inlierRatio = 0f,
            confidence = 0f,
            transform = null,
            mappedQuad = null,
            meanReprojectionErrorPx = null,
            referenceFeatureCoverage = 0f,
            currentFeatureCoverage = 0f,
            transformCondition = null,
            heldLastGoodTransform = false,
            rejectionReason = reason
        )

    fun reset() {
        activeTracks.forEach(::releaseTrack)
        dormantTracks.forEach(::releaseTrack)
        activeTracks.clear()
        dormantTracks.clear()
        referenceQuad = emptyList()
        referenceSupportQuad = emptyList()
        referenceFrameWidth = 0
        referenceFrameHeight = 0
        nextTrackId = 1
        frameCounter = 0
        consecutiveHighConfidenceFrames = 0
        consecutiveRejectedFrames = 0
        heldFrameCount = 0
        lastAcceptedTransform = null
        lastAcceptedMappedQuad = null
        prevGray?.release()
        prevGray = null
    }

    override fun close() {
        reset()
    }

    private fun stableTrackCount(): Int {
        return activeTracks.count { it.age >= config.stableFeatureAgeFrames }
    }

    private fun trackFrameToFrame(
        previousGray: Mat,
        grayFrame: Mat,
        exclusionMask: Mat?,
        initialFlowTransform: SurfaceTransform?
    ): FlowTrackingOutcome {
        if (activeTracks.isEmpty()) return FlowTrackingOutcome(emptyList(), 0, emptyList())

        val prevPoints = MatOfPoint2f(*activeTracks.map { it.currentPoint }.toTypedArray())
        val initialPoints =
            initialFlowTransform?.let { transform ->
                activeTracks.map { transform.map(it.currentPoint) ?: copyPoint(it.currentPoint) }
            }
        val nextPoints =
            if (initialPoints != null) MatOfPoint2f(*initialPoints.toTypedArray()) else MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        val backPoints = MatOfPoint2f()
        val backStatus = MatOfByte()
        val backErr = MatOfFloat()
        val performBackwardCheck =
            config.forwardBackwardCheckEveryNFrames <= 1 ||
                frameCounter % config.forwardBackwardCheckEveryNFrames == 0
        return try {
            Video.calcOpticalFlowPyrLK(
                previousGray,
                grayFrame,
                prevPoints,
                nextPoints,
                status,
                err,
                flowWinSize,
                5,
                flowTermCriteria,
                if (initialPoints != null) Video.OPTFLOW_USE_INITIAL_FLOW else 0,
                1e-4
            )
            if (performBackwardCheck) {
                backPoints.fromArray(*activeTracks.map { it.currentPoint }.toTypedArray())
                Video.calcOpticalFlowPyrLK(
                    grayFrame,
                    previousGray,
                    nextPoints,
                    backPoints,
                    backStatus,
                    backErr,
                    flowWinSize,
                    5,
                    flowTermCriteria,
                    Video.OPTFLOW_USE_INITIAL_FLOW,
                    1e-4
                )
            }

            val nextArray = nextPoints.toArray()
            val prevArray = prevPoints.toArray()
            val statusArray = status.toArray()
            val errArray = err.toArray()
            val backArray = if (performBackwardCheck) backPoints.toArray() else emptyArray()
            val backStatusArray = if (performBackwardCheck) backStatus.toArray() else byteArrayOf()
            val priorResiduals = DoubleArray(statusArray.size) { Double.NaN }
            val forwardBackwardErrors = DoubleArray(statusArray.size) { Double.NaN }
            for (i in statusArray.indices) {
                if (statusArray[i].toInt() != 1) continue
                val nextPoint = nextArray.getOrNull(i) ?: continue
                initialPoints?.getOrNull(i)?.let { predicted ->
                    priorResiduals[i] = hypot(nextPoint.x - predicted.x, nextPoint.y - predicted.y)
                }
                if (performBackwardCheck && backStatusArray.getOrNull(i)?.toInt() == 1) {
                    val previousPoint = prevArray.getOrNull(i) ?: continue
                    val backPoint = backArray.getOrNull(i) ?: continue
                    forwardBackwardErrors[i] = hypot(backPoint.x - previousPoint.x, backPoint.y - previousPoint.y)
                }
            }
            val finitePriorResiduals = priorResiduals.filter(Double::isFinite)
            val priorMedian = if (finitePriorResiduals.isEmpty()) 0.0 else median(finitePriorResiduals)
            val priorMad =
                if (finitePriorResiduals.isEmpty()) {
                    0.0
                } else {
                    median(finitePriorResiduals.map { residual -> abs(residual - priorMedian) })
                }
            val priorScale = max(2.5, 1.0 + 2.5 * 1.4826 * priorMad)
            val candidates = ArrayList<FlowCandidate>(activeTracks.size)
            val occludedTracks = ArrayList<Track>()
            var maskRejectedTrackCount = 0
            for (i in statusArray.indices) {
                if (statusArray[i].toInt() != 1) continue
                if (performBackwardCheck && backStatusArray.getOrNull(i)?.toInt() != 1) continue
                if ((errArray.getOrNull(i)?.toDouble() ?: 0.0) > config.maxLkError) continue
                val prevPoint = prevArray.getOrNull(i) ?: continue
                val nextPoint = nextArray.getOrNull(i) ?: continue
                if (performBackwardCheck) {
                    val forwardBackwardError = forwardBackwardErrors[i]
                    if (!forwardBackwardError.isFinite() ||
                        forwardBackwardError > config.maxForwardBackwardErrorPx
                    ) continue
                }
                if (!isInside(nextPoint, grayFrame.cols(), grayFrame.rows())) continue
                val masked = isMasked(prevPoint, exclusionMask) || isMasked(nextPoint, exclusionMask)
                if (masked) {
                    maskRejectedTrackCount += 1
                    val track = activeTracks[i]
                    track.staticProbability = min(track.staticProbability, 0.06)
                    track.currentPoint = copyPoint(nextPoint)
                    track.dormantFrames = 0
                    occludedTracks.add(track)
                    continue
                }
                val lkError = errArray.getOrNull(i)?.toDouble() ?: config.maxLkError
                val lkReliability = cauchyReliability(lkError, 18.0)
                val forwardBackwardReliability =
                    if (performBackwardCheck) {
                        cauchyReliability(forwardBackwardErrors[i], 2.5)
                    } else {
                        1.0
                    }
                val priorResidual = priorResiduals[i]
                val priorReliability =
                    if (priorResidual.isFinite()) {
                        val excess = max(0.0, priorResidual - priorMedian - 1.0)
                        cauchyReliability(excess, priorScale)
                    } else {
                        1.0
                    }
                val evidence =
                    sqrt(
                        (lkReliability * forwardBackwardReliability * priorReliability)
                            .coerceIn(0.0, 1.0)
                    )
                val track = activeTracks[i]
                val alpha =
                    if (evidence < track.staticProbability) {
                        config.reliabilityDecayAlpha
                    } else {
                        config.reliabilityRecoveryAlpha
                    }
                track.staticProbability =
                    (track.staticProbability + alpha * (evidence - track.staticProbability))
                        .coerceIn(0.02, 1.0)
                track.lastPriorResidualPx = if (priorResidual.isFinite()) priorResidual else 0.0
                if (track.staticProbability < config.minTrackReliability) continue
                candidates.add(FlowCandidate(track, prevPoint, nextPoint, track.staticProbability))
            }
            FlowTrackingOutcome(candidates, maskRejectedTrackCount, occludedTracks)
        } finally {
            prevPoints.release()
            nextPoints.release()
            status.release()
            err.release()
            backPoints.release()
            backStatus.release()
            backErr.release()
        }
    }

    private fun filterFrameToFrameConsistent(candidates: List<FlowCandidate>): List<Track> {
        val (accepted, usedSurfaceModel) = when {
            candidates.size >= config.minSurfaceModelTracks -> {
                val filtered = filterCandidatesByHomography(candidates)
                if (filtered.size >= config.minSurfaceModelInliers) {
                    filtered to true
                } else {
                    val sparseFiltered = filterCandidatesByTranslationConsistency(candidates)
                    if (sparseFiltered.size >= config.sparseMotionMinInliers) {
                        sparseFiltered to true
                    } else {
                        candidates to false
                    }
                }
            }
            candidates.size >= config.sparseMotionMinTracks -> {
                val sparseFiltered = filterCandidatesByTranslationConsistency(candidates)
                if (sparseFiltered.size >= config.sparseMotionMinInliers) {
                    sparseFiltered to true
                } else {
                    candidates to false
                }
            }
            else -> candidates to false
        }

        val survivors = ArrayList<Track>(accepted.size)
        for (candidate in accepted) {
            val track = candidate.track
            track.currentPoint = copyPoint(candidate.nextPoint)
            track.age += 1
            track.surfaceInlier = usedSurfaceModel
            survivors.add(track)
        }
        return survivors
    }

    private fun filterCandidatesByHomography(candidates: List<FlowCandidate>): List<FlowCandidate> {
        val src = MatOfPoint2f(*candidates.map { it.prevPoint }.toTypedArray())
        val dst = MatOfPoint2f(*candidates.map { it.nextPoint }.toTypedArray())
        val mask = Mat()
        val homography = try {
            Calib3d.findHomography(src, dst, Calib3d.RANSAC, config.surfaceModelRansacThresholdPx, mask)
        } catch (_: Exception) {
            src.release()
            dst.release()
            mask.release()
            return emptyList()
        }

        return try {
            if (homography.empty() || mask.empty()) return emptyList()
            val maskBytes = ByteArray(mask.total().toInt())
            mask.get(0, 0, maskBytes)
            val filtered = ArrayList<FlowCandidate>(candidates.size)
            for (i in candidates.indices) {
                if (maskBytes.getOrNull(i)?.toInt() != 0) {
                    filtered.add(candidates[i])
                }
            }
            filtered
        } finally {
            src.release()
            dst.release()
            mask.release()
            homography.release()
        }
    }

    private fun filterCandidatesByTranslationConsistency(candidates: List<FlowCandidate>): List<FlowCandidate> {
        if (candidates.size < config.sparseMotionMinTracks) return emptyList()
        val dxs = ArrayList<Double>(candidates.size)
        val dys = ArrayList<Double>(candidates.size)
        for (candidate in candidates) {
            dxs.add(candidate.nextPoint.x - candidate.prevPoint.x)
            dys.add(candidate.nextPoint.y - candidate.prevPoint.y)
        }
        val medianDx = median(dxs)
        val medianDy = median(dys)
        val residuals = ArrayList<Double>(candidates.size)
        for (candidate in candidates) {
            val errX = (candidate.prevPoint.x + medianDx) - candidate.nextPoint.x
            val errY = (candidate.prevPoint.y + medianDy) - candidate.nextPoint.y
            residuals.add(sqrt(errX * errX + errY * errY))
        }
        if (median(residuals) > config.sparseMotionMedianErrorPx) return emptyList()
        val filtered = ArrayList<FlowCandidate>(candidates.size)
        for (i in candidates.indices) {
            if (residuals[i] <= config.sparseMotionInlierThresholdPx) {
                filtered.add(candidates[i])
            }
        }
        return filtered
    }

    private fun estimateSurfaceTransform(tracks: List<Track>): EstimationOutcome {
        if (tracks.size < config.minTracksForTransform) {
            return EstimationOutcome(rejectionReason = "too_few_tracks:${tracks.size}")
        }
        val srcPoints = tracks.map { it.referencePoint }
        val dstPoints = tracks.map { it.currentPoint }
        val validated = ArrayList<ValidatedEstimate>(4)
        val rejections = ArrayList<String>(4)
        var diagnosticReferenceCoverage = 0.0
        var diagnosticCurrentCoverage = 0.0
        var diagnosticCondition: Double? = null

        fun consider(modelName: String, estimate: TransformEstimate?, minimumCoverage: Double) {
            if (estimate == null) {
                rejections.add("$modelName:fit_failed")
                return
            }
            val validation = validateEstimate(tracks, estimate, minimumCoverage)
            diagnosticReferenceCoverage = max(diagnosticReferenceCoverage, validation.referenceCoverage)
            diagnosticCurrentCoverage = max(diagnosticCurrentCoverage, validation.currentCoverage)
            validation.transformCondition?.let { diagnosticCondition = it }
            if (!validation.accepted || validation.mappedQuad == null) {
                rejections.add("$modelName:${validation.reason ?: "geometry_rejected"}")
                return
            }
            val inlierRatio = estimate.inlierTrackIds.size.toDouble() / max(1, tracks.size).toDouble()
            val complexityPenalty =
                when (estimate.transform.model) {
                    MotionModel.TRANSLATION -> 0.0
                    MotionModel.SIMILARITY -> 0.10
                    MotionModel.AFFINE -> 0.20
                    MotionModel.HOMOGRAPHY -> 0.35
                }
            validated.add(
                ValidatedEstimate(
                    estimate = estimate,
                    mappedQuad = validation.mappedQuad,
                    referenceCoverage = validation.referenceCoverage,
                    currentCoverage = validation.currentCoverage,
                    transformCondition = validation.transformCondition,
                    modelScore = estimate.meanReprojError + complexityPenalty + (1.0 - inlierRatio) * 3.0
                )
            )
        }

        if (tracks.size >= config.minHomographySupportTracks &&
            hasSufficientSupportSpread(
                srcPoints,
                referenceFrameWidth,
                referenceFrameHeight,
                config.minHomographySpanRatio,
                config.minHomographyAreaRatio
            )
        ) {
            consider(
                "homography",
                estimateHomography(tracks, srcPoints, dstPoints),
                config.minHomographyFeatureCoverage
            )
        } else {
            rejections.add("homography:insufficient_support")
        }
        if (tracks.size >= config.minAffineSupportTracks &&
            hasSufficientSupportSpread(
                srcPoints,
                referenceFrameWidth,
                referenceFrameHeight,
                config.minAffineSpanRatio,
                config.minAffineAreaRatio
            )
        ) {
            consider("affine", estimateAffine(tracks, srcPoints, dstPoints), config.minAffineFeatureCoverage)
        } else {
            rejections.add("affine:insufficient_support")
        }
        if (tracks.size >= config.minSimilaritySupportTracks &&
            hasSufficientSupportSpread(
                srcPoints,
                referenceFrameWidth,
                referenceFrameHeight,
                config.minSimilaritySpanRatio,
                config.minSimilarityAreaRatio
            )
        ) {
            consider(
                "similarity",
                estimateSimilarity(tracks, srcPoints, dstPoints),
                config.minSimilarityFeatureCoverage
            )
        } else {
            rejections.add("similarity:insufficient_support")
        }
        consider(
            "translation",
            estimateTranslation(tracks, srcPoints, dstPoints),
            config.minTranslationFeatureCoverage
        )

        val accepted = validated.minByOrNull { it.modelScore }
        val decisionNote =
            buildList {
                addAll(rejections)
                if (validated.size > 1 && accepted != null) {
                    add("selected:${accepted.estimate.transform.model.name.lowercase()}")
                }
            }.takeIf { it.isNotEmpty() }?.joinToString("|")
        return EstimationOutcome(
            accepted = accepted,
            referenceCoverage = accepted?.referenceCoverage ?: diagnosticReferenceCoverage,
            currentCoverage = accepted?.currentCoverage ?: diagnosticCurrentCoverage,
            transformCondition = accepted?.transformCondition ?: diagnosticCondition,
            rejectionReason = decisionNote
        )
    }

    private fun validateEstimate(
        tracks: List<Track>,
        estimate: TransformEstimate,
        minimumCoverage: Double
    ): ValidationResult {
        val mappedQuad = mapQuad(referenceQuad, estimate.transform)
            ?: return ValidationResult(false, reason = "quad_mapping_failed")
        validateQuadGeometry(mappedQuad)?.let { reason ->
            return ValidationResult(false, mappedQuad = mappedQuad, reason = reason)
        }

        val inlierTracks = tracks.filter { estimate.inlierTrackIds.contains(it.id) }
        val referencePoints = inlierTracks.map { it.referencePoint }
        val currentPoints = inlierTracks.map { it.currentPoint }
        val currentSupportQuad =
            if (config.useFullFrameFeatureSupport) {
                fullFrameQuad(referenceFrameWidth, referenceFrameHeight)
            } else {
                mapQuad(referenceSupportQuad, estimate.transform)
                    ?: return ValidationResult(false, mappedQuad = mappedQuad, reason = "support_mapping_failed")
            }
        val currentCoverage = computeFeatureCoverage(currentPoints, currentSupportQuad)
        val referenceCoverage =
            if (config.useFullFrameFeatureSupport) {
                currentCoverage
            } else {
                computeFeatureCoverage(referencePoints, referenceSupportQuad)
            }
        if (referenceCoverage < minimumCoverage) {
            return ValidationResult(
                false,
                mappedQuad,
                referenceCoverage,
                currentCoverage,
                reason = "reference_coverage"
            )
        }
        if (currentCoverage < minimumCoverage) {
            return ValidationResult(
                false,
                mappedQuad,
                referenceCoverage,
                currentCoverage,
                reason = "current_coverage"
            )
        }
        val hasRequiredSupport =
            when (estimate.transform.model) {
                MotionModel.HOMOGRAPHY, MotionModel.AFFINE ->
                    if (config.useFullFrameFeatureSupport) {
                        hasSupportSpanWithinQuad(currentPoints, currentSupportQuad)
                    } else {
                        hasSupportSpanWithinQuad(referencePoints, referenceSupportQuad) &&
                            hasSupportSpanWithinQuad(currentPoints, currentSupportQuad)
                    }
                MotionModel.SIMILARITY ->
                    if (config.useFullFrameFeatureSupport) {
                        hasSimilaritySupportExtent(currentPoints, currentSupportQuad)
                    } else {
                        hasSimilaritySupportExtent(referencePoints, referenceSupportQuad) &&
                            hasSimilaritySupportExtent(currentPoints, currentSupportQuad)
                    }
                MotionModel.TRANSLATION -> true
            }
        if (!hasRequiredSupport) {
            return ValidationResult(
                false,
                mappedQuad,
                referenceCoverage,
                currentCoverage,
                reason = "support_span"
            )
        }

        val condition =
            (estimate.transform as? HomographyTransform)?.let {
                normalizedHomographyCondition(it.coefficientsCopy())
            }
        if (condition != null &&
            (!condition.isFinite() || condition > config.maxNormalizedHomographyCondition)
        ) {
            return ValidationResult(
                false,
                mappedQuad,
                referenceCoverage,
                currentCoverage,
                condition,
                "homography_condition"
            )
        }
        return ValidationResult(
            true,
            mappedQuad,
            referenceCoverage,
            currentCoverage,
            condition
        )
    }

    private fun estimateHomography(
        tracks: List<Track>,
        srcPoints: List<Point>,
        dstPoints: List<Point>
    ): TransformEstimate? {
        val src = MatOfPoint2f(*srcPoints.toTypedArray())
        val dst = MatOfPoint2f(*dstPoints.toTypedArray())
        val mask = Mat()
        val homography = try {
            Calib3d.findHomography(src, dst, Calib3d.RANSAC, config.transformRansacThresholdPx, mask)
        } catch (_: Exception) {
            src.release()
            dst.release()
            mask.release()
            return null
        }

        return try {
            if (homography.empty() || mask.empty()) return null
            val inlierIndices = collectInlierIndices(mask, tracks.size)
            if (inlierIndices.size < config.minHomographySupportTracks) return null
            val inlierSrc = inlierIndices.map { srcPoints[it] }
            val inlierDst = inlierIndices.map { dstPoints[it] }
            val meanError = computePerspectiveReprojectionError(inlierSrc, inlierDst, homography)
            if (meanError > config.maxMeanReprojErrorPx) return null
            val values = DoubleArray(9)
            homography.get(0, 0, values)
            TransformEstimate(
                transform = HomographyTransform(values),
                inlierTrackIds = inlierIndices.mapTo(LinkedHashSet()) { tracks[it].id },
                meanReprojError = meanError
            )
        } finally {
            src.release()
            dst.release()
            mask.release()
            homography.release()
        }
    }

    private fun estimateAffine(
        tracks: List<Track>,
        srcPoints: List<Point>,
        dstPoints: List<Point>
    ): TransformEstimate? {
        val src = MatOfPoint2f(*srcPoints.toTypedArray())
        val dst = MatOfPoint2f(*dstPoints.toTypedArray())
        val mask = Mat()
        val affine = try {
            Calib3d.estimateAffine2D(
                src,
                dst,
                mask,
                Calib3d.RANSAC,
                config.transformRansacThresholdPx,
                2000,
                0.99,
                10
            )
        } catch (_: Exception) {
            src.release()
            dst.release()
            mask.release()
            return null
        }

        return try {
            if (affine.empty() || mask.empty()) return null
            val inlierIndices = collectInlierIndices(mask, tracks.size)
            if (inlierIndices.size < config.minAffineSupportTracks) return null
            val inlierSrc = inlierIndices.map { srcPoints[it] }
            val inlierDst = inlierIndices.map { dstPoints[it] }
            val meanError = computeAffineReprojectionError(inlierSrc, inlierDst, affine)
            if (meanError > config.maxMeanReprojErrorPx) return null
            val values = DoubleArray(6)
            affine.get(0, 0, values)
            TransformEstimate(
                transform = AffineTransform(values, MotionModel.AFFINE),
                inlierTrackIds = inlierIndices.mapTo(LinkedHashSet()) { tracks[it].id },
                meanReprojError = meanError
            )
        } finally {
            src.release()
            dst.release()
            mask.release()
            affine.release()
        }
    }

    private fun estimateSimilarity(
        tracks: List<Track>,
        srcPoints: List<Point>,
        dstPoints: List<Point>
    ): TransformEstimate? {
        val src = MatOfPoint2f(*srcPoints.toTypedArray())
        val dst = MatOfPoint2f(*dstPoints.toTypedArray())
        val mask = Mat()
        val similarity = try {
            Calib3d.estimateAffinePartial2D(
                src,
                dst,
                mask,
                Calib3d.RANSAC,
                config.transformRansacThresholdPx,
                2000,
                0.99,
                10
            )
        } catch (_: Exception) {
            src.release()
            dst.release()
            mask.release()
            return null
        }

        return try {
            if (similarity.empty() || mask.empty()) return null
            val inlierIndices = collectInlierIndices(mask, tracks.size)
            if (inlierIndices.size < config.minSimilaritySupportTracks) return null
            val inlierSrc = inlierIndices.map { srcPoints[it] }
            val inlierDst = inlierIndices.map { dstPoints[it] }
            val meanError = computeAffineReprojectionError(inlierSrc, inlierDst, similarity)
            if (meanError > config.maxMeanReprojErrorPx) return null
            val values = DoubleArray(6)
            similarity.get(0, 0, values)
            TransformEstimate(
                transform = AffineTransform(values, MotionModel.SIMILARITY),
                inlierTrackIds = inlierIndices.mapTo(LinkedHashSet()) { tracks[it].id },
                meanReprojError = meanError
            )
        } finally {
            src.release()
            dst.release()
            mask.release()
            similarity.release()
        }
    }

    private fun estimateTranslation(
        tracks: List<Track>,
        srcPoints: List<Point>,
        dstPoints: List<Point>
    ): TransformEstimate? {
        if (tracks.size < config.minSimilaritySupportTracks) return null
        val dxs = ArrayList<Double>(tracks.size)
        val dys = ArrayList<Double>(tracks.size)
        for (i in tracks.indices) {
            dxs.add(dstPoints[i].x - srcPoints[i].x)
            dys.add(dstPoints[i].y - srcPoints[i].y)
        }
        val medianDx = median(dxs)
        val medianDy = median(dys)
        val residuals = ArrayList<Double>(tracks.size)
        for (i in tracks.indices) {
            val errX = (srcPoints[i].x + medianDx) - dstPoints[i].x
            val errY = (srcPoints[i].y + medianDy) - dstPoints[i].y
            residuals.add(sqrt(errX * errX + errY * errY))
        }
        val medianError = median(residuals)
        if (medianError > config.maxTranslationMedianErrorPx) return null
        val inlierIndices = residuals.indices.filter { residuals[it] <= config.maxTranslationMedianErrorPx }
        if (inlierIndices.size < config.minSimilaritySupportTracks) return null
        return TransformEstimate(
            transform = TranslationTransform(medianDx, medianDy),
            inlierTrackIds = inlierIndices.mapTo(LinkedHashSet()) { tracks[it].id },
            meanReprojError = medianError
        )
    }

    private fun replenishTracks(grayFrame: Mat, exclusionMask: Mat?, transform: SurfaceTransform) {
        val needed = config.maxFeatures - activeTracks.size
        if (needed <= 0) return
        val inverse = transform.invert() ?: return
        val currentSupportQuad =
            if (config.useFullFrameFeatureSupport) {
                fullFrameQuad(grayFrame.cols(), grayFrame.rows())
            } else {
                mapQuad(referenceSupportQuad, transform) ?: return
            }
        val quadMask = buildQuadMask(grayFrame.rows(), grayFrame.cols(), currentSupportQuad)
        val extras = try {
            findNewPoints(
                grayFrame,
                activeTracks.map { it.currentPoint },
                needed,
                quadMask,
                exclusionMask,
                currentSupportQuad
            )
        } finally {
            quadMask.release()
        }

        for (point in extras) {
            val referencePoint = inverse.map(point) ?: continue
            if (config.useFullFrameFeatureSupport) {
                val marginX = referenceFrameWidth * 4.0
                val marginY = referenceFrameHeight * 4.0
                if (referencePoint.x !in -marginX..marginX + referenceFrameWidth ||
                    referencePoint.y !in -marginY..marginY + referenceFrameHeight
                ) {
                    continue
                }
            } else if (!isInside(referencePoint, referenceFrameWidth, referenceFrameHeight)) {
                continue
            }
            activeTracks.add(
                Track(
                    id = nextTrackId++,
                    referencePoint = referencePoint,
                    currentPoint = copyPoint(point),
                    staticProbability = config.initialTrackReliability,
                    appearancePatch =
                        if (config.enableOcclusionTrackMemory) extractAppearancePatch(grayFrame, point) else null,
                )
            )
        }
    }

    private fun advanceDormantTracks(initialFlowTransform: SurfaceTransform?) {
        val expired = ArrayList<Track>()
        for (track in dormantTracks) {
            track.dormantFrames += 1
            initialFlowTransform?.map(track.currentPoint)?.let { predicted ->
                track.currentPoint = predicted
            }
            if (track.dormantFrames > config.maxDormantFrames) expired.add(track)
        }
        dormantTracks.removeAll(expired.toSet())
        expired.forEach(::releaseTrack)
    }

    private fun addDormantTracks(tracks: List<Track>) {
        for (track in tracks) {
            if (dormantTracks.none { dormant -> dormant.id == track.id }) dormantTracks.add(track)
        }
        if (dormantTracks.size <= config.maxDormantTracks) return
        val toRemove =
            dormantTracks
                .sortedWith(compareByDescending<Track> { it.dormantFrames }.thenBy { it.staticProbability })
                .take(dormantTracks.size - config.maxDormantTracks)
        dormantTracks.removeAll(toRemove.toSet())
        toRemove.forEach(::releaseTrack)
    }

    private fun reacquireDormantTracks(grayFrame: Mat, exclusionMask: Mat?): Int {
        if (dormantTracks.isEmpty()) return 0
        val recovered = ArrayList<Track>()
        val attempted =
            dormantTracks
                .sortedWith(compareBy<Track> { it.dormantFrames }.thenByDescending { it.age })
                .take(config.maxReacquisitionAttemptsPerFrame)
        for (track in attempted) {
            val template = track.appearancePatch ?: continue
            if (template.empty() || isMasked(track.currentPoint, exclusionMask)) continue
            val halfWidth = template.cols() / 2
            val halfHeight = template.rows() / 2
            val radius = config.appearanceSearchRadiusPx
            val left = max(0, track.currentPoint.x.toInt() - radius - halfWidth)
            val top = max(0, track.currentPoint.y.toInt() - radius - halfHeight)
            val right = min(grayFrame.cols(), track.currentPoint.x.toInt() + radius + halfWidth + 1)
            val bottom = min(grayFrame.rows(), track.currentPoint.y.toInt() + radius + halfHeight + 1)
            if (right - left < template.cols() || bottom - top < template.rows()) continue
            val search = grayFrame.submat(Rect(left, top, right - left, bottom - top))
            val response = Mat()
            try {
                Imgproc.matchTemplate(search, template, response, Imgproc.TM_CCOEFF_NORMED)
                val match = Core.minMaxLoc(response)
                if (!match.maxVal.isFinite() || match.maxVal < config.minReacquisitionScore) continue
                val point =
                    Point(
                        left + match.maxLoc.x + halfWidth,
                        top + match.maxLoc.y + halfHeight,
                    )
                if (isMasked(point, exclusionMask) ||
                    isNearAny(point, activeTracks.map(Track::currentPoint))
                ) continue
                track.currentPoint = point
                track.staticProbability = max(0.45, track.staticProbability)
                track.dormantFrames = 0
                track.surfaceInlier = false
                recovered.add(track)
            } finally {
                search.release()
                response.release()
            }
        }
        if (recovered.isEmpty()) return 0
        dormantTracks.removeAll(recovered.toSet())
        activeTracks.addAll(recovered)
        return recovered.size
    }

    private fun extractAppearancePatch(grayFrame: Mat, point: Point): Mat? {
        val radius = config.appearancePatchRadiusPx
        val centerX = point.x.toInt()
        val centerY = point.y.toInt()
        val left = centerX - radius
        val top = centerY - radius
        val size = radius * 2 + 1
        if (left < 0 || top < 0 || left + size > grayFrame.cols() || top + size > grayFrame.rows()) {
            return null
        }
        val view = grayFrame.submat(Rect(left, top, size, size))
        return try {
            view.clone()
        } finally {
            view.release()
        }
    }

    private fun releaseTrack(track: Track) {
        track.appearancePatch?.release()
        track.appearancePatch = null
    }

    private fun findNewPoints(
        grayFrame: Mat,
        existing: List<Point>,
        maxToAdd: Int,
        roiMask: Mat?,
        exclusionMask: Mat?,
        distributionQuad: List<Point>
    ): List<Point> {
        if (maxToAdd <= 0) return emptyList()
        val corners = MatOfPoint()
        return try {
            Imgproc.goodFeaturesToTrack(
                grayFrame,
                corners,
                config.maxDetectedCorners,
                config.detectQualityLevel,
                config.detectMinDistancePx,
                roiMask,
                3,
                false,
                0.04
            )
            val columns = max(1, config.featureCoverageColumns)
            val rows = max(1, config.featureCoverageRows)
            val cellCount = columns * rows
            val maximumPerCell = max(1, (config.maxFeatures + cellCount - 1) / cellCount)
            val existingPerCell = IntArray(cellCount)
            for (point in existing) {
                featureCell(point, distributionQuad)?.let { cell -> existingPerCell[cell] += 1 }
            }
            val candidatesByCell = List(cellCount) { ArrayList<Point>() }
            for (point in corners.toArray()) {
                if (isMasked(point, exclusionMask)) continue
                if (isNearAny(point, existing)) continue
                val cell = featureCell(point, distributionQuad) ?: continue
                candidatesByCell[cell].add(point)
            }

            val result = ArrayList<Point>(maxToAdd)
            val nextCandidate = IntArray(cellCount)
            var addedInPass: Boolean
            do {
                addedInPass = false
                for (cell in 0 until cellCount) {
                    if (result.size >= maxToAdd) break
                    if (existingPerCell[cell] >= maximumPerCell) continue
                    val bucket = candidatesByCell[cell]
                    while (nextCandidate[cell] < bucket.size) {
                        val point = bucket[nextCandidate[cell]++]
                        if (isNearAny(point, result)) continue
                        result.add(copyPoint(point))
                        existingPerCell[cell] += 1
                        addedInPass = true
                        break
                    }
                }
            } while (result.size < maxToAdd && addedInPass)
            result
        } finally {
            corners.release()
        }
    }

    private fun featureCell(point: Point, quad: List<Point>): Int? {
        if (quad.size != 4 || !isInsideConvexQuad(point, quad)) return null
        val minX = quad.minOf { it.x }
        val maxX = quad.maxOf { it.x }
        val minY = quad.minOf { it.y }
        val maxY = quad.maxOf { it.y }
        val width = maxX - minX
        val height = maxY - minY
        if (width <= 1e-6 || height <= 1e-6) return null
        val columns = max(1, config.featureCoverageColumns)
        val rows = max(1, config.featureCoverageRows)
        val column = (((point.x - minX) / width) * columns).toInt().coerceIn(0, columns - 1)
        val row = (((point.y - minY) / height) * rows).toInt().coerceIn(0, rows - 1)
        return row * columns + column
    }

    private fun buildQuadMask(rows: Int, cols: Int, quad: List<Point>): Mat {
        val mask = Mat(rows, cols, CvType.CV_8UC1, Scalar(0.0))
        val polygon = MatOfPoint(
            *quad.map {
                Point(
                    it.x.coerceIn(0.0, max(0, cols - 1).toDouble()),
                    it.y.coerceIn(0.0, max(0, rows - 1).toDouble())
                )
            }.toTypedArray()
        )
        try {
            Imgproc.fillConvexPoly(mask, polygon, Scalar(255.0))
        } finally {
            polygon.release()
        }
        return mask
    }

    private fun fullFrameQuad(width: Int, height: Int): List<Point> {
        val maxX = max(0, width - 1).toDouble()
        val maxY = max(0, height - 1).toDouble()
        return listOf(
            Point(0.0, 0.0),
            Point(maxX, 0.0),
            Point(maxX, maxY),
            Point(0.0, maxY)
        )
    }

    private fun mapQuad(quad: List<Point>, transform: SurfaceTransform): List<Point>? {
        val mapped = ArrayList<Point>(quad.size)
        for (point in quad) {
            val next = transform.map(point) ?: return null
            mapped.add(next)
        }
        return mapped
    }

    private fun updatePrevGray(grayFrame: Mat) {
        val cached = prevGray
        if (cached == null ||
            cached.rows() != grayFrame.rows() ||
            cached.cols() != grayFrame.cols() ||
            cached.type() != grayFrame.type()
        ) {
            cached?.release()
            prevGray = grayFrame.clone()
        } else {
            grayFrame.copyTo(cached)
        }
    }

    private fun computeConfidence(activeTrackCount: Int, inlierCount: Int, meanReprojError: Double?): Float {
        if (meanReprojError == null || inlierCount <= 0 || activeTrackCount <= 0) return 0f
        val countScore = (inlierCount.toFloat() / config.minHomographySupportTracks.toFloat()).coerceIn(0f, 1f)
        val ratioScore = (inlierCount.toFloat() / activeTrackCount.toFloat()).coerceIn(0f, 1f)
        val errorScore = (1.0 - (meanReprojError / config.maxMeanReprojErrorPx).coerceIn(0.0, 1.0)).toFloat()
        return (0.4f * countScore + 0.4f * ratioScore + 0.2f * errorScore).coerceIn(0f, 1f)
    }

    private fun collectInlierIndices(mask: Mat, expectedSize: Int): List<Int> {
        if (mask.empty()) return (0 until expectedSize).toList()
        val bytes = ByteArray(mask.total().toInt())
        mask.get(0, 0, bytes)
        val result = ArrayList<Int>(expectedSize)
        for (i in 0 until min(expectedSize, bytes.size)) {
            if (bytes[i].toInt() != 0) {
                result.add(i)
            }
        }
        return result
    }

    private fun computePerspectiveReprojectionError(srcPoints: List<Point>, dstPoints: List<Point>, homography: Mat): Double {
        val src = MatOfPoint2f(*srcPoints.toTypedArray())
        val projected = MatOfPoint2f()
        return try {
            Core.perspectiveTransform(src, projected, homography)
            val projectedPoints = projected.toArray()
            var errorSum = 0.0
            for (i in projectedPoints.indices) {
                val dx = projectedPoints[i].x - dstPoints[i].x
                val dy = projectedPoints[i].y - dstPoints[i].y
                errorSum += sqrt(dx * dx + dy * dy)
            }
            if (projectedPoints.isEmpty()) Double.MAX_VALUE else errorSum / projectedPoints.size
        } finally {
            src.release()
            projected.release()
        }
    }

    private fun computeAffineReprojectionError(srcPoints: List<Point>, dstPoints: List<Point>, affine: Mat): Double {
        var errorSum = 0.0
        for (i in srcPoints.indices) {
            val projected = affineTransformPoint(srcPoints[i], affine) ?: return Double.MAX_VALUE
            val dx = projected.x - dstPoints[i].x
            val dy = projected.y - dstPoints[i].y
            errorSum += sqrt(dx * dx + dy * dy)
        }
        return if (srcPoints.isEmpty()) Double.MAX_VALUE else errorSum / srcPoints.size
    }

    private fun affineTransformPoint(point: Point, affine: Mat): Point? {
        if (affine.rows() != 2 || affine.cols() != 3) return null
        val coefficients = DoubleArray(6)
        affine.get(0, 0, coefficients)
        val x = coefficients[0] * point.x + coefficients[1] * point.y + coefficients[2]
        val y = coefficients[3] * point.x + coefficients[4] * point.y + coefficients[5]
        return if (x.isFinite() && y.isFinite()) Point(x, y) else null
    }

    private fun hasSufficientSupportSpread(
        points: List<Point>,
        frameWidth: Int,
        frameHeight: Int,
        minSpanRatio: Double,
        minAreaRatio: Double
    ): Boolean {
        if (points.isEmpty() || frameWidth <= 0 || frameHeight <= 0) return false
        var minX = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY
        for (point in points) {
            minX = min(minX, point.x)
            maxX = max(maxX, point.x)
            minY = min(minY, point.y)
            maxY = max(maxY, point.y)
        }
        val spanX = maxX - minX
        val spanY = maxY - minY
        val spanRatioX = spanX / max(1.0, frameWidth.toDouble())
        val spanRatioY = spanY / max(1.0, frameHeight.toDouble())
        val areaRatio = (spanX * spanY) / max(1.0, frameWidth.toDouble() * frameHeight.toDouble())
        return spanRatioX >= minSpanRatio && spanRatioY >= minSpanRatio && areaRatio >= minAreaRatio
    }

    private fun validateQuadGeometry(mappedQuad: List<Point>): String? {
        if (mappedQuad.size != 4 || mappedQuad.any { !it.x.isFinite() || !it.y.isFinite() }) {
            return "quad_non_finite"
        }
        if (!isConvexQuad(mappedQuad)) return "quad_non_convex"
        val referenceSignedArea = signedQuadArea(referenceQuad)
        val mappedSignedArea = signedQuadArea(mappedQuad)
        if (referenceSignedArea * mappedSignedArea <= 0.0) return "quad_winding_changed"

        val frameArea = max(1.0, referenceFrameWidth.toDouble() * referenceFrameHeight.toDouble())
        val mappedAreaRatio = abs(mappedSignedArea) / frameArea
        if (mappedAreaRatio < config.minMappedQuadAreaRatio) return "quad_too_small"
        if (!config.allowOffscreenMappedQuad && mappedAreaRatio > config.maxMappedQuadAreaRatio) {
            return "quad_too_large"
        }

        val edges = quadEdgeLengths(mappedQuad)
        if (edges.any { !it.isFinite() || it < config.minMappedEdgeLengthPx }) return "edge_too_short"
        if (max(edges[0], edges[2]) / max(1e-6, min(edges[0], edges[2])) > config.maxOppositeEdgeRatio ||
            max(edges[1], edges[3]) / max(1e-6, min(edges[1], edges[3])) > config.maxOppositeEdgeRatio
        ) {
            return "opposite_edge_ratio"
        }
        if (minimumCornerAngleDegrees(mappedQuad) < config.minCornerAngleDegrees) {
            return "corner_angle"
        }

        if (!config.allowOffscreenMappedQuad) {
            val outsideX = referenceFrameWidth * config.maxCornerOutsideFrameRatio
            val outsideY = referenceFrameHeight * config.maxCornerOutsideFrameRatio
            if (mappedQuad.any {
                    it.x < -outsideX || it.x > referenceFrameWidth + outsideX ||
                        it.y < -outsideY || it.y > referenceFrameHeight + outsideY
                }
            ) {
                return "quad_outside_frame"
            }
        }

        val previousQuad = lastAcceptedMappedQuad
        if (previousQuad != null && previousQuad.size == 4) {
            val previousArea = abs(signedQuadArea(previousQuad))
            val areaScale = abs(mappedSignedArea) / max(1e-6, previousArea)
            if (areaScale < config.minFrameAreaScale || areaScale > config.maxFrameAreaScale) {
                return "frame_area_jump"
            }
            val previousEdges = quadEdgeLengths(previousQuad)
            for (index in edges.indices) {
                val edgeScale = edges[index] / max(1e-6, previousEdges[index])
                if (edgeScale < config.minFrameEdgeScale || edgeScale > config.maxFrameEdgeScale) {
                    return "frame_edge_jump"
                }
            }
            val frameDiagonal =
                sqrt(
                    referenceFrameWidth.toDouble() * referenceFrameWidth.toDouble() +
                        referenceFrameHeight.toDouble() * referenceFrameHeight.toDouble()
                )
            val maximumCornerMotion =
                mappedQuad.indices.maxOf { index ->
                    val dx = mappedQuad[index].x - previousQuad[index].x
                    val dy = mappedQuad[index].y - previousQuad[index].y
                    sqrt(dx * dx + dy * dy)
                }
            if (maximumCornerMotion > frameDiagonal * config.maxFrameCornerMotionRatio) {
                return "frame_corner_jump"
            }
        }
        return null
    }

    private fun computeFeatureCoverage(points: List<Point>, quad: List<Point>): Double {
        if (points.isEmpty() || quad.size != 4) return 0.0
        val minX = quad.minOf { it.x }
        val maxX = quad.maxOf { it.x }
        val minY = quad.minOf { it.y }
        val maxY = quad.maxOf { it.y }
        val width = maxX - minX
        val height = maxY - minY
        if (width <= 1e-6 || height <= 1e-6) return 0.0
        val columns = max(1, config.featureCoverageColumns)
        val rows = max(1, config.featureCoverageRows)
        val occupied = HashSet<Int>()
        for (point in points) {
            if (!isInsideConvexQuad(point, quad)) continue
            val column = (((point.x - minX) / width) * columns).toInt().coerceIn(0, columns - 1)
            val row = (((point.y - minY) / height) * rows).toInt().coerceIn(0, rows - 1)
            occupied.add(row * columns + column)
        }
        return occupied.size.toDouble() / (columns * rows).toDouble()
    }

    private fun hasSupportSpanWithinQuad(points: List<Point>, quad: List<Point>): Boolean {
        if (points.isEmpty() || quad.size != 4) return false
        val quadWidth = quad.maxOf { it.x } - quad.minOf { it.x }
        val quadHeight = quad.maxOf { it.y } - quad.minOf { it.y }
        if (quadWidth <= 1e-6 || quadHeight <= 1e-6) return false
        val pointSpanX = points.maxOf { it.x } - points.minOf { it.x }
        val pointSpanY = points.maxOf { it.y } - points.minOf { it.y }
        return pointSpanX / quadWidth >= config.minSupportSpanWithinQuad &&
            pointSpanY / quadHeight >= config.minSupportSpanWithinQuad
    }

    private fun hasSimilaritySupportExtent(points: List<Point>, quad: List<Point>): Boolean {
        if (points.size < 2 || quad.size != 4) return false
        var maximumDistanceSquared = 0.0
        for (first in 0 until points.lastIndex) {
            for (second in first + 1 until points.size) {
                val dx = points[first].x - points[second].x
                val dy = points[first].y - points[second].y
                maximumDistanceSquared = max(maximumDistanceSquared, dx * dx + dy * dy)
            }
        }
        val quadWidth = quad.maxOf { it.x } - quad.minOf { it.x }
        val quadHeight = quad.maxOf { it.y } - quad.minOf { it.y }
        val quadDiagonal = sqrt(quadWidth * quadWidth + quadHeight * quadHeight)
        if (quadDiagonal <= 1e-6) return false
        return sqrt(maximumDistanceSquared) / quadDiagonal >= config.minSimilaritySupportExtent
    }

    private fun signedQuadArea(quad: List<Point>): Double {
        if (quad.size < 3) return 0.0
        var twiceArea = 0.0
        for (index in quad.indices) {
            val next = (index + 1) % quad.size
            twiceArea += quad[index].x * quad[next].y - quad[next].x * quad[index].y
        }
        return twiceArea * 0.5
    }

    private fun quadEdgeLengths(quad: List<Point>): DoubleArray =
        DoubleArray(quad.size) { index ->
            val next = (index + 1) % quad.size
            val dx = quad[next].x - quad[index].x
            val dy = quad[next].y - quad[index].y
            sqrt(dx * dx + dy * dy)
        }

    private fun isConvexQuad(quad: List<Point>): Boolean {
        if (quad.size != 4) return false
        var sign = 0
        for (index in quad.indices) {
            val a = quad[index]
            val b = quad[(index + 1) % quad.size]
            val c = quad[(index + 2) % quad.size]
            val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if (abs(cross) < 1e-6) return false
            val nextSign = if (cross > 0.0) 1 else -1
            if (sign == 0) sign = nextSign else if (sign != nextSign) return false
        }
        return true
    }

    private fun isInsideConvexQuad(point: Point, quad: List<Point>): Boolean {
        if (quad.size != 4) return false
        var sign = 0
        for (index in quad.indices) {
            val a = quad[index]
            val b = quad[(index + 1) % quad.size]
            val cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)
            if (abs(cross) < 1e-6) continue
            val nextSign = if (cross > 0.0) 1 else -1
            if (sign == 0) sign = nextSign else if (sign != nextSign) return false
        }
        return true
    }

    private fun minimumCornerAngleDegrees(quad: List<Point>): Double {
        var minimumAngle = 180.0
        for (index in quad.indices) {
            val previous = quad[(index + quad.size - 1) % quad.size]
            val current = quad[index]
            val next = quad[(index + 1) % quad.size]
            val ax = previous.x - current.x
            val ay = previous.y - current.y
            val bx = next.x - current.x
            val by = next.y - current.y
            val aLength = sqrt(ax * ax + ay * ay)
            val bLength = sqrt(bx * bx + by * by)
            if (aLength <= 1e-6 || bLength <= 1e-6) return 0.0
            val cosine = ((ax * bx + ay * by) / (aLength * bLength)).coerceIn(-1.0, 1.0)
            minimumAngle = min(minimumAngle, acos(cosine) * 180.0 / Math.PI)
        }
        return minimumAngle
    }

    private fun normalizedHomographyCondition(coefficients: DoubleArray): Double {
        if (coefficients.size != 9 || referenceFrameWidth <= 0 || referenceFrameHeight <= 0) {
            return Double.POSITIVE_INFINITY
        }
        val width = referenceFrameWidth.toDouble()
        val height = referenceFrameHeight.toDouble()
        val normalized =
            doubleArrayOf(
                coefficients[0], coefficients[1] * height / width, coefficients[2] / width,
                coefficients[3] * width / height, coefficients[4], coefficients[5] / height,
                coefficients[6] * width, coefficients[7] * height, coefficients[8]
            )
        val scale = normalized.maxOf { abs(it) }
        if (!scale.isFinite() || scale <= 1e-12) return Double.POSITIVE_INFINITY
        for (index in normalized.indices) normalized[index] /= scale
        val inverse = invert3x3(normalized) ?: return Double.POSITIVE_INFINITY
        return matrixOneNorm(normalized) * matrixOneNorm(inverse)
    }

    private fun invert3x3(matrix: DoubleArray): DoubleArray? {
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
        if (!determinant.isFinite() || abs(determinant) < 1e-12) return null
        return doubleArrayOf(
            (e * i - f * h) / determinant,
            (c * h - b * i) / determinant,
            (b * f - c * e) / determinant,
            (f * g - d * i) / determinant,
            (a * i - c * g) / determinant,
            (c * d - a * f) / determinant,
            (d * h - e * g) / determinant,
            (b * g - a * h) / determinant,
            (a * e - b * d) / determinant
        )
    }

    private fun matrixOneNorm(matrix: DoubleArray): Double {
        var maximum = 0.0
        for (column in 0 until 3) {
            var sum = 0.0
            for (row in 0 until 3) sum += abs(matrix[row * 3 + column])
            maximum = max(maximum, sum)
        }
        return maximum
    }

    private fun isNearAny(point: Point, others: List<Point>): Boolean {
        val minDistanceSquared = config.minFeatureSeparationPx * config.minFeatureSeparationPx
        for (other in others) {
            val dx = point.x - other.x
            val dy = point.y - other.y
            if (dx * dx + dy * dy <= minDistanceSquared) {
                return true
            }
        }
        return false
    }

    private fun isInside(point: Point, width: Int, height: Int): Boolean {
        return point.x >= 0.0 && point.y >= 0.0 && point.x < width && point.y < height
    }

    private fun isMasked(point: Point, mask: Mat?): Boolean {
        if (mask == null || mask.empty()) return false
        val x = point.x.toInt()
        val y = point.y.toInt()
        if (x < 0 || y < 0 || x >= mask.cols() || y >= mask.rows()) return false
        return (mask.get(y, x)?.firstOrNull() ?: 0.0) > 0.0
    }

    private fun isUsableGray(grayFrame: Mat): Boolean {
        return !grayFrame.empty() && grayFrame.channels() == 1
    }

    private fun copyPoint(point: Point): Point {
        return Point(point.x, point.y)
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 1) {
            sorted[mid]
        } else {
            0.5 * (sorted[mid - 1] + sorted[mid])
        }
    }

    private fun cauchyReliability(error: Double, scale: Double): Double {
        if (!error.isFinite() || !scale.isFinite() || scale <= 0.0) return 0.0
        val normalized = error / scale
        return 1.0 / (1.0 + normalized * normalized)
    }
}

