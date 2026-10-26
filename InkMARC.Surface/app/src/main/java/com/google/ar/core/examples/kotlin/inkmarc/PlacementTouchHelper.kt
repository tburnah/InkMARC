package com.inkmarc

import android.content.Context
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.BlockingQueue
import kotlin.math.PI
import kotlin.math.atan2

class PlacementTouchHelper(context: Context) : View.OnTouchListener {
  private val tapDetector =
    GestureDetector(
      context,
      object : GestureDetector.SimpleOnGestureListener() {
        override fun onSingleTapUp(e: MotionEvent): Boolean {
          queuedSingleTaps.offer(MotionEvent.obtain(e))
          return true
        }

        override fun onDown(e: MotionEvent): Boolean = true

        override fun onScroll(
          e1: MotionEvent?,
          e2: MotionEvent,
          distanceX: Float,
          distanceY: Float
        ): Boolean {
          if (e2.pointerCount != 1 || scaleDetector.isInProgress) return false
          synchronized(lock) {
            pendingPanX += -distanceX
            pendingPanY += -distanceY
          }
          return true
        }
      }
    )

  private val scaleDetector =
    ScaleGestureDetector(
      context,
      object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
          synchronized(lock) {
            pendingScaleFactor *= detector.scaleFactor
          }
          return true
        }
      }
    )

  private val queuedSingleTaps: BlockingQueue<MotionEvent> = ArrayBlockingQueue(16)
  private val lock = Any()
  private var pendingScaleFactor = 1f
  private var pendingRotationRadians = 0f
  private var pendingPanX = 0f
  private var pendingPanY = 0f
  private var previousRotationAngle: Float? = null

  fun pollTap(): MotionEvent? = queuedSingleTaps.poll()

  fun consumeTransformDelta(): TransformDelta {
    synchronized(lock) {
      val delta = TransformDelta(pendingScaleFactor, pendingRotationRadians, pendingPanX, pendingPanY)
      pendingScaleFactor = 1f
      pendingRotationRadians = 0f
      pendingPanX = 0f
      pendingPanY = 0f
      return delta
    }
  }

  override fun onTouch(view: View, motionEvent: MotionEvent): Boolean {
    scaleDetector.onTouchEvent(motionEvent)
    tapDetector.onTouchEvent(motionEvent)

    when (motionEvent.actionMasked) {
      MotionEvent.ACTION_POINTER_DOWN -> {
        if (motionEvent.pointerCount >= 2) {
          previousRotationAngle = rotationAngle(motionEvent)
        }
      }
      MotionEvent.ACTION_MOVE -> {
        if (motionEvent.pointerCount >= 2) {
          val angle = rotationAngle(motionEvent)
          val previous = previousRotationAngle
          if (previous != null) {
            synchronized(lock) {
              pendingRotationRadians += normalizeAngle(angle - previous)
            }
          }
          previousRotationAngle = angle
        }
      }
      MotionEvent.ACTION_POINTER_UP -> {
        previousRotationAngle =
          if (motionEvent.pointerCount - 1 >= 2) rotationAngle(motionEvent) else null
      }
      MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
        previousRotationAngle = null
      }
    }

    return true
  }

  private fun rotationAngle(event: MotionEvent): Float {
    val dx = event.getX(1) - event.getX(0)
    val dy = event.getY(1) - event.getY(0)
    return atan2(dy, dx)
  }

  private fun normalizeAngle(angle: Float): Float {
    var normalized = angle
    while (normalized > PI) normalized -= (2f * PI).toFloat()
    while (normalized < -PI) normalized += (2f * PI).toFloat()
    return normalized
  }

  data class TransformDelta(
    val scaleFactor: Float,
    val rotationRadians: Float,
    val panX: Float,
    val panY: Float,
  )
}
