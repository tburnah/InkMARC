package com.inkmarc

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

class FallbackOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
  private data class CursorState(
    val x: Float,
    val y: Float,
    val alpha: Float,
    val label: String?,
  )

  private val lock = Any()
  private val quadPath = Path()
  private val outlinePaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = 0xffffb300.toInt()
      style = Paint.Style.STROKE
      strokeWidth = 6f
    }
  private val cornerPaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = 0xff4dd0e1.toInt()
      style = Paint.Style.FILL
    }
  private val cursorStrokePaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = 0xff111111.toInt()
      style = Paint.Style.STROKE
      strokeWidth = 10f
      strokeCap = Paint.Cap.ROUND
    }
  private val cursorPaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = 0xffffffff.toInt()
      style = Paint.Style.STROKE
      strokeWidth = 5f
      strokeCap = Paint.Cap.ROUND
    }
  private val labelStrokePaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = 0xff111111.toInt()
      style = Paint.Style.STROKE
      strokeWidth = 8f
      textSize = 38f
      textAlign = Paint.Align.LEFT
      strokeJoin = Paint.Join.ROUND
    }
  private val labelPaint =
    Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = 0xffffffff.toInt()
      style = Paint.Style.FILL
      textSize = 38f
      textAlign = Paint.Align.LEFT
    }
  private val drawQuadOutline = true
  private var quad: FloatArray? = null
  private var cursor: CursorState? = null

  init {
    isClickable = false
    isFocusable = false
  }

  fun updateQuad(points: FloatArray?) {
    synchronized(lock) {
      quad = points?.copyOf()
    }
    postInvalidateOnAnimation()
  }

  fun updateCursor(screenPoint: FloatArray?, alpha: Float = 1f, label: String? = null) {
    synchronized(lock) {
      cursor =
        if (screenPoint != null && screenPoint.size >= 2) {
          CursorState(screenPoint[0], screenPoint[1], alpha.coerceIn(0f, 1f), label)
        } else {
          null
        }
    }
    postInvalidateOnAnimation()
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    val snapshot =
      synchronized(lock) {
        Pair(quad?.copyOf(), cursor?.copy())
      }
    val points = snapshot.first
    val cursorPoint = snapshot.second

    if (drawQuadOutline && points != null && points.size >= 8) {
      quadPath.reset()
      quadPath.moveTo(points[0], points[1])
      for (i in 2 until points.size step 2) {
        quadPath.lineTo(points[i], points[i + 1])
      }
      quadPath.close()
      canvas.drawPath(quadPath, outlinePaint)
      for (i in points.indices step 2) {
        canvas.drawCircle(points[i], points[i + 1], 9f, cornerPaint)
      }
    }

    if (cursorPoint != null) {
      val x = cursorPoint.x
      val y = cursorPoint.y
      val arm = 16f
      val alphaInt = (cursorPoint.alpha * 255f).toInt().coerceIn(0, 255)
      cursorStrokePaint.alpha = alphaInt
      cursorPaint.alpha = alphaInt
      labelStrokePaint.alpha = alphaInt
      labelPaint.alpha = alphaInt
      canvas.drawLine(x - arm, y - arm, x + arm, y + arm, cursorStrokePaint)
      canvas.drawLine(x - arm, y + arm, x + arm, y - arm, cursorStrokePaint)
      canvas.drawLine(x - arm, y - arm, x + arm, y + arm, cursorPaint)
      canvas.drawLine(x - arm, y + arm, x + arm, y - arm, cursorPaint)
      cursorPoint.label?.let { label ->
        val labelX = x + 24f
        val labelY = y - 24f
        canvas.drawText(label, labelX, labelY, labelStrokePaint)
        canvas.drawText(label, labelX, labelY, labelPaint)
      }
    }
  }
}

