package com.inkmarc

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.Frame
import java.util.ArrayDeque
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

class GravityDirectionProvider(context: Context) : DefaultLifecycleObserver {
  private data class Quaternion(
    val x: Double,
    val y: Double,
    val z: Double,
    val w: Double,
  )

  private data class OrientationSample(
    val timestampNs: Long,
    val orientation: Quaternion,
  )

  private data class GyroSample(
    val timestampNs: Long,
    val x: Double,
    val y: Double,
    val z: Double,
  )

  private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
  private val gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
  private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
  private val rotationSensor =
    sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
      ?: sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
  private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
  @Volatile private var latestDeviceGravity = floatArrayOf(0f, 0f, 9.81f)
  private val orientationLock = Any()
  private val orientationHistory = ArrayDeque<OrientationSample>(128)
  private val gyroHistory = ArrayDeque<GyroSample>(256)
  private var latestGyro: GyroSample? = null

  private val sensorListener =
    object : SensorEventListener {
      override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
          Sensor.TYPE_GRAVITY, Sensor.TYPE_ACCELEROMETER -> {
            if (event.values.size < 3) return
            latestDeviceGravity =
              floatArrayOf(event.values[0], event.values[1], event.values[2]).normalized()
          }
          Sensor.TYPE_GAME_ROTATION_VECTOR, Sensor.TYPE_ROTATION_VECTOR -> {
            quaternionFromRotationVector(event.values)?.let { quaternion ->
              synchronized(orientationLock) {
                orientationHistory.addLast(OrientationSample(event.timestamp, quaternion))
                while (orientationHistory.size > 128) orientationHistory.removeFirst()
              }
            }
          }
          Sensor.TYPE_GYROSCOPE -> {
            if (event.values.size < 3) return
            synchronized(orientationLock) {
              val sample =
                GyroSample(
                  event.timestamp,
                  event.values[0].toDouble(),
                  event.values[1].toDouble(),
                  event.values[2].toDouble(),
                )
              latestGyro = sample
              gyroHistory.addLast(sample)
              while (gyroHistory.size > 256) gyroHistory.removeFirst()
            }
          }
        }
      }

      override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    }

  override fun onResume(owner: LifecycleOwner) {
    gravitySensor?.let {
      sensorManager.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_FASTEST)
    } ?: accelerometer?.let {
      sensorManager.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_FASTEST)
    }
    rotationSensor?.let {
      sensorManager.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_FASTEST)
    }
    gyroscope?.let {
      sensorManager.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_FASTEST)
    }
  }

  override fun onPause(owner: LifecycleOwner) {
    sensorManager.unregisterListener(sensorListener)
    synchronized(orientationLock) {
      orientationHistory.clear()
      gyroHistory.clear()
      latestGyro = null
    }
  }

  /** Returns a row-major device-to-world rotation matrix at the requested elapsed-realtime time. */
  fun getDeviceToWorldRotation(timestampNs: Long): DoubleArray? {
    val samples: List<OrientationSample>
    val gyro: GyroSample?
    synchronized(orientationLock) {
      if (orientationHistory.isEmpty()) return null
      samples = orientationHistory.toList()
      gyro = latestGyro
    }

    var before: OrientationSample? = null
    var after: OrientationSample? = null
    for (sample in samples) {
      if (sample.timestampNs <= timestampNs) before = sample
      if (sample.timestampNs >= timestampNs) {
        after = sample
        break
      }
    }
    val orientation =
      when {
        before != null && after != null && after.timestampNs > before.timestampNs -> {
          val fraction =
            (timestampNs - before.timestampNs).toDouble() /
              (after.timestampNs - before.timestampNs).toDouble()
          slerp(before.orientation, after.orientation, fraction.coerceIn(0.0, 1.0))
        }
        before != null -> extrapolate(before, gyro, timestampNs)
        else -> after?.orientation
      } ?: return null
    return quaternionToMatrix(orientation)
  }

  fun getAngularSpeedRadPerSecond(timestampNs: Long): Double? {
    val nearest =
      synchronized(orientationLock) {
        gyroHistory.minByOrNull { sample -> absDifference(sample.timestampNs, timestampNs) }
      } ?: return null
    if (absDifference(nearest.timestampNs, timestampNs) > 40_000_000L) return null
    return sqrt(nearest.x * nearest.x + nearest.y * nearest.y + nearest.z * nearest.z)
  }

  fun getWorldUp(frame: Frame): FloatArray {
    val matrix = FloatArray(16)
    frame.androidSensorPose.toMatrix(matrix, 0)
    val gravity = latestDeviceGravity
    val worldDown =
      floatArrayOf(
        matrix[0] * gravity[0] + matrix[4] * gravity[1] + matrix[8] * gravity[2],
        matrix[1] * gravity[0] + matrix[5] * gravity[1] + matrix[9] * gravity[2],
        matrix[2] * gravity[0] + matrix[6] * gravity[1] + matrix[10] * gravity[2],
      ).normalized()
    return floatArrayOf(-worldDown[0], -worldDown[1], -worldDown[2]).normalized()
  }

  private fun extrapolate(
    sample: OrientationSample,
    gyro: GyroSample?,
    timestampNs: Long,
  ): Quaternion {
    if (gyro == null || timestampNs <= sample.timestampNs) return sample.orientation
    val dtSeconds = ((timestampNs - sample.timestampNs) / 1e9).coerceIn(0.0, 0.05)
    if (dtSeconds <= 0.0 || absDifference(gyro.timestampNs, timestampNs) > 50_000_000L) {
      return sample.orientation
    }
    val speed = sqrt(gyro.x * gyro.x + gyro.y * gyro.y + gyro.z * gyro.z)
    if (speed <= 1e-8) return sample.orientation
    val halfAngle = speed * dtSeconds * 0.5
    val scale = sin(halfAngle) / speed
    val delta = Quaternion(gyro.x * scale, gyro.y * scale, gyro.z * scale, cos(halfAngle))
    return multiply(sample.orientation, delta).normalized()
  }

  private fun quaternionFromRotationVector(values: FloatArray): Quaternion? {
    if (values.size < 3) return null
    val x = values[0].toDouble()
    val y = values[1].toDouble()
    val z = values[2].toDouble()
    val w =
      if (values.size >= 4) values[3].toDouble()
      else sqrt(max(0.0, 1.0 - x * x - y * y - z * z))
    return Quaternion(x, y, z, w).normalized()
  }

  private fun slerp(first: Quaternion, secondValue: Quaternion, fraction: Double): Quaternion {
    var second = secondValue
    var dot = first.x * second.x + first.y * second.y + first.z * second.z + first.w * second.w
    if (dot < 0.0) {
      second = Quaternion(-second.x, -second.y, -second.z, -second.w)
      dot = -dot
    }
    if (dot > 0.9995) {
      return Quaternion(
          first.x + fraction * (second.x - first.x),
          first.y + fraction * (second.y - first.y),
          first.z + fraction * (second.z - first.z),
          first.w + fraction * (second.w - first.w),
        )
        .normalized()
    }
    val angle = acos(dot.coerceIn(-1.0, 1.0))
    val denominator = sin(angle)
    val firstWeight = sin((1.0 - fraction) * angle) / denominator
    val secondWeight = sin(fraction * angle) / denominator
    return Quaternion(
      first.x * firstWeight + second.x * secondWeight,
      first.y * firstWeight + second.y * secondWeight,
      first.z * firstWeight + second.z * secondWeight,
      first.w * firstWeight + second.w * secondWeight,
    )
  }

  private fun multiply(first: Quaternion, second: Quaternion): Quaternion =
    Quaternion(
      first.w * second.x + first.x * second.w + first.y * second.z - first.z * second.y,
      first.w * second.y - first.x * second.z + first.y * second.w + first.z * second.x,
      first.w * second.z + first.x * second.y - first.y * second.x + first.z * second.w,
      first.w * second.w - first.x * second.x - first.y * second.y - first.z * second.z,
    )

  private fun Quaternion.normalized(): Quaternion {
    val length = sqrt(x * x + y * y + z * z + w * w)
    if (length <= 1e-12) return Quaternion(0.0, 0.0, 0.0, 1.0)
    return Quaternion(x / length, y / length, z / length, w / length)
  }

  private fun quaternionToMatrix(quaternion: Quaternion): DoubleArray {
    val q = quaternion.normalized()
    val xx = q.x * q.x
    val yy = q.y * q.y
    val zz = q.z * q.z
    val xy = q.x * q.y
    val xz = q.x * q.z
    val yz = q.y * q.z
    val wx = q.w * q.x
    val wy = q.w * q.y
    val wz = q.w * q.z
    return doubleArrayOf(
      1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
      2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
      2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy),
    )
  }

  private fun absDifference(first: Long, second: Long): Long =
    if (first >= second) first - second else second - first
}

private fun FloatArray.normalized(): FloatArray {
  val length = sqrt((this[0] * this[0] + this[1] * this[1] + this[2] * this[2]).toDouble()).toFloat()
  if (length <= 1e-6f) return floatArrayOf(0f, 1f, 0f)
  return floatArrayOf(this[0] / length, this[1] / length, this[2] / length)
}
