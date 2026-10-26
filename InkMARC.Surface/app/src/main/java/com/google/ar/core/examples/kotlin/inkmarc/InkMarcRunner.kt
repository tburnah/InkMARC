package com.inkmarc

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import android.util.Log
import java.nio.FloatBuffer
import java.util.Collections

class InkMarcRunner(private val assets: AssetManager) {
  companion object {
    private const val TAG = "InkMarcRunner"
    private const val PRIMARY_ASSET = "models/inktrack_rgb_geometry.onnx"
    private const val FALLBACK_ASSET = "inktrack_rgb_geometry.onnx"
    private val INPUT_SHAPE = longArrayOf(1, 6, 672, 432)
  }

  data class InkMarcOutputs(
    val stateLogits: FloatArray,
    val touchLogits: FloatArray,
    val xy: FloatArray,
  )

  private var environment: OrtEnvironment? = null
  private var session: OrtSession? = null
  private var inputName: String? = null

  fun run(inputTensorData: FloatArray): InkMarcOutputs? {
    val ortSession = initializeSessionIfNeeded() ?: return null
    val env = environment ?: return null
    val resolvedInputName = inputName ?: return null

    OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensorData), INPUT_SHAPE).use { inputTensor ->
      ortSession.run(Collections.singletonMap(resolvedInputName, inputTensor)).use { outputs ->
        val stateLogits = (outputs[0] as? OnnxTensor)?.readFloatArray() ?: return null
        val touchLogits = (outputs[1] as? OnnxTensor)?.readFloatArray() ?: return null
        val xy = (outputs[2] as? OnnxTensor)?.readFloatArray() ?: return null
        return InkMarcOutputs(
          stateLogits = stateLogits,
          touchLogits = touchLogits,
          xy = xy,
        )
      }
    }
  }

  fun hasModelAsset(): Boolean = findAssetPath() != null

  fun close() {
    session?.close()
    session = null
    environment = null
  }

  private fun initializeSessionIfNeeded(): OrtSession? {
    session?.let { return it }
    return synchronized(this) {
      session?.let { return@synchronized it }
      val env = environment ?: OrtEnvironment.getEnvironment().also { environment = it }
      val options = OrtSession.SessionOptions()
      options.setIntraOpNumThreads(1)
      options.setInterOpNumThreads(1)
      options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

      val assetPath = findAssetPath() ?: return@synchronized null
      val modelBytes = assets.open(assetPath).use { it.readBytes() }
      val createdSession = env.createSession(modelBytes, options)
      inputName =
        createdSession.inputNames.firstOrNull()
          ?: throw IllegalStateException("InkMARC ONNX model has no inputs.")
      session = createdSession
      Log.i(TAG, "Initialized InkMARC session from $assetPath")
      createdSession
    }
  }

  private fun findAssetPath(): String? {
    return when {
      assetExists(PRIMARY_ASSET) -> PRIMARY_ASSET
      assetExists(FALLBACK_ASSET) -> FALLBACK_ASSET
      else -> null
    }
  }

  private fun assetExists(assetPath: String): Boolean =
    try {
      assets.open(assetPath).close()
      true
    } catch (_: Exception) {
      false
    }
}

private fun OnnxTensor.readFloatArray(): FloatArray {
  val buffer = floatBuffer
  val output = FloatArray(buffer.remaining())
  buffer.duplicate().get(output)
  return output
}
