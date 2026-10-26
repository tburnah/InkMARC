/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.inkmarc

import android.opengl.GLSurfaceView
import android.view.TextureView
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.examples.java.common.helpers.SnackbarHelper

/** Contains UI elements for InkMARC. */
class InkMarcView(val activity: InkMarcActivity) : DefaultLifecycleObserver {
  val root = View.inflate(activity, R.layout.activity_main, null)
  val surfaceView = root.findViewById<GLSurfaceView>(R.id.surfaceview)
  val fallbackCameraView = root.findViewById<TextureView>(R.id.fallback_camera_view)
  val fallbackOverlayView = root.findViewById<FallbackOverlayView>(R.id.fallback_overlay)
  private val fallbackStatusView = root.findViewById<TextView>(R.id.fallback_status)
  private val recordingStatusView = root.findViewById<TextView>(R.id.recording_status)
  private val finishCaptureButton = root.findViewById<Button>(R.id.finish_capture_button)

  val snackbarHelper = SnackbarHelper()
  val touchHelper =
    PlacementTouchHelper(activity).also {
      surfaceView.setOnTouchListener(it)
      fallbackCameraView.setOnTouchListener(it)
    }

  fun useFallbackCamera() {
    surfaceView.visibility = View.GONE
    fallbackCameraView.visibility = View.VISIBLE
    fallbackStatusView.visibility = View.VISIBLE
  }

  override fun onResume(owner: LifecycleOwner) {
    if (InkMarcActivity.ENABLE_ARCORE) surfaceView.onResume()
  }

  override fun onPause(owner: LifecycleOwner) {
    if (InkMarcActivity.ENABLE_ARCORE) surfaceView.onPause()
  }

  fun updateRecordingStatus(text: String) {
    if (!InkMarcActivity.ENABLE_DATA_RECORDING) return
    recordingStatusView.text = text
    recordingStatusView.visibility = View.VISIBLE
  }

  fun hideRecordingStatus() {
    recordingStatusView.visibility = View.GONE
  }

  fun showFinishCaptureButton(show: Boolean) {
    finishCaptureButton.visibility = if (show) View.VISIBLE else View.GONE
    if (show) {
      finishCaptureButton.setOnClickListener { activity.finishCurrentCapture() }
    }
  }

  fun disableFinishCaptureButton() {
    finishCaptureButton.isEnabled = false
  }

  fun updateFallbackStatus(text: String) {
    if (!InkMarcActivity.ENABLE_FALLBACK_TRACKER) return
    fallbackStatusView.text = text
  }

  fun updateInferenceCursor(screenPoint: FloatArray?, alpha: Float = 1f, label: String? = null) {
    fallbackOverlayView.updateCursor(screenPoint, alpha, label)
  }

  fun showOcclusionDialogIfNeeded() = Unit
}

