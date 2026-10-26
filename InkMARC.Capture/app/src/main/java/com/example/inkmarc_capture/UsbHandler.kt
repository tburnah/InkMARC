package com.example.inkmarc_capture

import android.content.Context
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbManager
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Manages the connection and data stream for a connected USB HID device, such as a Wacom tablet.
 *
 * This class handles device discovery, permission requests, and endpoint setup. It reads data
 * from the device on a background thread using coroutines and parses the incoming byte packets
 * to extract stylus information (coordinates, pressure, tilt). The parsed data is then passed
 * to the UI thread via the `onStylusData` callback.
 *
 * @param context The application context, used to get the `UsbManager` system service.
 * @param onStylusData A callback function invoked on the main thread with the latest stylus data.
 *                     The parameters are nullable to indicate when the stylus is out of range.
 */
class UsbHandler(
    context: Context,
    private val onStylusData: (x: Int?, y: Int?, pressure: Int?, tiltX: Int?, tiltY: Int?) -> Unit
) {

    private val usbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
    private var deviceConnection: UsbDeviceConnection? = null
    private var dataEndpoint: UsbEndpoint? = null
    @Volatile private var running = false
    private val ioScope = CoroutineScope(Job() + Dispatchers.IO)
    private var readJob: Job? = null

    /**
     * Tracks whether the stylus is lifted or out of range.
     * When true, the handler sends null data via the callback.
     */
    private var inNaNMode = false

    /**
     * Starts the process of connecting to and reading from the specified USB device.
     *
     * If permission has already been granted, it proceeds to set up the device. Otherwise, it logs an error.
     * It will not start a new connection if one is already active.
     *
     * @param device The `UsbDevice` to connect to.
     */
    fun start(device: UsbDevice) {
        Log.d(TAG, "start() called with device: ${device.deviceName}")
        if (deviceConnection != null) {
            Log.d(TAG, "Handler already running, ignoring start request.")
            return
        }

        if (usbManager.hasPermission(device)) {
            setupDevice(device)
        } else {
            Log.e(TAG, "No permission to access USB device: ${device.deviceName}")
        }
    }

    /**
     * Opens the device connection, claims all interfaces, and finds the target data endpoint.
     *
     * This method iterates through the device's interfaces and their endpoints to find the first
     * suitable Interrupt IN endpoint, which is expected to carry the stylus data.
     *
     * @param device The `UsbDevice` to set up.
     */
    private fun setupDevice(device: UsbDevice) {
        Log.d(TAG, "setupDevice() for ${device.deviceName}. Interface count: ${device.interfaceCount}")

        deviceConnection = usbManager.openDevice(device)
        if (deviceConnection == null) {
            Log.e(TAG, "Failed to open device connection.")
            return
        }

        // Claim all available interfaces to ensure exclusive access.
        for (i in 0 until device.interfaceCount) {
            val intf = device.getInterface(i)
            val claimResult = deviceConnection?.claimInterface(intf, true)
            if (claimResult == true) {
                Log.d(TAG, "Successfully claimed interface $i.")

                // Find the first available Interrupt IN endpoint.
                if (dataEndpoint == null) {
                    for (j in 0 until intf.endpointCount) {
                        val ep = intf.getEndpoint(j)
                        if (ep.type == UsbConstants.USB_ENDPOINT_XFER_INT && ep.direction == UsbConstants.USB_DIR_IN) {
                            Log.d(TAG, "Found target data endpoint on interface $i!")
                            dataEndpoint = ep
                        }
                    }
                }
            } else {
                Log.w(TAG, "Failed to claim interface $i.")
            }
        }

        if (dataEndpoint != null) {
            startReading(dataEndpoint!!)
        } else {
            Log.e(TAG, "No suitable data endpoint found after checking all interfaces.")
            stop()
        }
    }

    /**
     * Decodes the raw tilt value from the device.
     *
     * The format is assumed to be a 7-bit signed integer, where the 7th bit is the sign.
     *
     * @param raw The raw integer value from the USB packet.
     * @return A signed integer representing the tilt, from -64 to 63.
     */
    private fun decodeTilt(raw: Int): Int {
        val magnitude = raw and 0x3F         // 6-bit magnitude
        return if ((raw and 0x40) != 0) {
            magnitude - 64                   // Negative tilt value
        } else {
            magnitude                        // Positive tilt value
        }
    }

    /**
     * Starts a continuous loop to read and process data from the USB endpoint.
     *
     * This function runs in a coroutine on an I/O thread. It reads data packets, parses them
     * according to a presumed format, and invokes the `onStylusData` callback on the main thread.
     *
     * @param endpoint The `UsbEndpoint` to read data from.
     */
    private fun startReading(endpoint: UsbEndpoint) {
        Log.d(TAG, "startReading() on endpoint with max packet size: ${endpoint.maxPacketSize}")
        running = true
        readJob?.cancel()
        readJob = ioScope.launch {
            val buffer = ByteArray(endpoint.maxPacketSize)
            while (running) {
                val conn = deviceConnection
                if (conn == null) {
                    Log.e(TAG, "Device connection is null, stopping reader.")
                    break
                }
                // Read data from the USB endpoint. This is a blocking call.
                val bytesRead = conn.bulkTransfer(endpoint, buffer, buffer.size, 100)

                if (bytesRead > 0) {
                    // The first byte seems to indicate the report type.
                    if (buffer[0].toInt() == 0x10) {

                        val secondByte = buffer[1].toInt() and 0xFF

                        // The second byte seems to indicate if the pen is in range.
                        when (secondByte) {
                            0x60 -> inNaNMode = false // Pen is in range and providing data.
                            0x40 -> inNaNMode = true  // Pen is lifted or out of range.
                        }

                        if (inNaNMode) {
                            // If out of range, send nulls to signal this state.
                            withContext(Dispatchers.Main) {
                                onStylusData(null, null, null, null, null)
                            }
                            continue
                        }

                        // --- Packet Parsing ---
                        // The following byte indices are based on observation and may need adjustment
                        // for different devices.

                        val x = ((buffer[3].toInt() and 0xFF) shl 8) or (buffer[2].toInt() and 0xFF)
                        val y = ((buffer[6].toInt() and 0xFF) shl 8) or (buffer[5].toInt() and 0xFF)
                        val pressure = ((buffer[9].toInt() and 0xFF) shl 8) or (buffer[8].toInt() and 0xFF)

                        // Tilt data from bytes 10 and 11.
                        val rawTiltX = buffer[10].toInt() and 0xFF
                        val rawTiltY = buffer[11].toInt() and 0xFF

                        val tiltX = decodeTilt(rawTiltX)
                        val tiltY = decodeTilt(rawTiltY)

                        // Switch to the main thread to update UI or renderer state.
                        withContext(Dispatchers.Main) {
                            onStylusData(x, y, pressure, tiltX, tiltY)
                        }
                    }
                }
            }
            Log.d(TAG, "Stopped reading loop.")
        }
    }

    /**
     * Stops the data reading loop and releases all USB resources.
     *
     * This cancels the coroutine, closes the device connection, and nullifies references
     * to prevent memory leaks.
     */
    fun stop() {
        Log.d(TAG, "stop() called")
        running = false
        readJob?.cancel()
        readJob = null
        deviceConnection?.close()
        deviceConnection = null
        dataEndpoint = null
        inNaNMode = false
    }

    companion object {
        private const val TAG = "UsbHandler"
    }
}
