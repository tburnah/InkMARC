package com.example.inkmarc_capture

import android.content.Context
import android.util.Log
import java.io.File
import java.io.IOException

object ParticipantIdManager {

    private const val FILENAME = "participant_id.txt"
    private var participantId: Int? = null

    fun getParticipantId(context: Context): Int {
        return participantId ?: synchronized(this) {
            participantId ?: loadAndIncrementParticipantId(context.applicationContext).also {
                participantId = it
            }
        }
    }

    private fun loadAndIncrementParticipantId(context: Context): Int {
        val file = File(context.filesDir, FILENAME)
        val currentId: Int = try {
            if (file.exists()) {
                file.readText().trim().toInt()
            } else {
                700
            }
        } catch (e: Exception) { // Covers IOException and NumberFormatException
            Log.e("ParticipantIdManager", "Error reading participant ID, defaulting to 1", e)
            700
        }

        try {
            file.writeText((currentId + 1).toString())
        } catch (e: IOException) {
            Log.e("ParticipantIdManager", "Failed to write next participant ID", e)
        }

        return currentId
    }
}
