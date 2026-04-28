package com.afun9.bookreader.audiobook

import java.security.MessageDigest

data class AudiobookQueueItem(
    val chapterId: String,
    val segmentIndex: Int,
    val priority: Int,
)

data class TtsCacheKey(
    val textHash: String,
    val modelVersion: String,
    val voiceProfileId: String,
    val ruleVersion: String,
    val speed: Float,
) {
    fun stableKey(): String {
        val raw = listOf(textHash, modelVersion, voiceProfileId, ruleVersion, speed.toString())
            .joinToString(separator = "") { value -> "${value.length}:$value" }
        val digest = MessageDigest.getInstance("SHA-256").digest(raw.toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it) }
    }
}
