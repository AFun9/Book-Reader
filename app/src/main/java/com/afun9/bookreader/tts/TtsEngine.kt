package com.afun9.bookreader.tts

import com.afun9.bookreader.reader.TextSegment
import com.afun9.bookreader.voice.VoiceProfile

data class GeneratedAudio(
    val segmentId: String,
    val audioPath: String,
    val durationMs: Long,
)

interface TtsEngine {
    suspend fun generate(segment: TextSegment, voice: VoiceProfile, speed: Float): GeneratedAudio
}
