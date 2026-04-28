package com.afun9.bookreader.tts

import com.afun9.bookreader.reader.TextSegment
import com.afun9.bookreader.voice.VoiceProfile

class OnnxTtsEngine(
    private val modelAssetRoot: String,
) : TtsEngine {
    fun manifestAssetPath(): String = "$modelAssetRoot/model/tts/optimized_manifest.json"

    override suspend fun generate(
        segment: TextSegment,
        voice: VoiceProfile,
        speed: Float,
    ): GeneratedAudio {
        throw UnsupportedOperationException(
            "ONNX generation binding will load ${manifestAssetPath()} and generate segment ${segment.id}.",
        )
    }
}
