package com.afun9.bookreader.tts

import org.junit.Assert.assertEquals
import org.junit.Test

class OnnxTtsEngineTest {
    @Test
    fun exposesBundledModelAssetPath() {
        val engine = OnnxTtsEngine(modelAssetRoot = "models/moss_tts_nano_fp16_graphopt")

        assertEquals(
            "models/moss_tts_nano_fp16_graphopt/model/tts/optimized_manifest.json",
            engine.manifestAssetPath(),
        )
    }
}
