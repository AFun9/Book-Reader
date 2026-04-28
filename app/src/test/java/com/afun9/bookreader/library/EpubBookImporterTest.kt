package com.afun9.bookreader.library

import java.io.ByteArrayInputStream
import org.junit.Assert.assertTrue
import org.junit.Test

class EpubBookImporterTest {
    @Test
    fun rejectsMalformedEpub() {
        val result = EpubBookImporter()
            .importBytes("bad.epub", ByteArrayInputStream("not a zip".toByteArray()))

        assertTrue(result.isFailure)
    }
}
