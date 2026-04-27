package com.afun9.bookreader.library

import com.afun9.bookreader.reader.ReadingSegmenter
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

class XiyoujiSampleTest {
    @Test
    fun importsAndSegmentsLocalXiyoujiSample() {
        val samplePath = Path.of(
            System.getenv("XIYOUJI_TXT_PATH")
                ?: "/home/model/project/MOSS-TTS-Nano/assets/西游记.txt",
        )
        assumeTrue("西游记 sample not found at $samplePath", Files.exists(samplePath))

        val content = String(Files.readAllBytes(samplePath), StandardCharsets.UTF_8)
        val imported = TxtBookImporter().importText("西游记.txt", content)

        assertEquals("西游记", imported.book.title)
        assertTrue(imported.chapters.size >= 90)
        assertEquals("第一回：灵根育孕源流出，心性修持大道生", imported.chapters[1].title)

        val segments = ReadingSegmenter(maxChars = 120)
            .segmentChapter(imported.chapters[1].id, imported.chapters[1].content)

        assertTrue(segments.size > 20)
        assertTrue(segments.all { it.text.length <= 120 })
        segments.take(100).forEach { segment ->
            assertEquals(
                segment.text,
                imported.chapters[1].content.substring(segment.startOffset, segment.endOffset),
            )
        }
    }
}
