package com.afun9.bookreader.reader

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ReadingSegmenterTest {
    @Test
    fun splitsLongParagraphBySentenceBoundary() {
        val text = "第一句很短。第二句也很短！第三句继续推进剧情？"
        val segments = ReadingSegmenter(maxChars = 12).segmentChapter("chapter-1", text)

        assertEquals(3, segments.size)
        assertEquals("第一句很短。", segments[0].text)
    }

    @Test
    fun hardSplitsVeryLongSentence() {
        val text = "这是一句没有任何标点但是非常非常非常非常非常非常长的句子"
        val segments = ReadingSegmenter(maxChars = 10).segmentChapter("chapter-1", text)

        assertTrue(segments.size > 1)
        assertTrue(segments.all { it.text.length <= 10 })
        segments.forEach { segment ->
            assertEquals(segment.text, text.substring(segment.startOffset, segment.endOffset))
        }
    }

    @Test
    fun offsetsAreMonotonicAndPointToOriginalText() {
        val text = """
            第一段第一句。第一段第二句。

            第二段第一句！
        """.trimIndent()

        val segments = ReadingSegmenter(maxChars = 20).segmentChapter("chapter-1", text)

        assertEquals(3, segments.size)
        segments.forEachIndexed { index, segment ->
            assertEquals("chapter-1-$index", segment.id)
            assertEquals(index, segment.index)
            assertEquals(segment.text, text.substring(segment.startOffset, segment.endOffset))
        }
        assertTrue(segments.zipWithNext().all { (left, right) -> left.endOffset <= right.startOffset })
    }

    @Test
    fun keepsClosingQuoteWithQuotedSentence() {
        val text = "他说：“你好。”她笑了。"
        val segments = ReadingSegmenter(maxChars = 20).segmentChapter("chapter-1", text)

        assertEquals(listOf("他说：“你好。”", "她笑了。"), segments.map { it.text })
        segments.forEach { segment ->
            assertEquals(segment.text, text.substring(segment.startOffset, segment.endOffset))
        }
    }

    @Test
    fun keepsCornerClosingQuoteWithQuotedSentence() {
        val text = "他说：「你好。」她笑了。"
        val segments = ReadingSegmenter(maxChars = 20).segmentChapter("chapter-1", text)

        assertEquals(listOf("他说：「你好。」", "她笑了。"), segments.map { it.text })
    }

    @Test
    fun splitsEnglishPeriodAndSemicolon() {
        val text = "Hello. World; Done."
        val segments = ReadingSegmenter(maxChars = 20).segmentChapter("chapter-1", text)

        assertEquals(listOf("Hello.", "World;", "Done."), segments.map { it.text })
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonPositiveMaxChars() {
        ReadingSegmenter(maxChars = 0)
    }
}
