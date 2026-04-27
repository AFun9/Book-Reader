package com.afun9.bookreader.library

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

class TxtBookImporterTest {
    private val importer = TxtBookImporter()

    @Test
    fun importTextSplitsChineseChapterHeadings() {
        val content = """
            第一章 初见
            雨落在青石巷。
            他第一次看见她。

            第二章 风起
            长街吹起晚风。
            灯火次第亮起。
        """.trimIndent()

        val importedBook = importer.importText("sample.txt", content)

        assertEquals("sample", importedBook.book.title)
        assertEquals(null, importedBook.book.author)
        assertEquals(null, importedBook.book.coverPath)
        assertEquals("sample.txt", importedBook.book.sourcePath)
        assertEquals(0f, importedBook.book.progress)
        assertEquals(2, importedBook.chapters.size)

        val firstChapter = importedBook.chapters[0]
        assertEquals(importedBook.book.id, firstChapter.bookId)
        assertEquals(0, firstChapter.index)
        assertEquals("第一章 初见", firstChapter.title)
        assertEquals(
            """
                雨落在青石巷。
                他第一次看见她。
            """.trimIndent(),
            firstChapter.content,
        )

        val secondChapter = importedBook.chapters[1]
        assertEquals(importedBook.book.id, secondChapter.bookId)
        assertEquals(1, secondChapter.index)
        assertEquals("第二章 风起", secondChapter.title)
        assertEquals(
            """
                长街吹起晚风。
                灯火次第亮起。
            """.trimIndent(),
            secondChapter.content,
        )
        assertNotEquals(firstChapter.id, secondChapter.id)
    }

    @Test
    fun importTextCreatesSingleBodyChapterWhenNoHeadingExists() {
        val content = """

            这是没有章节标题的正文。
            它应当作为一个章节导入。

        """.trimIndent()

        val importedBook = importer.importText("plain.txt", content)

        assertEquals("plain", importedBook.book.title)
        assertEquals(1, importedBook.chapters.size)
        assertEquals("正文", importedBook.chapters.single().title)
        assertEquals(
            """
                这是没有章节标题的正文。
                它应当作为一个章节导入。
            """.trimIndent(),
            importedBook.chapters.single().content,
        )
    }

    @Test
    fun importTextUsesStableBookIdForSameFileAndContent() {
        val content = """
            第一章 初见
            相同的书籍内容。
        """.trimIndent()

        val firstImport = importer.importText("stable.txt", content)
        val secondImport = importer.importText("stable.txt", content)

        assertEquals(firstImport.book.id, secondImport.book.id)
    }

    @Test
    fun importTextPreservesPrefaceBeforeFirstHeading() {
        val content = """
            作者序
            这是序言内容。

            第一章 初见
            正文章节内容。
        """.trimIndent()

        val importedBook = importer.importText("preface.txt", content)

        assertEquals(2, importedBook.chapters.size)
        assertEquals("序章", importedBook.chapters[0].title)
        assertEquals(
            """
                作者序
                这是序言内容。
            """.trimIndent(),
            importedBook.chapters[0].content,
        )
        assertEquals("第一章 初见", importedBook.chapters[1].title)
    }

    @Test
    fun importTextSplitsIndentedChapterHeadings() {
        val content = """
              第一章 初见
            缩进标题也应该识别。
              第二章 风起
            第二章正文。
        """.trimIndent()

        val importedBook = importer.importText("indented.txt", content)

        assertEquals(2, importedBook.chapters.size)
        assertEquals("第一章 初见", importedBook.chapters[0].title)
        assertEquals("第二章 风起", importedBook.chapters[1].title)
    }

    @Test
    fun importTextKeepsSentenceLikeLinesInsideChapterBody() {
        val content = """
            第一章 初见
            第二章正文。
            她合上书页。

            第三章 风起
            后续章节。
        """.trimIndent()

        val importedBook = importer.importText("sentence-like.txt", content)

        assertEquals(2, importedBook.chapters.size)
        assertEquals("第一章 初见", importedBook.chapters[0].title)
        assertEquals(
            """
                第二章正文。
                她合上书页。
            """.trimIndent(),
            importedBook.chapters[0].content,
        )
        assertEquals("第三章 风起", importedBook.chapters[1].title)
    }

    @Test
    fun importTextAllowsPunctuationInSeparatedChapterTitle() {
        val content = """
            第一章 你是谁？
            她没有回答。
        """.trimIndent()

        val importedBook = importer.importText("question.txt", content)

        assertEquals(1, importedBook.chapters.size)
        assertEquals("第一章 你是谁？", importedBook.chapters.single().title)
        assertEquals("她没有回答。", importedBook.chapters.single().content)
    }

    @Test
    fun importTextFallsBackToUntitledBookWhenFileNameIsBlank() {
        val importedBook = importer.importText("   .txt", "没有标题的内容")

        assertEquals("未命名小说", importedBook.book.title)
    }
}
