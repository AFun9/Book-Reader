package com.afun9.bookreader.library

import java.security.MessageDigest

class TxtBookImporter : BookImporter {
    private val chapterPattern = Regex("""(?m)^[ \t]*(第[一二三四五六七八九十百千万0-9]+章)([ \t]*.*)$""")

    override fun importText(fileName: String, content: String): ImportedBook {
        val title = fileName.substringBeforeLast('.').trim().ifBlank { "未命名小说" }
        val bookId = stableId("$fileName:${content.length}:$content")
        val chapters = splitChapters(bookId, content)
        return ImportedBook(
            book = Book(
                id = bookId,
                title = title,
                author = null,
                coverPath = null,
                sourcePath = fileName,
            ),
            chapters = chapters,
        )
    }

    private fun splitChapters(bookId: String, content: String): List<Chapter> {
        val matches = chapterPattern.findAll(content)
            .filter(::isChapterHeading)
            .toList()
        if (matches.isEmpty()) {
            return listOf(
                Chapter(
                    id = "$bookId-0",
                    bookId = bookId,
                    index = 0,
                    title = "正文",
                    content = content.trim(),
                ),
            )
        }

        val chapters = mutableListOf<Chapter>()
        val preface = content.substring(0, matches.first().range.first).trim()
        if (preface.isNotEmpty()) {
            chapters += Chapter(
                id = "$bookId-0",
                bookId = bookId,
                index = 0,
                title = "序章",
                content = preface,
            )
        }

        matches.forEachIndexed { matchIndex, match ->
            val chapterIndex = chapters.size
            val title = "${match.groupValues[1]}${match.groupValues[2]}".trim()
            val bodyStart = match.range.last + 1
            val bodyEnd = matches.getOrNull(matchIndex + 1)?.range?.first ?: content.length
            chapters += Chapter(
                id = "$bookId-$chapterIndex",
                bookId = bookId,
                index = chapterIndex,
                title = title,
                content = content.substring(bodyStart, bodyEnd).trim(),
            )
        }
        return chapters
    }

    private fun isChapterHeading(match: MatchResult): Boolean {
        val suffix = match.groupValues[2]
        val title = "${match.groupValues[1]}$suffix".trim()
        return suffix.isBlank() ||
            suffix.first().isWhitespace() ||
            title.lastOrNull() !in sentenceEndPunctuation
    }

    private fun stableId(input: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(input.toByteArray())
        return digest.take(8).joinToString("") { "%02x".format(it) }
    }

    private companion object {
        val sentenceEndPunctuation = setOf('。', '！', '？', '!', '?', '；', ';', '，', ',')
    }
}
