package com.afun9.bookreader.reader

class ReadingSegmenter(
    private val maxChars: Int = 120,
) {
    init {
        require(maxChars > 0) { "maxChars must be positive" }
    }

    private val sentenceEndings = setOf('。', '！', '？', '!', '?', '.', ';', '；')
    private val closingPunctuation = setOf('”', '’', '"', '\'', '）', ')', '】', ']', '》', '』', '」')

    fun segmentChapter(chapterId: String, content: String): List<TextSegment> {
        val output = mutableListOf<TextSegment>()
        var searchOffset = 0
        splitParagraphs(content).forEach { paragraph ->
            splitSentences(paragraph).forEach { sentence ->
                splitHard(sentence).forEach { part ->
                    val start = content.indexOf(part, searchOffset)
                    check(start >= 0) { "Segment text not found in source content" }
                    val end = start + part.length
                    output += TextSegment(
                        id = "$chapterId-${output.size}",
                        chapterId = chapterId,
                        index = output.size,
                        startOffset = start,
                        endOffset = end,
                        text = part,
                    )
                    searchOffset = end
                }
            }
        }
        return output
    }

    private fun splitParagraphs(content: String): List<String> =
        content.split(Regex("""\n\s*\n|\r\n\s*\r\n"""))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

    private fun splitSentences(paragraph: String): List<String> {
        val sentences = mutableListOf<String>()
        val builder = StringBuilder()
        var sentenceEndingPending = false
        paragraph.forEach { char ->
            if (sentenceEndingPending && char !in closingPunctuation) {
                sentences += builder.toString().trim()
                builder.clear()
                sentenceEndingPending = false
            }
            builder.append(char)
            if (char in sentenceEndings) {
                sentenceEndingPending = true
            }
        }
        if (builder.isNotBlank()) {
            sentences += builder.toString().trim()
        }
        return sentences
    }

    private fun splitHard(sentence: String): List<String> {
        if (sentence.length <= maxChars) return listOf(sentence)
        return sentence.chunked(maxChars)
    }
}
