package com.afun9.bookreader.reader

enum class PageMode {
    SimulatedTurn,
    HorizontalPaging,
    VerticalScroll,
}

data class ReaderSettings(
    val fontFamily: String = "system",
    val fontSizeSp: Int = 20,
    val lineSpacing: Float = 1.35f,
    val paragraphSpacingDp: Int = 12,
    val marginDp: Int = 18,
    val pageMode: PageMode = PageMode.HorizontalPaging,
    val nightMode: Boolean = false,
    val keepScreenAwake: Boolean = false,
    val volumeKeyPaging: Boolean = true,
)

data class TextSegment(
    val id: String,
    val chapterId: String,
    val index: Int,
    val startOffset: Int,
    val endOffset: Int,
    val text: String,
)
