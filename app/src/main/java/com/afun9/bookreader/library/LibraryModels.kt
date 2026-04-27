package com.afun9.bookreader.library

data class Book(
    val id: String,
    val title: String,
    val author: String?,
    val coverPath: String?,
    val sourcePath: String,
    val progress: Float = 0f,
)

data class Chapter(
    val id: String,
    val bookId: String,
    val index: Int,
    val title: String,
    val content: String,
)

data class ImportedBook(
    val book: Book,
    val chapters: List<Chapter>,
)
