package com.afun9.bookreader.library

interface BookImporter {
    fun importText(fileName: String, content: String): ImportedBook
}
