package com.afun9.bookreader.library

import java.io.InputStream
import java.util.zip.ZipInputStream

class EpubBookImporter {
    fun importBytes(fileName: String, input: InputStream): Result<ImportedBook> {
        return runCatching {
            ZipInputStream(input).use { zip ->
                val firstEntry = zip.nextEntry ?: error("Empty EPUB archive: $fileName")
                if (firstEntry.name.isBlank()) {
                    error("Malformed EPUB archive: $fileName")
                }
            }
            error("EPUB metadata parsing is not complete: $fileName")
        }
    }
}
