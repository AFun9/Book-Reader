package com.afun9.bookreader.audiobook

class AudiobookQueue(
    private val prefetchCount: Int = 3,
) {
    init {
        require(prefetchCount >= 0) { "prefetchCount must be non-negative" }
    }

    fun plan(
        chapterId: String,
        currentIndex: Int,
        segmentCount: Int,
    ): List<AudiobookQueueItem> {
        if (segmentCount <= 0 || currentIndex !in 0 until segmentCount) {
            return emptyList()
        }

        val lastIndex = currentIndex.toLong()
            .plus(prefetchCount.toLong())
            .coerceAtMost((segmentCount - 1).toLong())
            .toInt()
        return (currentIndex..lastIndex).mapIndexed { priority, segmentIndex ->
            AudiobookQueueItem(
                chapterId = chapterId,
                segmentIndex = segmentIndex,
                priority = priority,
            )
        }
    }
}
