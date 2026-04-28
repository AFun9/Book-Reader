package com.afun9.bookreader.audiobook

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AudiobookQueueTest {
    @Test
    fun prioritizesCurrentSegmentThenNearbySegments() {
        val items = AudiobookQueue(prefetchCount = 3)
            .plan(chapterId = "c1", currentIndex = 10, segmentCount = 20)

        assertEquals(listOf(10, 11, 12, 13), items.map { it.segmentIndex })
        assertEquals(listOf(0, 1, 2, 3), items.map { it.priority })
        assertTrue(items.all { it.chapterId == "c1" })
    }

    @Test
    fun cacheKeyChangesWhenRuleVersionChanges() {
        val baseKey = TtsCacheKey(
            textHash = "text-abc",
            modelVersion = "moss-0.1",
            voiceProfileId = "voice-1",
            ruleVersion = "rules-1",
            speed = 1.0f,
        )
        val changedRuleKey = baseKey.copy(ruleVersion = "rules-2")

        assertNotEquals(baseKey.stableKey(), changedRuleKey.stableKey())
    }

    @Test
    fun cacheKeyChangesWhenSpeedChanges() {
        val baseKey = TtsCacheKey("text-abc", "moss-0.1", "voice-1", "rules-1", speed = 1.0f)

        assertNotEquals(baseKey.stableKey(), baseKey.copy(speed = 1.1f).stableKey())
    }

    @Test
    fun cacheKeySerializationIsNotConfusedByDelimiters() {
        val first = TtsCacheKey("a|b", "c", "d", "e", speed = 1.0f)
        val second = TtsCacheKey("a", "b|c", "d", "e", speed = 1.0f)

        assertNotEquals(first.stableKey(), second.stableKey())
    }

    @Test
    fun stableKeyIsDeterministicHexSha256() {
        val key = TtsCacheKey("text-abc", "moss-0.1", "voice-1", "rules-1", speed = 1.0f)
            .stableKey()

        assertEquals(64, key.length)
        assertTrue(key.all { it in '0'..'9' || it in 'a'..'f' })
        assertEquals(key, TtsCacheKey("text-abc", "moss-0.1", "voice-1", "rules-1", 1.0f).stableKey())
    }

    @Test
    fun clampsPlanNearEndOfChapter() {
        val items = AudiobookQueue(prefetchCount = 3)
            .plan(chapterId = "c1", currentIndex = 18, segmentCount = 20)

        assertEquals(listOf(18, 19), items.map { it.segmentIndex })
        assertEquals(listOf(0, 1), items.map { it.priority })
    }

    @Test
    fun returnsEmptyPlanForInvalidRanges() {
        val queue = AudiobookQueue(prefetchCount = 3)

        assertEquals(emptyList<AudiobookQueueItem>(), queue.plan("c1", currentIndex = 0, segmentCount = 0))
        assertEquals(emptyList<AudiobookQueueItem>(), queue.plan("c1", currentIndex = -1, segmentCount = 20))
        assertEquals(emptyList<AudiobookQueueItem>(), queue.plan("c1", currentIndex = 20, segmentCount = 20))
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNegativePrefetchCount() {
        AudiobookQueue(prefetchCount = -1)
    }
}
