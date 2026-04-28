package com.afun9.bookreader.voice

import org.junit.Assert.assertEquals
import org.junit.Test

class VoiceRuleResolverTest {
    @Test
    fun characterRuleBeatsDialogueRule() {
        val resolver = VoiceRuleResolver(
            listOf(
                VoiceRule(DialogueRule, "dialogue"),
                VoiceRule(CharacterRule("李明"), "li-ming"),
                VoiceRule(NarratorRule, "narrator"),
            ),
        )

        val voiceProfileId = resolver.resolve(
            SpeechContext(characterName = "李明", quoted = true, innerMonologue = false),
        )

        assertEquals("li-ming", voiceProfileId)
    }

    @Test
    fun narratorRuleIsFallback() {
        val resolver = VoiceRuleResolver(
            listOf(
                VoiceRule(DialogueRule, "dialogue"),
                VoiceRule(InnerMonologueRule, "inner"),
                VoiceRule(NarratorRule, "narrator"),
            ),
        )

        val voiceProfileId = resolver.resolve(
            SpeechContext(characterName = null, quoted = false, innerMonologue = false),
        )

        assertEquals("narrator", voiceProfileId)
    }

    @Test
    fun innerMonologueRuleBeatsNarratorRule() {
        val resolver = VoiceRuleResolver(
            listOf(
                VoiceRule(NarratorRule, "narrator"),
                VoiceRule(InnerMonologueRule, "inner"),
            ),
        )

        val voiceProfileId = resolver.resolve(
            SpeechContext(characterName = null, quoted = false, innerMonologue = true),
        )

        assertEquals("inner", voiceProfileId)
    }

    @Test
    fun dialogueRuleBeatsNarratorRule() {
        val resolver = VoiceRuleResolver(
            listOf(
                VoiceRule(NarratorRule, "narrator"),
                VoiceRule(DialogueRule, "dialogue"),
            ),
        )

        val voiceProfileId = resolver.resolve(
            SpeechContext(characterName = null, quoted = true, innerMonologue = false),
        )

        assertEquals("dialogue", voiceProfileId)
    }
}
