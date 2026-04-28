package com.afun9.bookreader.voice

enum class VoiceKind {
    BuiltIn,
    Cloned,
}

data class VoiceProfile(
    val id: String,
    val name: String,
    val kind: VoiceKind,
    val referenceAudioPath: String? = null,
)

data class CharacterCandidate(
    val id: String,
    val displayName: String,
    val occurrences: Int,
    val ignored: Boolean = false,
)

sealed interface VoiceRuleType

data object NarratorRule : VoiceRuleType

data object DialogueRule : VoiceRuleType

data object InnerMonologueRule : VoiceRuleType

data class CharacterRule(val characterName: String) : VoiceRuleType

data class VoiceRule(
    val type: VoiceRuleType,
    val voiceProfileId: String,
)

data class SpeechContext(
    val characterName: String?,
    val quoted: Boolean,
    val innerMonologue: Boolean,
)
