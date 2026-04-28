package com.afun9.bookreader.voice

class VoiceRuleResolver(
    private val rules: List<VoiceRule>,
) {
    fun resolve(context: SpeechContext): String {
        val characterName = context.characterName
        if (characterName != null) {
            rules.firstOrNull { it.type == CharacterRule(characterName) }
                ?.let { return it.voiceProfileId }
        }
        if (context.quoted) {
            rules.firstOrNull { it.type == DialogueRule }
                ?.let { return it.voiceProfileId }
        }
        if (context.innerMonologue) {
            rules.firstOrNull { it.type == InnerMonologueRule }
                ?.let { return it.voiceProfileId }
        }
        return rules.first { it.type == NarratorRule }.voiceProfileId
    }
}
