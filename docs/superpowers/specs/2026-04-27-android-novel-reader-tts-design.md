# Android Novel Reader With Offline TTS Design

## Goal

Build an Android novel reader that works as a complete local reading app and uses the optimized MOSS TTS Nano ONNX package for fully offline audiobook generation.

The first version is a balanced MVP: it must feel like a real reader, not only a TTS demo. It includes local book import, bookshelf management, reading settings, offline listening, voice cloning, and rule-based multi-voice narration.

## Scope

In scope:

- Android app.
- Local-only import. No book store.
- TXT and EPUB import.
- Bookshelf with multiple books, covers, progress, search, sort, and delete.
- Reading page with common reader interactions.
- Page modes: simulated page turn, horizontal paging, vertical scroll. Default is horizontal paging.
- Reader settings: font, font size, line spacing, paragraph spacing, margins, theme, night mode, brightness, keep screen awake, volume-key paging.
- Fully offline TTS with the built-in `moss_tts_nano_fp16_graphopt` model package.
- Segment-level streaming audiobook generation to control autoregressive KV cache growth.
- Background playback, playback queue, current segment highlight, speed control, sleep timer, and manual chapter caching.
- Voice Studio with built-in voices, cloned voices, rule assignment, preview, and cache invalidation prompts.
- Semi-automatic character detection: rules extract candidates and users confirm, merge, rename, ignore, and assign voices.

Out of scope for the first version:

- Online book store.
- Cloud sync.
- PDF support.
- Social comments or shared annotations.
- Fully automatic speaker attribution across the whole novel.
- Cross-device model download flow. The model is bundled with the app.

## Product Structure

The app has six major modules.

### Library

Library owns local book import and bookshelf management.

TXT import should detect text encoding, parse title and author when possible, split chapters, and generate a default cover when no cover exists. EPUB import should read metadata, table of contents, and embedded cover when available.

The bookshelf displays each book with cover, title, author, reading progress, and recent reading state. It supports search, sorting, grouping, delete, and re-parse.

### Reader

Reader owns chapter rendering, pagination, scroll mode, reading progress, table of contents, bookmarks, and reading preferences.

The reading page uses a familiar mobile-reader structure: immersive text, light top navigation, bottom tool panel, and a settings sheet. The default reading mode is horizontal paging, with simulated page turn and vertical scroll available in settings.

When audiobook playback is active, the reader highlights the current generated segment. The first version highlights the current segment or paragraph, not word-level alignment.

### Audiobook

Audiobook owns playback, segment queues, generated audio cache, background playback, and synchronization between playback and the reading page.

The default generation strategy is hybrid:

- Generate the current segment first.
- Keep a small queue of the next 2 to 5 segments.
- Continue pre-generation only when device state allows it.
- Let users manually cache the current chapter.

The player supports play, pause, previous segment, next segment, playback speed, sleep timer, notification controls, and resume from last listened position.

### Voice Studio

Voice Studio owns built-in voices, cloned voices, voice previews, and narration rules.

Users can:

- Choose a narrator voice.
- Choose a default dialogue voice.
- Clone a voice from imported reference audio.
- Preview each voice.
- Assign voices to detected character candidates.
- Configure rule priority.

Voice rule priority is:

1. Character-specific rule.
2. Quoted dialogue rule.
3. Inner monologue or special text rule.
4. Narrator rule.

Changing voice settings or rules invalidates affected generated audio cache entries.

### TTS Engine

TTS Engine wraps the bundled optimized ONNX package:

`optimized_export_workspace/runs/moss_tts_nano_fp16_graphopt`

The engine runs fully offline through ONNX Runtime. The model is autoregressive, so the app must not generate a whole chapter in one pass. Instead, it cuts text into bounded generation units.

Generation unit rules:

- Prefer natural paragraphs.
- If a paragraph is too long, split on sentence punctuation.
- Preserve quoted dialogue boundaries where possible.
- If one sentence is still too long, hard split by a configurable character limit, initially around 80 to 150 Chinese characters.
- Run each unit independently and release KV cache after generation.

Only one active model generation task should run at a time. The queue prioritizes the currently playing segment, then near-future playback segments, then optional background cache jobs.

### Storage

Storage is a combination of Room database and local files.

Database entities include:

- Book.
- Chapter.
- Text segment.
- Reading progress.
- Bookmark.
- Voice profile.
- Cloned voice profile.
- Character candidate.
- Voice rule.
- TTS cache entry.
- App settings.

File storage includes:

- Imported book files or normalized copies.
- EPUB extracted cover assets.
- Generated default covers.
- Generated audio segments.
- Bundled model assets copied or mapped for ONNX Runtime.

TTS cache keys include:

- Text hash.
- Book id and chapter id.
- Segment id or text offset range.
- Model version.
- Voice profile id.
- Voice rule version.
- Speech parameters such as speed and style.

## Main User Flows

### Import A Book

1. User taps import on the bookshelf.
2. User selects TXT or EPUB.
3. App parses metadata, cover, and chapters.
4. App shows a confirmation screen for title, author, cover, and grouping.
5. App saves normalized metadata and chapter index.
6. Book appears on the bookshelf.

### Read A Book

1. User opens a book from the bookshelf.
2. Reader opens at the last reading position.
3. User pages horizontally by default.
4. User can open table of contents, progress panel, settings, bookmarks, or listening controls.
5. Reading progress is saved continuously at chapter and offset level.

### Listen To A Book

1. User taps listen from the reader page.
2. App resolves the active voice rules and current text segment.
3. Audiobook queue asks TTS Engine to generate current segment.
4. Playback starts when the first segment is ready.
5. The app pre-generates nearby segments.
6. Reader highlights the currently playing segment.
7. User can cache the current chapter manually.

### Configure Voices

1. User opens Voice Studio.
2. User chooses built-in voices or creates cloned voices from reference audio.
3. App extracts character candidates from current book.
4. User confirms, merges, renames, ignores, and assigns voices.
5. App increments the rule set version and invalidates affected cache entries.

## Performance And Device Policy

Because the model is bundled and runs offline, the app should manage device resources carefully.

Recommended policies:

- Keep only one TTS generation task active.
- Limit background pre-generation when battery is low.
- Pause optional background caching when temperature is high.
- Allow foreground current-segment generation even when background caching is paused.
- Surface clear status: ready, generating, cached, waiting, paused by device state.
- Keep cache size configurable and provide manual cleanup.

## Error Handling

Import errors:

- Unsupported file type.
- Encoding detection failure.
- EPUB metadata parse failure.
- Empty or malformed book.

Reader errors:

- Missing chapter.
- Failed pagination after settings change.

TTS errors:

- Model initialization failure.
- Segment generation failure.
- Insufficient storage for cache.
- Voice clone input invalid or too short.

Each error should have a user-facing recovery path, such as retry, re-import, skip segment, use default voice, or clear cache.

## Testing Strategy

Parser tests:

- TXT encoding and chapter parsing.
- EPUB metadata, cover, and table of contents parsing.
- Long paragraph segmentation.
- Dialogue boundary segmentation.

Reader tests:

- Progress persistence.
- Page mode switching.
- Settings persistence.
- Chapter navigation.

TTS tests:

- Segment queue priority.
- Cache key invalidation when voice rules change.
- Generation unit length boundaries.
- Offline model load smoke test.
- Playback starts after first segment is ready.

Voice tests:

- Voice profile creation.
- Clone profile import validation.
- Character candidate merge, rename, ignore, and assignment.
- Rule priority resolution.

Storage tests:

- Book deletion cleans dependent records and optional audio cache.
- Cache cleanup respects active playback.
- Model version change invalidates incompatible cache entries.

## Open Product Choices

The design is ready for implementation planning. Before implementation, the team may still choose exact Android stack details such as Jetpack Compose versus XML views, ONNX Runtime packaging strategy, database schema names, and exact UI visual style.
