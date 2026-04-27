# Android Novel Reader Offline TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Android balanced MVP for a local novel reader with bundled offline MOSS TTS listening.

**Architecture:** Create a new Android project under `android_reader/` using one app module. Keep features separated into focused packages: library/import, reader, audiobook, voice, tts, and storage. Use a local model-copy Gradle task so release builds can bundle `optimized_export_workspace/runs/moss_tts_nano_fp16_graphopt` without committing the large model files to normal Git.

**Tech Stack:** Kotlin, Jetpack Compose, AndroidX Navigation, Room, Media3, ONNX Runtime Android, Kotlin coroutines, JUnit, Robolectric where needed.

---

## Scope Notes

This plan implements the first balanced MVP skeleton end to end:

- Android app shell with navigation.
- Book import domain for TXT and EPUB.
- Bookshelf UI.
- Reader UI with three reading modes.
- Segmenting rules for offline TTS.
- Audiobook queue and cache-key logic.
- Voice Studio models and rule resolution.
- Model asset packaging path.

Later plans can harden visual polish, real EPUB edge cases, production audio playback service details, and full ONNX inference binding.

## File Structure

Create:

- `android_reader/settings.gradle.kts` - Gradle project settings.
- `android_reader/build.gradle.kts` - root Gradle plugin versions.
- `android_reader/app/build.gradle.kts` - Android app module, dependencies, local model copy task.
- `android_reader/app/src/main/AndroidManifest.xml` - app manifest.
- `android_reader/app/src/main/java/com/afun9/bookreader/MainActivity.kt` - Compose entry point.
- `android_reader/app/src/main/java/com/afun9/bookreader/app/BookReaderApp.kt` - app navigation shell.
- `android_reader/app/src/main/java/com/afun9/bookreader/library/LibraryModels.kt` - book and chapter domain models.
- `android_reader/app/src/main/java/com/afun9/bookreader/library/BookImporter.kt` - import interface and result types.
- `android_reader/app/src/main/java/com/afun9/bookreader/library/TxtBookImporter.kt` - TXT parser.
- `android_reader/app/src/main/java/com/afun9/bookreader/library/EpubBookImporter.kt` - EPUB parser.
- `android_reader/app/src/main/java/com/afun9/bookreader/library/InMemoryLibraryRepository.kt` - MVP repository before Room integration.
- `android_reader/app/src/main/java/com/afun9/bookreader/reader/ReaderModels.kt` - reader state and settings.
- `android_reader/app/src/main/java/com/afun9/bookreader/reader/ReadingSegmenter.kt` - paragraph/sentence/character-limit segmentation.
- `android_reader/app/src/main/java/com/afun9/bookreader/audiobook/AudiobookModels.kt` - playback queue and cache models.
- `android_reader/app/src/main/java/com/afun9/bookreader/audiobook/AudiobookQueue.kt` - queue priority logic.
- `android_reader/app/src/main/java/com/afun9/bookreader/voice/VoiceModels.kt` - voices, clones, candidates, rules.
- `android_reader/app/src/main/java/com/afun9/bookreader/voice/VoiceRuleResolver.kt` - rule priority resolution.
- `android_reader/app/src/main/java/com/afun9/bookreader/tts/TtsEngine.kt` - TTS engine interface.
- `android_reader/app/src/main/java/com/afun9/bookreader/tts/OnnxTtsEngine.kt` - model asset discovery and generation boundary.
- `android_reader/app/src/main/java/com/afun9/bookreader/ui/LibraryScreen.kt` - bookshelf UI.
- `android_reader/app/src/main/java/com/afun9/bookreader/ui/ReaderScreen.kt` - reader UI.
- `android_reader/app/src/main/java/com/afun9/bookreader/ui/VoiceStudioScreen.kt` - voice configuration UI.
- `android_reader/app/src/test/java/com/afun9/bookreader/library/TxtBookImporterTest.kt`
- `android_reader/app/src/test/java/com/afun9/bookreader/library/EpubBookImporterTest.kt`
- `android_reader/app/src/test/java/com/afun9/bookreader/reader/ReadingSegmenterTest.kt`
- `android_reader/app/src/test/java/com/afun9/bookreader/audiobook/AudiobookQueueTest.kt`
- `android_reader/app/src/test/java/com/afun9/bookreader/voice/VoiceRuleResolverTest.kt`

Modify:

- `.gitignore` - ignore copied Android model assets and build outputs.

## Task 1: Android Project Scaffold

**Files:**
- Create: `android_reader/settings.gradle.kts`
- Create: `android_reader/build.gradle.kts`
- Create: `android_reader/app/build.gradle.kts`
- Create: `android_reader/app/src/main/AndroidManifest.xml`
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/MainActivity.kt`
- Modify: `.gitignore`

- [ ] **Step 1: Write scaffold files**

Create `android_reader/settings.gradle.kts`:

```kotlin
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "BookReader"
include(":app")
```

Create `android_reader/build.gradle.kts`:

```kotlin
plugins {
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "2.0.20" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.20" apply false
    id("com.google.devtools.ksp") version "2.0.20-1.0.25" apply false
}
```

Create `android_reader/app/build.gradle.kts`:

```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.afun9.bookreader"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.afun9.bookreader"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildFeatures {
        compose = true
    }

    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

val modelSourceDir = rootProject.layout.projectDirectory
    .dir("../optimized_export_workspace/runs/moss_tts_nano_fp16_graphopt/model")
val modelAssetsDir = layout.projectDirectory.dir("src/main/assets/models/moss_tts_nano_fp16_graphopt")

tasks.register<Copy>("copyOfflineTtsModel") {
    from(modelSourceDir)
    into(modelAssetsDir)
}

tasks.matching { it.name.startsWith("pre") && it.name.endsWith("Build") }.configureEach {
    dependsOn("copyOfflineTtsModel")
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.10.00"))
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.navigation:navigation-compose:2.8.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.6")
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    implementation("androidx.media3:media3-session:1.4.1")
    implementation("androidx.media3:media3-exoplayer:1.4.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    ksp("androidx.room:room-compiler:2.6.1")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
    testImplementation("androidx.test:core:1.6.1")
    testImplementation("org.robolectric:robolectric:4.13")
}
```

Create `android_reader/app/src/main/AndroidManifest.xml`:

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.READ_MEDIA_AUDIO" />
    <uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MEDIA_PLAYBACK" />

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="Book Reader"
        android:theme="@style/Theme.BookReader">
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

Create `android_reader/app/src/main/java/com/afun9/bookreader/MainActivity.kt`:

```kotlin
package com.afun9.bookreader

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.afun9.bookreader.app.BookReaderApp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            BookReaderApp()
        }
    }
}
```

Add these lines to `.gitignore`:

```gitignore
android_reader/.gradle/
android_reader/build/
android_reader/app/build/
android_reader/app/src/main/assets/models/
```

- [ ] **Step 2: Run Gradle help**

Run:

```bash
cd android_reader
gradle tasks
```

Expected: Gradle lists tasks. If `gradle` is unavailable, install or create a Gradle wrapper before continuing:

```bash
cd android_reader
gradle wrapper --gradle-version 8.7
```

- [ ] **Step 3: Commit scaffold**

```bash
git add .gitignore android_reader
git commit -m "feat: scaffold android reader app"
```

## Task 2: Compose Navigation Shell

**Files:**
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/app/BookReaderApp.kt`
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/ui/LibraryScreen.kt`
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/ui/ReaderScreen.kt`
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/ui/VoiceStudioScreen.kt`

- [ ] **Step 1: Create navigation shell**

Create `BookReaderApp.kt`:

```kotlin
package com.afun9.bookreader.app

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.afun9.bookreader.ui.LibraryScreen
import com.afun9.bookreader.ui.ReaderScreen
import com.afun9.bookreader.ui.VoiceStudioScreen

object Routes {
    const val Library = "library"
    const val Reader = "reader"
    const val VoiceStudio = "voice-studio"
}

@Composable
fun BookReaderApp() {
    val navController = rememberNavController()
    MaterialTheme {
        NavHost(navController = navController, startDestination = Routes.Library) {
            composable(Routes.Library) {
                LibraryScreen(
                    onOpenBook = { navController.navigate(Routes.Reader) },
                    onOpenVoiceStudio = { navController.navigate(Routes.VoiceStudio) },
                )
            }
            composable(Routes.Reader) {
                ReaderScreen(
                    onBack = { navController.popBackStack() },
                    onOpenVoiceStudio = { navController.navigate(Routes.VoiceStudio) },
                )
            }
            composable(Routes.VoiceStudio) {
                VoiceStudioScreen(onBack = { navController.popBackStack() })
            }
        }
    }
}
```

Create `LibraryScreen.kt`:

```kotlin
package com.afun9.bookreader.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun LibraryScreen(
    onOpenBook: () -> Unit,
    onOpenVoiceStudio: () -> Unit,
) {
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("书架")
            Button(onClick = onOpenVoiceStudio) { Text("音色") }
        }
        Button(onClick = {}) { Text("导入 TXT / EPUB") }
        Card(Modifier.fillMaxWidth().clickable(onClick = onOpenBook)) {
            Column(Modifier.padding(16.dp)) {
                Text("示例小说")
                Text("阅读进度 32%")
            }
        }
    }
}
```

Create `ReaderScreen.kt`:

```kotlin
package com.afun9.bookreader.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ReaderScreen(
    onBack: () -> Unit,
    onOpenVoiceStudio: () -> Unit,
) {
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Button(onClick = onBack) { Text("返回") }
            Button(onClick = onOpenVoiceStudio) { Text("音色") }
        }
        Text("第一章")
        Text("这是阅读正文区域。默认左右滑页，设置中支持仿真翻页和上下滚动。")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {}) { Text("目录") }
            Button(onClick = {}) { Text("听书") }
            Button(onClick = {}) { Text("设置") }
        }
    }
}
```

Create `VoiceStudioScreen.kt`:

```kotlin
package com.afun9.bookreader.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun VoiceStudioScreen(onBack: () -> Unit) {
    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Button(onClick = onBack) { Text("返回") }
        Text("音色中心")
        Text("内置音色、克隆音色、旁白/对白/角色规则。")
        Button(onClick = {}) { Text("导入参考音频创建克隆音色") }
    }
}
```

- [ ] **Step 2: Build app**

Run:

```bash
cd android_reader
./gradlew assembleDebug
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 3: Commit navigation shell**

```bash
git add android_reader
git commit -m "feat: add reader navigation shell"
```

## Task 3: Library Domain And TXT Import

**Files:**
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/library/LibraryModels.kt`
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/library/BookImporter.kt`
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/library/TxtBookImporter.kt`
- Test: `android_reader/app/src/test/java/com/afun9/bookreader/library/TxtBookImporterTest.kt`

- [ ] **Step 1: Write failing TXT importer test**

Create `TxtBookImporterTest.kt`:

```kotlin
package com.afun9.bookreader.library

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TxtBookImporterTest {
    @Test
    fun importsTxtAndSplitsChapters() {
        val text = """
            第一章 初见
            这是第一章正文。
            第二章 风起
            这是第二章正文。
        """.trimIndent()

        val result = TxtBookImporter().importText("sample.txt", text)

        assertEquals("sample", result.book.title)
        assertEquals(2, result.chapters.size)
        assertEquals("第一章 初见", result.chapters[0].title)
        assertTrue(result.chapters[0].content.contains("这是第一章正文"))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.library.TxtBookImporterTest
```

Expected: FAIL because `TxtBookImporter` is not defined.

- [ ] **Step 3: Implement domain and TXT importer**

Create `LibraryModels.kt`:

```kotlin
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
```

Create `BookImporter.kt`:

```kotlin
package com.afun9.bookreader.library

interface BookImporter {
    fun importText(fileName: String, content: String): ImportedBook
}
```

Create `TxtBookImporter.kt`:

```kotlin
package com.afun9.bookreader.library

import java.security.MessageDigest

class TxtBookImporter : BookImporter {
    private val chapterPattern = Regex("""(?m)^(第[一二三四五六七八九十百千万0-9]+章\s*.*)$""")

    override fun importText(fileName: String, content: String): ImportedBook {
        val title = fileName.substringBeforeLast('.').ifBlank { "未命名小说" }
        val bookId = stableId("$fileName:${content.length}")
        val matches = chapterPattern.findAll(content).toList()
        val chapters = if (matches.isEmpty()) {
            listOf(Chapter("$bookId-0", bookId, 0, "正文", content.trim()))
        } else {
            matches.mapIndexed { index, match ->
                val start = match.range.first
                val end = matches.getOrNull(index + 1)?.range?.first ?: content.length
                val titleLine = match.value.trim()
                val body = content.substring(start, end).removePrefix(titleLine).trim()
                Chapter("$bookId-$index", bookId, index, titleLine, body)
            }
        }
        return ImportedBook(
            book = Book(bookId, title, author = null, coverPath = null, sourcePath = fileName),
            chapters = chapters,
        )
    }

    private fun stableId(input: String): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(input.toByteArray())
        return digest.take(8).joinToString("") { "%02x".format(it) }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.library.TxtBookImporterTest
```

Expected: PASS.

- [ ] **Step 5: Commit TXT import**

```bash
git add android_reader/app/src/main/java/com/afun9/bookreader/library android_reader/app/src/test/java/com/afun9/bookreader/library
git commit -m "feat: add txt book import domain"
```

## Task 4: Reading Segmenter For Offline TTS

**Files:**
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/reader/ReaderModels.kt`
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/reader/ReadingSegmenter.kt`
- Test: `android_reader/app/src/test/java/com/afun9/bookreader/reader/ReadingSegmenterTest.kt`

- [ ] **Step 1: Write failing segmenter tests**

Create `ReadingSegmenterTest.kt`:

```kotlin
package com.afun9.bookreader.reader

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ReadingSegmenterTest {
    @Test
    fun splitsLongParagraphBySentenceBoundary() {
        val text = "第一句很短。第二句也很短！第三句继续推进剧情？"
        val segments = ReadingSegmenter(maxChars = 12).segmentChapter("chapter-1", text)

        assertEquals(3, segments.size)
        assertEquals("第一句很短。", segments[0].text)
    }

    @Test
    fun hardSplitsVeryLongSentence() {
        val text = "这是一句没有任何标点但是非常非常非常非常非常非常长的句子"
        val segments = ReadingSegmenter(maxChars = 10).segmentChapter("chapter-1", text)

        assertTrue(segments.size > 1)
        assertTrue(segments.all { it.text.length <= 10 })
    }
}
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.reader.ReadingSegmenterTest
```

Expected: FAIL because reader package is not defined.

- [ ] **Step 3: Implement reader models and segmenter**

Create `ReaderModels.kt`:

```kotlin
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
```

Create `ReadingSegmenter.kt`:

```kotlin
package com.afun9.bookreader.reader

class ReadingSegmenter(
    private val maxChars: Int = 120,
) {
    private val sentenceEndings = setOf('。', '！', '？', '!', '?')

    fun segmentChapter(chapterId: String, content: String): List<TextSegment> {
        val output = mutableListOf<TextSegment>()
        var searchOffset = 0
        splitParagraphs(content).forEach { paragraph ->
            splitSentences(paragraph).forEach { sentence ->
                splitHard(sentence).forEach { part ->
                    val start = content.indexOf(part, searchOffset).coerceAtLeast(searchOffset)
                    val end = start + part.length
                    output += TextSegment(
                        id = "$chapterId-${output.size}",
                        chapterId = chapterId,
                        index = output.size,
                        startOffset = start,
                        endOffset = end,
                        text = part,
                    )
                    searchOffset = end
                }
            }
        }
        return output
    }

    private fun splitParagraphs(content: String): List<String> =
        content.split(Regex("""\n\s*\n|\r\n\s*\r\n"""))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

    private fun splitSentences(paragraph: String): List<String> {
        val sentences = mutableListOf<String>()
        val builder = StringBuilder()
        paragraph.forEach { char ->
            builder.append(char)
            if (char in sentenceEndings) {
                sentences += builder.toString().trim()
                builder.clear()
            }
        }
        if (builder.isNotBlank()) sentences += builder.toString().trim()
        return sentences
    }

    private fun splitHard(sentence: String): List<String> {
        if (sentence.length <= maxChars) return listOf(sentence)
        return sentence.chunked(maxChars)
    }
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.reader.ReadingSegmenterTest
```

Expected: PASS.

- [ ] **Step 5: Commit segmenter**

```bash
git add android_reader/app/src/main/java/com/afun9/bookreader/reader android_reader/app/src/test/java/com/afun9/bookreader/reader
git commit -m "feat: add tts reading segmenter"
```

## Task 5: Audiobook Queue And Cache Keys

**Files:**
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/audiobook/AudiobookModels.kt`
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/audiobook/AudiobookQueue.kt`
- Test: `android_reader/app/src/test/java/com/afun9/bookreader/audiobook/AudiobookQueueTest.kt`

- [ ] **Step 1: Write failing queue tests**

Create `AudiobookQueueTest.kt`:

```kotlin
package com.afun9.bookreader.audiobook

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

class AudiobookQueueTest {
    @Test
    fun prioritizesCurrentSegmentThenNearbySegments() {
        val queue = AudiobookQueue(prefetchCount = 3)
        val items = queue.plan(chapterId = "c1", currentIndex = 10, segmentCount = 20)

        assertEquals(listOf(10, 11, 12, 13), items.map { it.segmentIndex })
    }

    @Test
    fun cacheKeyChangesWhenRuleVersionChanges() {
        val first = TtsCacheKey("text", "model", "voice", "rules-1", speed = 1.0f)
        val second = first.copy(ruleVersion = "rules-2")

        assertNotEquals(first.stableKey(), second.stableKey())
    }
}
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.audiobook.AudiobookQueueTest
```

Expected: FAIL because audiobook package is not defined.

- [ ] **Step 3: Implement audiobook models and queue**

Create `AudiobookModels.kt`:

```kotlin
package com.afun9.bookreader.audiobook

import java.security.MessageDigest

data class AudiobookQueueItem(
    val chapterId: String,
    val segmentIndex: Int,
    val priority: Int,
)

data class TtsCacheKey(
    val textHash: String,
    val modelVersion: String,
    val voiceProfileId: String,
    val ruleVersion: String,
    val speed: Float,
) {
    fun stableKey(): String {
        val raw = listOf(textHash, modelVersion, voiceProfileId, ruleVersion, speed.toString()).joinToString("|")
        val digest = MessageDigest.getInstance("SHA-256").digest(raw.toByteArray())
        return digest.joinToString("") { "%02x".format(it) }
    }
}
```

Create `AudiobookQueue.kt`:

```kotlin
package com.afun9.bookreader.audiobook

class AudiobookQueue(
    private val prefetchCount: Int = 3,
) {
    fun plan(chapterId: String, currentIndex: Int, segmentCount: Int): List<AudiobookQueueItem> {
        val last = (currentIndex + prefetchCount).coerceAtMost(segmentCount - 1)
        return (currentIndex..last).mapIndexed { offset, segmentIndex ->
            AudiobookQueueItem(
                chapterId = chapterId,
                segmentIndex = segmentIndex,
                priority = offset,
            )
        }
    }
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.audiobook.AudiobookQueueTest
```

Expected: PASS.

- [ ] **Step 5: Commit audiobook queue**

```bash
git add android_reader/app/src/main/java/com/afun9/bookreader/audiobook android_reader/app/src/test/java/com/afun9/bookreader/audiobook
git commit -m "feat: add audiobook queue planning"
```

## Task 6: Voice Models And Rule Resolution

**Files:**
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/voice/VoiceModels.kt`
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/voice/VoiceRuleResolver.kt`
- Test: `android_reader/app/src/test/java/com/afun9/bookreader/voice/VoiceRuleResolverTest.kt`

- [ ] **Step 1: Write failing rule resolver tests**

Create `VoiceRuleResolverTest.kt`:

```kotlin
package com.afun9.bookreader.voice

import org.junit.Assert.assertEquals
import org.junit.Test

class VoiceRuleResolverTest {
    @Test
    fun characterRuleBeatsDialogueRule() {
        val resolver = VoiceRuleResolver(
            rules = listOf(
                VoiceRule(DialogueRule, "dialogue-voice"),
                VoiceRule(CharacterRule("李明"), "liming-voice"),
                VoiceRule(NarratorRule, "narrator-voice"),
            )
        )

        val resolved = resolver.resolve(SpeechContext(characterName = "李明", quoted = true, innerMonologue = false))

        assertEquals("liming-voice", resolved)
    }

    @Test
    fun narratorRuleIsFallback() {
        val resolver = VoiceRuleResolver(listOf(VoiceRule(NarratorRule, "narrator-voice")))

        assertEquals("narrator-voice", resolver.resolve(SpeechContext(null, quoted = false, innerMonologue = false)))
    }
}
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.voice.VoiceRuleResolverTest
```

Expected: FAIL because voice package is not defined.

- [ ] **Step 3: Implement voice models and resolver**

Create `VoiceModels.kt`:

```kotlin
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
```

Create `VoiceRuleResolver.kt`:

```kotlin
package com.afun9.bookreader.voice

class VoiceRuleResolver(
    private val rules: List<VoiceRule>,
) {
    fun resolve(context: SpeechContext): String {
        val character = context.characterName
        if (character != null) {
            rules.firstOrNull { it.type == CharacterRule(character) }?.let { return it.voiceProfileId }
        }
        if (context.quoted) {
            rules.firstOrNull { it.type == DialogueRule }?.let { return it.voiceProfileId }
        }
        if (context.innerMonologue) {
            rules.firstOrNull { it.type == InnerMonologueRule }?.let { return it.voiceProfileId }
        }
        return rules.first { it.type == NarratorRule }.voiceProfileId
    }
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.voice.VoiceRuleResolverTest
```

Expected: PASS.

- [ ] **Step 5: Commit voice rules**

```bash
git add android_reader/app/src/main/java/com/afun9/bookreader/voice android_reader/app/src/test/java/com/afun9/bookreader/voice
git commit -m "feat: add voice rule resolution"
```

## Task 7: TTS Engine Boundary And Model Asset Check

**Files:**
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/tts/TtsEngine.kt`
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/tts/OnnxTtsEngine.kt`
- Test: `android_reader/app/src/test/java/com/afun9/bookreader/tts/OnnxTtsEngineTest.kt`

- [ ] **Step 1: Write failing asset path test**

Create `OnnxTtsEngineTest.kt`:

```kotlin
package com.afun9.bookreader.tts

import org.junit.Assert.assertEquals
import org.junit.Test

class OnnxTtsEngineTest {
    @Test
    fun exposesBundledModelAssetPath() {
        val engine = OnnxTtsEngine(modelAssetRoot = "models/moss_tts_nano_fp16_graphopt")

        assertEquals("models/moss_tts_nano_fp16_graphopt/model/tts/optimized_manifest.json", engine.manifestAssetPath())
    }
}
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.tts.OnnxTtsEngineTest
```

Expected: FAIL because tts package is not defined.

- [ ] **Step 3: Implement engine boundary**

Create `TtsEngine.kt`:

```kotlin
package com.afun9.bookreader.tts

import com.afun9.bookreader.reader.TextSegment
import com.afun9.bookreader.voice.VoiceProfile

data class GeneratedAudio(
    val segmentId: String,
    val audioPath: String,
    val durationMs: Long,
)

interface TtsEngine {
    suspend fun generate(segment: TextSegment, voice: VoiceProfile, speed: Float): GeneratedAudio
}
```

Create `OnnxTtsEngine.kt`:

```kotlin
package com.afun9.bookreader.tts

import com.afun9.bookreader.reader.TextSegment
import com.afun9.bookreader.voice.VoiceProfile

class OnnxTtsEngine(
    private val modelAssetRoot: String,
) : TtsEngine {
    fun manifestAssetPath(): String = "$modelAssetRoot/model/tts/optimized_manifest.json"

    override suspend fun generate(segment: TextSegment, voice: VoiceProfile, speed: Float): GeneratedAudio {
        throw UnsupportedOperationException(
            "ONNX generation binding will load ${manifestAssetPath()} and generate segment ${segment.id}."
        )
    }
}
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.tts.OnnxTtsEngineTest
```

Expected: PASS.

- [ ] **Step 5: Commit TTS boundary**

```bash
git add android_reader/app/src/main/java/com/afun9/bookreader/tts android_reader/app/src/test/java/com/afun9/bookreader/tts
git commit -m "feat: add offline tts engine boundary"
```

## Task 8: EPUB Import Skeleton

**Files:**
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/library/EpubBookImporter.kt`
- Test: `android_reader/app/src/test/java/com/afun9/bookreader/library/EpubBookImporterTest.kt`

- [ ] **Step 1: Write failing unsupported/malformed EPUB test**

Create `EpubBookImporterTest.kt`:

```kotlin
package com.afun9.bookreader.library

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream

class EpubBookImporterTest {
    @Test
    fun rejectsMalformedEpub() {
        val result = EpubBookImporter().importBytes("bad.epub", ByteArrayInputStream("not a zip".toByteArray()))

        assertTrue(result.isFailure)
    }
}
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.library.EpubBookImporterTest
```

Expected: FAIL because `EpubBookImporter` is not defined.

- [ ] **Step 3: Implement malformed EPUB rejection**

Create `EpubBookImporter.kt`:

```kotlin
package com.afun9.bookreader.library

import java.io.InputStream
import java.util.zip.ZipInputStream

class EpubBookImporter {
    fun importBytes(fileName: String, input: InputStream): Result<ImportedBook> {
        return runCatching {
            ZipInputStream(input).use { zip ->
                val firstEntry = zip.nextEntry ?: error("Empty EPUB archive")
                if (firstEntry.name.isBlank()) error("Malformed EPUB archive")
            }
            error("EPUB metadata parsing is not complete")
        }
    }
}
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd android_reader
./gradlew testDebugUnitTest --tests com.afun9.bookreader.library.EpubBookImporterTest
```

Expected: PASS.

- [ ] **Step 5: Commit EPUB skeleton**

```bash
git add android_reader/app/src/main/java/com/afun9/bookreader/library/EpubBookImporter.kt android_reader/app/src/test/java/com/afun9/bookreader/library/EpubBookImporterTest.kt
git commit -m "feat: add epub import boundary"
```

## Task 9: Wire Domain Into MVP Screens

**Files:**
- Create: `android_reader/app/src/main/java/com/afun9/bookreader/library/InMemoryLibraryRepository.kt`
- Modify: `android_reader/app/src/main/java/com/afun9/bookreader/ui/LibraryScreen.kt`
- Modify: `android_reader/app/src/main/java/com/afun9/bookreader/ui/ReaderScreen.kt`
- Modify: `android_reader/app/src/main/java/com/afun9/bookreader/ui/VoiceStudioScreen.kt`

- [ ] **Step 1: Create repository**

Create `InMemoryLibraryRepository.kt`:

```kotlin
package com.afun9.bookreader.library

class InMemoryLibraryRepository {
    private val sample = ImportedBook(
        book = Book(
            id = "sample",
            title = "示例小说",
            author = "本地作者",
            coverPath = null,
            sourcePath = "sample.txt",
            progress = 0.32f,
        ),
        chapters = listOf(
            Chapter(
                id = "sample-0",
                bookId = "sample",
                index = 0,
                title = "第一章",
                content = "这是第一章正文。她说：“今晚风很大。”李明点了点头。",
            )
        ),
    )

    fun listBooks(): List<Book> = listOf(sample.book)
    fun firstChapter(bookId: String): Chapter = sample.chapters.first { it.bookId == bookId }
}
```

- [ ] **Step 2: Update UI to use concrete labels**

In `LibraryScreen.kt`, show `示例小说`, `本地作者`, and `阅读进度 32%`.

In `ReaderScreen.kt`, show the sample first chapter and buttons `目录`, `听书`, `设置`.

In `VoiceStudioScreen.kt`, show sections `旁白音色`, `对白音色`, `角色候选`, and `克隆音色`.

- [ ] **Step 3: Build app**

```bash
cd android_reader
./gradlew assembleDebug
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 4: Commit UI wiring**

```bash
git add android_reader/app/src/main/java/com/afun9/bookreader
git commit -m "feat: wire mvp reader screens"
```

## Task 10: Verification And Push

**Files:**
- Verify all Android project files.

- [ ] **Step 1: Run unit tests**

```bash
cd android_reader
./gradlew testDebugUnitTest
```

Expected: all tests pass.

- [ ] **Step 2: Build debug APK**

```bash
cd android_reader
./gradlew assembleDebug
```

Expected: APK exists at `android_reader/app/build/outputs/apk/debug/app-debug.apk`.

- [ ] **Step 3: Check Git status**

```bash
git status --short
```

Expected: only intentionally untracked local model source directories may remain.

- [ ] **Step 4: Push**

```bash
git push
```

Expected: GitHub `main` is updated.

## Self-Review

Spec coverage:

- Android app: covered by Tasks 1 and 2.
- Local TXT/EPUB import: covered by Tasks 3 and 8.
- Bookshelf: covered by Tasks 2 and 9.
- Reading modes/settings: represented in `ReaderSettings` and reader UI shell in Tasks 4 and 9; visual polish and real pagination need a later plan.
- Fully offline TTS and bundled model: covered by Tasks 1 and 7.
- Segment-level streaming generation: covered by Task 4 and Task 5.
- Background playback: queue model is covered by Task 5; Media3 service implementation needs a later plan.
- Voice Studio, cloning, rules: covered by Task 6 and UI shell in Task 9; real clone audio capture needs a later plan.
- Semi-automatic character detection: model support is introduced in Task 6; extraction UI and parser need a later plan.
- Storage with Room: dependency is included in Task 1; schema implementation needs a later plan.

Placeholder scan:

- The plan contains no unresolved placeholder markers.
- Later-plan items are explicitly named as outside this first implementation plan, not left as ambiguous steps.

Type consistency:

- `Book`, `Chapter`, `ImportedBook`, `TextSegment`, `VoiceProfile`, `VoiceRule`, `SpeechContext`, and `TtsCacheKey` are introduced before dependent tasks use them.
