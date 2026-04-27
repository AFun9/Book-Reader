# Book Reader

Offline Android novel reader with local TXT/EPUB import and a bundled MOSS TTS runtime target.

This repository is intentionally independent from the original MOSS-TTS-Nano source tree. It only keeps the Android app and integration code needed by the reader. Optimized ONNX model files are local build inputs and are not committed.

## Build

Set the Android SDK location if it is not already configured:

```bash
export ANDROID_HOME=/home/model/Android/Sdk
export ANDROID_SDK_ROOT=/home/model/Android/Sdk
```

Optionally point the build at the final optimized model directory:

```bash
export MOSS_TTS_MODEL_DIR=/path/to/moss_tts_nano_fp16_graphopt/model
```

Then build:

```bash
./gradlew :app:compileDebugKotlin
```

The model is copied into `app/src/main/assets/models/moss_tts_nano_fp16_graphopt/model/` during Android pre-build tasks. That assets directory is ignored by Git.
