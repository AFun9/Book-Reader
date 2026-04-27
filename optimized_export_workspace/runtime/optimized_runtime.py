from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import torch
import torchaudio

from optimized_export_workspace.common import read_json, run_dir_for_id


def _flatten3d_int32(nested: list[list[list[int]]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    dim2 = len(nested[0][0])
    data = np.asarray([value for plane in nested for row in plane for value in row], dtype=np.int32)
    return data, [dim0, dim1, dim2]


def _flatten2d_int32(nested: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    data = np.asarray([value for row in nested for value in row], dtype=np.int32)
    return data, [dim0, dim1]


def _write_waveform_to_wav(path: str | Path, waveform: np.ndarray, sample_rate: int) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(int(pcm16.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())
    return output_path


def _merge_audio_channels(channel_arrays: list[np.ndarray]) -> np.ndarray:
    if not channel_arrays:
        return np.zeros((0, 1), dtype=np.float32)
    if len(channel_arrays) == 1:
        return np.asarray(channel_arrays[0], dtype=np.float32).reshape(-1, 1)
    min_length = min(int(channel.shape[0]) for channel in channel_arrays)
    trimmed = [np.asarray(channel[:min_length], dtype=np.float32) for channel in channel_arrays]
    return np.stack(trimmed, axis=1)


class OptimizedTtsRuntime:
    def __init__(self, run_id: str, *, thread_count: int = 4, seed: int = 1234) -> None:
        self.run_dir = run_dir_for_id(run_id).resolve()
        self.tts_dir = self.run_dir / "model" / "tts"
        self.codec_dir = self.run_dir / "model" / "codec"
        self.audio_dir = self.run_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.tts_dir / "optimized_manifest.json"
        self.manifest = read_json(self.manifest_path)
        self.tts_meta = read_json(self.tts_dir / self.manifest["model_files"]["tts_meta"])
        self.codec_meta = read_json((self.tts_dir / self.manifest["model_files"]["codec_meta"]).resolve())
        self.thread_count = max(1, int(thread_count))
        self.rng = np.random.default_rng(int(seed))
        self.sp_model = spm.SentencePieceProcessor(
            model_file=str(self.tts_dir / self.manifest["model_files"]["tokenizer_model"])
        )
        self.sessions = self._create_sessions()

    def _session(self, path: Path) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self.thread_count
        options.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def _create_sessions(self) -> dict[str, ort.InferenceSession]:
        return {
            "prefill": self._session(self.tts_dir / self.manifest["files"]["prefill"]),
            "decode": self._session(self.tts_dir / self.manifest["files"]["decode_step"]),
            "local_fixed_sampled_frame": self._session(self.tts_dir / self.manifest["files"]["local_fixed_sampled_frame"]),
            "codec_encode": self._session(self.codec_dir / self.codec_meta["files"]["encode"]),
            "codec_decode": self._session(self.codec_dir / self.codec_meta["files"]["decode_full"]),
        }

    def encode_text(self, text: str) -> list[int]:
        return [int(token_id) for token_id in self.sp_model.encode(str(text or ""), out_type=int)]

    def list_builtin_voices(self) -> list[dict[str, Any]]:
        return list(self.manifest["builtin_voices"])

    def build_text_rows(self, token_ids: list[int]) -> list[list[int]]:
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        audio_pad = int(self.manifest["tts_config"]["audio_pad_token_id"])
        rows: list[list[int]] = []
        for token_id in token_ids:
            row = [audio_pad] * row_width
            row[0] = int(token_id)
            rows.append(row)
        return rows

    def build_audio_prefix_rows(self, prompt_audio_codes: list[list[int]]) -> list[list[int]]:
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        audio_pad = int(self.manifest["tts_config"]["audio_pad_token_id"])
        slot = int(self.manifest["tts_config"]["audio_user_slot_token_id"])
        rows: list[list[int]] = []
        for code_row in prompt_audio_codes:
            row = [audio_pad] * row_width
            row[0] = slot
            for index, token_id in enumerate(code_row[: row_width - 1]):
                row[index + 1] = int(token_id)
            rows.append(row)
        return rows

    def build_voice_clone_request_rows(
        self,
        prompt_audio_codes: list[list[int]],
        text_token_ids: list[int],
    ) -> dict[str, list[list[int]]]:
        prefix_text_token_ids = [
            *self.manifest["prompt_templates"]["user_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        suffix_text_token_ids = [
            int(self.manifest["tts_config"]["audio_end_token_id"]),
            *self.manifest["prompt_templates"]["user_prompt_after_reference_token_ids"],
            *text_token_ids,
            *self.manifest["prompt_templates"]["assistant_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        rows = [
            *self.build_text_rows(prefix_text_token_ids),
            *self.build_audio_prefix_rows(prompt_audio_codes),
            *self.build_text_rows(suffix_text_token_ids),
        ]
        return {"inputIds": rows, "attentionMask": [[1 for _ in rows]]}

    def resolve_prompt_audio_codes(
        self,
        *,
        voice: str | None = None,
        prompt_audio_path: str | Path | None = None,
    ) -> list[list[int]]:
        if prompt_audio_path:
            return self.encode_reference_audio(prompt_audio_path)
        voice_name = str(voice or self.list_builtin_voices()[0]["voice"])
        voice_row = next((item for item in self.list_builtin_voices() if item["voice"] == voice_name), None)
        if voice_row is None:
            raise ValueError(f"Built-in voice not found: {voice_name}")
        return [list(map(int, row)) for row in voice_row["prompt_audio_codes"]]

    def encode_reference_audio(self, reference_audio_path: str | Path) -> list[list[int]]:
        waveform, sample_rate = torchaudio.load(str(Path(reference_audio_path).expanduser().resolve()))
        waveform = waveform.to(torch.float32)
        target_sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        target_channels = int(self.codec_meta["codec_config"]["channels"])
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        if int(waveform.shape[0]) == 1 and target_channels > 1:
            waveform = waveform.repeat(target_channels, 1)
        elif int(waveform.shape[0]) > 1 and target_channels == 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform_np = waveform.unsqueeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        outputs = self.sessions["codec_encode"].run(
            None,
            {
                "waveform": waveform_np,
                "input_lengths": np.asarray([waveform_np.shape[-1]], dtype=np.int32),
            },
        )
        named_outputs = dict(zip([output.name for output in self.sessions["codec_encode"].get_outputs()], outputs, strict=True))
        audio_codes = np.asarray(named_outputs["audio_codes"], dtype=np.int32)
        code_length = int(np.asarray(named_outputs["audio_code_lengths"]).reshape(-1)[0])
        n_vq = int(self.codec_meta["codec_config"]["num_quantizers"])
        return [
            [int(audio_codes[0, frame_index, quantizer_index]) for quantizer_index in range(n_vq)]
            for frame_index in range(code_length)
        ]

    def _run_prefill(self, request_rows: dict[str, list[list[int]]]) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
        prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = _flatten2d_int32(request_rows["attentionMask"])
        outputs = self.sessions["prefill"].run(
            None,
            {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        named_outputs = dict(zip([output.name for output in self.sessions["prefill"].get_outputs()], outputs, strict=True))
        past_by_name = {
            output_name.replace("present_", "past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["prefill_output_names"][1:]
        }
        past_valid_length = sum(int(item) for item in request_rows["attentionMask"][0])
        return named_outputs["global_hidden_last"].astype(np.float32, copy=False), past_by_name, past_valid_length

    def _run_local_fixed_sampled_frame(
        self,
        global_hidden_last: np.ndarray,
        previous_token_sets_by_channel: list[set[int]],
    ) -> tuple[bool, list[int]]:
        codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        repetition_seen_mask = np.zeros((1, n_vq, codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= int(token_id) < codebook_size:
                    repetition_seen_mask[0, channel_index, int(token_id)] = 1
        outputs = self.sessions["local_fixed_sampled_frame"].run(
            None,
            {
                "global_hidden": global_hidden_last.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "assistant_random_u": np.asarray([min(0.99999994, max(0.0, float(self.rng.random())))], dtype=np.float32),
                "audio_random_u": np.asarray(
                    [[min(0.99999994, max(0.0, float(self.rng.random()))) for _ in range(n_vq)]],
                    dtype=np.float32,
                ),
            },
        )
        named_outputs = dict(zip([output.name for output in self.sessions["local_fixed_sampled_frame"].get_outputs()], outputs, strict=True))
        should_continue = bool(int(np.asarray(named_outputs["should_continue"]).reshape(-1)[0]))
        frame = np.asarray(named_outputs["frame_token_ids"]).reshape(-1).astype(np.int32, copy=False).tolist()
        return should_continue, [int(item) for item in frame]

    def _run_decode_step(
        self,
        frame: list[int],
        past_by_name: dict[str, np.ndarray],
        past_valid_length: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        next_row = np.full(
            (1, 1, row_width),
            int(self.manifest["tts_config"]["audio_pad_token_id"]),
            dtype=np.int32,
        )
        next_row[0, 0, 0] = int(self.manifest["tts_config"]["audio_assistant_slot_token_id"])
        for index, token_id in enumerate(frame):
            next_row[0, 0, index + 1] = int(token_id)
        feeds: dict[str, np.ndarray] = {
            "input_ids": next_row,
            "past_valid_lengths": np.asarray([past_valid_length], dtype=np.int32),
        }
        for input_name in self.tts_meta["onnx"]["decode_input_names"][2:]:
            feeds[input_name] = past_by_name[input_name]
        outputs = self.sessions["decode"].run(None, feeds)
        named_outputs = dict(zip([output.name for output in self.sessions["decode"].get_outputs()], outputs, strict=True))
        next_past = {
            output_name.replace("present_", "past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["decode_output_names"][1:]
        }
        return named_outputs["global_hidden_last"].astype(np.float32, copy=False), next_past

    def generate_audio_frames(
        self,
        request_rows: dict[str, list[list[int]]],
        *,
        max_new_frames: int,
    ) -> list[list[int]]:
        global_hidden_last, past_by_name, past_valid_length = self._run_prefill(request_rows)
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        previous_token_sets_by_channel = [set() for _ in range(n_vq)]
        generated_frames: list[list[int]] = []
        for _step_index in range(int(max_new_frames)):
            should_continue, frame = self._run_local_fixed_sampled_frame(
                global_hidden_last,
                previous_token_sets_by_channel,
            )
            if not should_continue:
                break
            generated_frames.append(frame)
            for channel_index, sampled_token in enumerate(frame):
                previous_token_sets_by_channel[channel_index].add(int(sampled_token))
            global_hidden_last, past_by_name = self._run_decode_step(frame, past_by_name, past_valid_length)
            past_valid_length += 1
        return generated_frames

    def decode_full_audio(self, generated_frames: list[list[int]]) -> tuple[np.ndarray, int]:
        if not generated_frames:
            return np.zeros((0, 1), dtype=np.float32), int(self.codec_meta["codec_config"]["sample_rate"])
        audio_codes, dims = _flatten3d_int32([generated_frames])
        outputs = self.sessions["codec_decode"].run(
            None,
            {
                "audio_codes": audio_codes.reshape(dims),
                "audio_code_lengths": np.asarray([len(generated_frames)], dtype=np.int32),
            },
        )
        named_outputs = dict(zip([output.name for output in self.sessions["codec_decode"].get_outputs()], outputs, strict=True))
        audio = np.asarray(named_outputs["audio"], dtype=np.float32)
        audio_length = int(np.asarray(named_outputs["audio_lengths"]).reshape(-1)[0])
        channels = [audio[0, channel_index, :audio_length] for channel_index in range(audio.shape[1])]
        return _merge_audio_channels(channels), int(self.codec_meta["codec_config"]["sample_rate"])

    def synthesize(
        self,
        *,
        text: str,
        voice: str | None = None,
        prompt_audio_path: str | Path | None = None,
        output_name: str = "optimized_output.wav",
        max_new_frames: int | None = None,
    ) -> dict[str, Any]:
        text_token_ids = self.encode_text(text)
        prompt_audio_codes = self.resolve_prompt_audio_codes(voice=voice, prompt_audio_path=prompt_audio_path)
        request_rows = self.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
        frame_budget = int(max_new_frames or self.manifest["generation_defaults"]["max_new_frames"])
        generated_frames = self.generate_audio_frames(request_rows, max_new_frames=frame_budget)
        waveform, sample_rate = self.decode_full_audio(generated_frames)
        audio_path = _write_waveform_to_wav(self.audio_dir / output_name, waveform, sample_rate)
        return {
            "audio_path": str(audio_path),
            "sample_rate": sample_rate,
            "generated_frames": generated_frames,
            "text_token_ids": text_token_ids,
        }

