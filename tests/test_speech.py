# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import numpy as np
import pytest
import torch
from transformers import BatchFeature

from vllm_qaic.speech import (
    is_qaic_speech_model,
    make_cohere_asr_decode_inputs,
    prepare_cohere_asr_qpc_inputs,
    requires_encoder_prefill_before_decode,
    strip_cohere_asr_renderer_bos,
)


def test_qaic_speech_model_types():
    assert is_qaic_speech_model("whisper")
    assert is_qaic_speech_model("cohere_asr")
    assert not is_qaic_speech_model("qwen3_vl")
    assert requires_encoder_prefill_before_decode("cohere_asr")
    assert not requires_encoder_prefill_before_decode("whisper")


def test_cohere_asr_strips_only_renderer_bos_and_renumbers_positions():
    input_ids, positions = strip_cohere_asr_renderer_bos(
        np.array([4, 13764, 7], dtype=np.int64),
        np.array([[0, 1, 2]], dtype=np.int64),
        4,
    )

    np.testing.assert_array_equal(input_ids, [13764, 7])
    np.testing.assert_array_equal(positions, [[0, 1]])

    unchanged_ids, unchanged_positions = strip_cohere_asr_renderer_bos(
        input_ids, positions, 4
    )
    np.testing.assert_array_equal(unchanged_ids, input_ids)
    np.testing.assert_array_equal(unchanged_positions, positions)


def test_cohere_asr_qpc_inputs_pad_to_descriptor_shape():
    input_features = np.arange(1 * 128 * 4, dtype=np.float32).reshape(1, 128, 4)
    prepared = prepare_cohere_asr_qpc_inputs(
        input_features,
        np.array([3], dtype=np.int64),
        [1, 128, 8],
        np.dtype(np.float16),
        [1],
        np.dtype(np.int64),
    )

    assert prepared["input_features"].shape == (1, 128, 8)
    assert prepared["input_features"].dtype == np.float16
    np.testing.assert_array_equal(prepared["input_features"][..., :4], input_features)
    np.testing.assert_array_equal(prepared["input_features"][..., 4:], 0)
    np.testing.assert_array_equal(prepared["feature_lengths"], [3])


def test_cohere_asr_qpc_inputs_accept_vllm_single_request_shapes():
    prepared = prepare_cohere_asr_qpc_inputs(
        np.ones((128, 4), dtype=np.float32),
        np.array(3, dtype=np.int64),
        [1, 128, 8],
        np.dtype(np.float16),
        [1],
        np.dtype(np.int64),
    )

    assert prepared["input_features"].shape == (1, 128, 8)
    np.testing.assert_array_equal(prepared["feature_lengths"], [3])


def test_cohere_asr_qpc_inputs_reject_invalid_binding_shapes():
    input_features = np.zeros((1, 128, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds"):
        prepare_cohere_asr_qpc_inputs(
            input_features,
            np.array([9], dtype=np.int64),
            [1, 128, 8],
            np.dtype(np.float32),
            [1],
            np.dtype(np.int64),
        )


def test_cohere_asr_qpc_inputs_reject_invalid_feature_length():
    with pytest.raises(ValueError, match="must describe"):
        prepare_cohere_asr_qpc_inputs(
            np.zeros((1, 128, 4), dtype=np.float32),
            np.array([5], dtype=np.int64),
            [1, 128, 8],
            np.dtype(np.float32),
            [1],
            np.dtype(np.int64),
        )


def test_cohere_asr_decode_inputs_follow_descriptor_shape():
    inputs = make_cohere_asr_decode_inputs(
        [1, 128, 3504], np.dtype(np.float32), [1], np.dtype(np.int64)
    )

    assert inputs["input_features"].shape == (1, 128, 1)
    assert inputs["input_features"].dtype == np.float32
    np.testing.assert_array_equal(inputs["input_features"], 0)
    np.testing.assert_array_equal(inputs["feature_lengths"], [1])


def test_cohere_asr_processor_extracts_audio_once(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.models.cohere_asr import CohereASRMultiModalProcessor
    from vllm_qaic.model_loader.qaic_custom_mm_processor import (
        QaicCohereASRMultiModalProcessor,
    )

    calls = {"vllm": 0, "native": 0}

    def tokenize_only(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        calls["vllm"] += 1
        assert mm_data == {}
        return BatchFeature({"input_ids": torch.tensor([[7]])})

    class NativeFeatureExtractor:
        sampling_rate = 16_000
        max_audio_clip_s = 35.0
        overlap_chunk_second = 5.0

        def __call__(self, audios, **kwargs):
            calls["native"] += 1
            assert len(audios) == 1
            return BatchFeature(
                {
                    "input_features": torch.ones((1, 4, 128)),
                    "attention_mask": torch.ones((1, 4), dtype=torch.int64),
                }
            )

    monkeypatch.setattr(
        CohereASRMultiModalProcessor, "_call_hf_processor", tokenize_only
    )
    processor = object.__new__(QaicCohereASRMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(max_audio_clip_s=35.0)
    )
    feature_extractor = NativeFeatureExtractor()
    processor.__dict__["_qaic_hf_processor"] = SimpleNamespace(
        feature_extractor=feature_extractor
    )

    outputs = processor._call_hf_processor(
        "prompt",
        {"audios": [np.zeros(16_000, dtype=np.float32)]},
        {},
        {},
    )

    assert calls == {"vllm": 1, "native": 1}
    assert outputs["input_features"].shape == (1, 128, 4)
    np.testing.assert_array_equal(outputs["length"], [4])
    assert feature_extractor.overlap_chunk_second == 5.0
