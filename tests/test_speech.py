# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import numpy as np
import pytest

from vllm_qaic.speech import (
    is_qaic_speech_model,
    make_cohere_asr_decode_inputs,
    prepare_cohere_asr_qpc_inputs,
    requires_real_audio_prefill,
    strip_cohere_asr_renderer_bos,
)


def test_qaic_speech_model_types():
    assert is_qaic_speech_model("whisper")
    assert is_qaic_speech_model("cohere_asr")
    assert not is_qaic_speech_model("qwen3_vl")
    assert requires_real_audio_prefill("cohere_asr")
    assert not requires_real_audio_prefill("whisper")


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
