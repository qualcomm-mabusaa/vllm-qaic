# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""QAIC contracts for upstream speech encoder-decoder models."""

from collections.abc import Sequence

import numpy as np


QAIC_SPEECH_MODEL_TYPES = frozenset({"whisper", "cohere_asr"})


def is_qaic_speech_model(model_type: str) -> bool:
    return model_type in QAIC_SPEECH_MODEL_TYPES


def requires_real_audio_prefill(model_type: str) -> bool:
    """Whether a QPC must first receive request audio before decode can run."""
    return model_type == "cohere_asr"


def strip_cohere_asr_renderer_bos(
    input_ids: np.ndarray, position_ids: np.ndarray, bos_token_id: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Remove vLLM's synthetic BOS before Cohere's native decoder prefix.

    Cohere ASR supplies its own decoder control prefix. If the generic
    encoder-decoder renderer prepends BOS, it is not part of the QEff QPC
    contract and the remaining positions must start at zero.
    """
    if bos_token_id is None or input_ids.size <= 1 or input_ids[0] != bos_token_id:
        return input_ids, position_ids
    position_ids = position_ids[..., 1:]
    return input_ids[1:], position_ids - position_ids[..., :1]


def make_cohere_asr_decode_inputs(
    input_features_shape: Sequence[int],
    input_features_dtype: np.dtype,
    feature_lengths_shape: Sequence[int],
    feature_lengths_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Create the descriptor-defined decode inputs for a Cohere ASR QPC."""
    decode_shape = list(input_features_shape)
    decode_shape[-1] = 1
    return {
        "input_features": np.zeros(decode_shape, dtype=input_features_dtype),
        "feature_lengths": np.ones(feature_lengths_shape, dtype=feature_lengths_dtype),
    }


def prepare_cohere_asr_qpc_inputs(
    input_features: np.ndarray,
    feature_lengths: np.ndarray,
    target_features_shape: Sequence[int],
    target_features_dtype: np.dtype,
    target_lengths_shape: Sequence[int],
    target_lengths_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Pad Cohere processor output to the fixed QPC encoder specialization.

    The upstream Cohere processor emits ``input_features`` and ``length``. QEff
    consumes the same per-item length as ``feature_lengths`` and requires the
    feature tensor to match the QPC's static encoder binding.
    """
    target_features_shape = list(target_features_shape)
    target_lengths_shape = list(target_lengths_shape)
    if input_features.ndim == len(target_features_shape) - 1:
        input_features = np.expand_dims(input_features, axis=0)
    if feature_lengths.ndim == 0:
        feature_lengths = feature_lengths.reshape(1)
    if list(input_features.shape[:-1]) != target_features_shape[:-1]:
        raise ValueError(
            "Cohere ASR input feature shape does not match the QPC binding: "
            f"got {list(input_features.shape)}, expected leading dimensions "
            f"{target_features_shape[:-1]}."
        )
    if input_features.shape[-1] > target_features_shape[-1]:
        raise ValueError(
            "Cohere ASR input feature length exceeds the QPC encoder binding: "
            f"got {input_features.shape[-1]}, maximum is {target_features_shape[-1]}."
        )
    if list(feature_lengths.shape) != target_lengths_shape:
        raise ValueError(
            "Cohere ASR feature_lengths shape does not match the QPC binding: "
            f"got {list(feature_lengths.shape)}, expected {target_lengths_shape}."
        )
    if np.any(feature_lengths < 1) or np.any(
        feature_lengths > input_features.shape[-1]
    ):
        raise ValueError(
            "Cohere ASR feature_lengths must describe the supplied input features."
        )

    padded_features = np.zeros(target_features_shape, dtype=target_features_dtype)
    padded_features[..., : input_features.shape[-1]] = input_features
    return {
        "input_features": padded_features,
        "feature_lengths": feature_lengths.astype(target_lengths_dtype, copy=False),
    }
