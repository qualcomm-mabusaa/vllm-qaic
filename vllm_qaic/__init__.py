# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

try:
    from vllm_qaic._version import __version__
except ImportError:
    __version__ = "unknown"


def _apply_global_platform_patches():
    _patch_torch_fp4_dtype()


# TODO: Remove once QEFF or torch-qaic is moved to Pytorch >= 2.11.x
def _patch_torch_fp4_dtype():
    """Add torch.float4_e2m1fn_x2 sentinel on CPU-only PyTorch builds.

    ``vllm.ir.tolerances`` references ``torch.float4_e2m1fn_x2`` at module level. That dtype
    does not exist in the CPU-only PyTorch build used by QAIC. ``torch.float32`` is used as a
    placeholder because the tolerances dict is only consumed by IR comparison tests, never
    during QAIC inference.
    """
    import torch

    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = torch.float32


def register():
    _apply_global_platform_patches()
    return "vllm_qaic.platform.QaicPlatform"


def register_general_plugins():
    # General plugins run in every API and engine-worker process after the
    # platform is initialized. Model-specific registry overrides therefore
    # survive the worker-process boundary.
    from vllm_qaic.model_loader.qaic_custom_mm_processor import (
        register_qaic_custom_mm_processor,
    )

    register_qaic_custom_mm_processor("cohere_asr")

    from vllm_qaic.distributed.kv_transfer.kv_connector import register_connector

    register_connector()


def register_connector():
    """Compatibility entry point for existing package installations."""
    register_general_plugins()
