"""Monkey-patch Llama-3 decoder layers to use the fused RMSNorm+QKV kernel.

Status: reference integration, not a production adapter. This module rewrites
`transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward` at import
time to exercise the fused kernel end-to-end and back the numbers in
`docs/week4_e2e.md`. It is pinned to the Transformers version listed in the
repo's Dockerfile (5.5.4); other versions may change the decoder layer's
internal API and need a matching patch. If you want to use the kernel in your
own serving stack, import `fused_rmsnorm_residual_matmul` from
`triton_fused_rmsnorm_qkv` directly and call it from your model code --
don't vendor this monkey-patch.

Fusion target: the `input_layernorm + q_proj/k_proj/v_proj` subgraph at the
top of each LlamaDecoderLayer. Our kernel fuses `(optional residual) +
RMSNorm + arbitrary-width matmul`; Llama's architecture does the residual
add AFTER attention, so we pass `residual=None` here and only exercise the
(norm + matmul) fusion.

Native GQA layout (Phase 6)
---------------------------
Llama-3-8B: H=4096, H_kv=1024 (num_key_value_heads=8 × head_dim=128). We
pack Q, K, V concatenated WITHOUT zero-padding:

    packed[0 : H          , :] = q_proj.weight   # (H,    H)
    packed[H : H+H_kv     , :] = k_proj.weight   # (H_kv, H)
    packed[H+H_kv : H+2H_kv, :] = v_proj.weight  # (H_kv, H)

After the fused call:
    Q = out[..., :H]
    K = out[..., H : H+H_kv]
    V = out[..., H+H_kv : H+2*H_kv]

This cuts the packed weight from (3H, H) = (12288, 4096) down to
(H+2*H_kv, H) = (6144, 4096) — halving QKV weight HBM traffic for Llama-3
(the Phase-5 zero-padded path doubled it). At decode (HBM-bound) this is
the dominant cost.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaAttention,
    LlamaDecoderLayer,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from triton_fused_rmsnorm_qkv import (
    fused_rmsnorm_residual_matmul,
    fused_rmsnorm_residual_matmul_fixed,
)


def _fused_entrypoint():
    """Pick fused kernel entrypoint based on TRITON_FUSED_DISABLE_AUTOTUNE env var."""
    if os.environ.get("TRITON_FUSED_DISABLE_AUTOTUNE") == "1":
        return fused_rmsnorm_residual_matmul_fixed
    return fused_rmsnorm_residual_matmul


LLAMA3_8B_CONFIG = dict(
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    max_position_embeddings=8192,
    rms_norm_eps=1e-5,
    vocab_size=128256,
    rope_theta=500000.0,
    tie_word_embeddings=False,
    attention_bias=False,
    mlp_bias=False,
)


def build_llama3_8b(
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    num_hidden_layers: Optional[int] = None,
) -> LlamaForCausalLM:
    """Instantiate a Llama-3-8B-shaped model with random weights."""
    cfg_dict = dict(LLAMA3_8B_CONFIG)
    if num_hidden_layers is not None:
        cfg_dict["num_hidden_layers"] = num_hidden_layers
    cfg = LlamaConfig(**cfg_dict)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = LlamaForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    return model.eval()


# ---------------------------------------------------------------------------
# Fused decoder layer forward (native GQA)
# ---------------------------------------------------------------------------


def _build_packed_qkv_weight(self_attn: LlamaAttention) -> torch.Tensor:
    """Pack q/k/v into a single (H + 2*H_kv, H) matrix — no zero padding."""
    q_w = self_attn.q_proj.weight
    k_w = self_attn.k_proj.weight
    v_w = self_attn.v_proj.weight
    H = q_w.shape[1]
    H_q = q_w.shape[0]
    H_kv = k_w.shape[0]
    assert H_q == H, f"q_proj output must equal input dim; got {H_q} vs {H}"
    assert v_w.shape[0] == H_kv, f"v_proj/k_proj must share output dim; got {v_w.shape[0]} vs {H_kv}"

    N_OUT = H_q + 2 * H_kv
    packed = torch.empty(N_OUT, H, dtype=q_w.dtype, device=q_w.device)
    packed[:H_q].copy_(q_w)
    packed[H_q : H_q + H_kv].copy_(k_w)
    packed[H_q + H_kv : H_q + 2 * H_kv].copy_(v_w)
    return packed


def _fused_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    use_cache: Optional[bool] = False,
    position_embeddings=None,
    **kwargs,
) -> torch.Tensor:
    """Replacement for LlamaDecoderLayer.forward using native-GQA fused kernel."""
    residual = hidden_states

    attn = self.self_attn
    H = hidden_states.shape[-1]
    H_kv = self._fused_H_kv

    # Fused RMSNorm + packed GQA projection. residual=None skips the add.
    qkv = _fused_entrypoint()(
        hidden_states,
        None,
        self.input_layernorm.weight,
        self._fused_packed_qkv,
        eps=self.input_layernorm.variance_epsilon,
    )
    # (B, S, H + 2*H_kv) — split by real widths, no padding to discard.
    query_states = qkv[..., :H]
    key_states = qkv[..., H : H + H_kv]
    value_states = qkv[..., H + H_kv : H + 2 * H_kv]

    input_shape = hidden_states.shape[:-1]
    hidden_shape_q = (*input_shape, attn.config.num_attention_heads, attn.head_dim)
    hidden_shape_kv = (*input_shape, attn.config.num_key_value_heads, attn.head_dim)
    query_states = query_states.contiguous().view(hidden_shape_q).transpose(1, 2)
    key_states = key_states.contiguous().view(hidden_shape_kv).transpose(1, 2)
    value_states = value_states.contiguous().view(hidden_shape_kv).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, attn.layer_idx
        )

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, eager_attention_forward
    )
    attn_output, _ = attention_interface(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=attn.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn.o_proj(attn_output)
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


# ---------------------------------------------------------------------------
# Patch / unpatch
# ---------------------------------------------------------------------------

_PATCH_SENTINEL = "_fused_original_forward"


def apply_fused_patch(model: LlamaForCausalLM) -> LlamaForCausalLM:
    """Monkey-patch every LlamaDecoderLayer in `model` to use the fused kernel."""
    for layer in model.model.layers:
        if hasattr(layer, _PATCH_SENTINEL):
            continue
        layer._fused_packed_qkv = _build_packed_qkv_weight(layer.self_attn)
        layer._fused_H_kv = layer.self_attn.k_proj.weight.shape[0]
        setattr(layer, _PATCH_SENTINEL, layer.forward)
        layer.forward = _fused_decoder_layer_forward.__get__(layer, type(layer))
    return model


def remove_fused_patch(model: LlamaForCausalLM) -> LlamaForCausalLM:
    """Restore original LlamaDecoderLayer.forward on every layer."""
    for layer in model.model.layers:
        if not hasattr(layer, _PATCH_SENTINEL):
            continue
        layer.forward = getattr(layer, _PATCH_SENTINEL)
        delattr(layer, _PATCH_SENTINEL)
        if hasattr(layer, "_fused_packed_qkv"):
            del layer._fused_packed_qkv
        if hasattr(layer, "_fused_H_kv"):
            del layer._fused_H_kv
    return model


@contextmanager
def fused_patch(model: LlamaForCausalLM):
    """Context manager that applies the fused patch for the scoped block."""
    apply_fused_patch(model)
    try:
        yield model
    finally:
        remove_fused_patch(model)
