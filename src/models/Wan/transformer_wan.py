# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from peft import LoraConfig, get_peft_model, TaskType
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if attn.cross_attention_dim_head is None:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanAttnProcessor:
    _attention_backend = None

    def __init__(self, return_attention_maps):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0."
            )
        self.return_attention_maps = return_attention_maps

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if not self.return_attention_maps:
            # Use fast dispatch
            # Cast attention_mask to match query dtype to avoid dtype mismatch
            attn_mask = attention_mask.to(query.dtype) if attention_mask is not None else None

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states = hidden_states.flatten(2, 3)
            attn_weights = None
            
        else:
            # Manual attention computation to get attention maps
            # query, key, value: (B, S, H, D) where H=heads, D=head_dim

            # Transpose to (B, H, S, D) for batched matrix multiplication
            q = query.transpose(1, 2)  # (B, H, S, D)
            k = key.transpose(1, 2)    # (B, H, S, D)
            v = value.transpose(1, 2)  # (B, H, S, D)

            # Compute attention scores: (B, H, S, S)
            scale = q.size(-1) ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Apply attention mask if provided
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            # Compute attention weights
            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, S, S)

            # Apply attention to values
            hidden_states = torch.matmul(attn_weights, v)  # (B, H, S, D)

            # Transpose back and flatten: (B, S, H, D) -> (B, S, H*D)
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)


        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states, attn_weights

class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None, #image embedding dimension
        cross_attention_dim_head: Optional[int] = None, #text embedding dimension
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if self.cross_attention_dim_head is None:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(
                self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class WanTransformerBlockOG(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        return_attention_maps: bool,
        qk_norm: str = "rms_norm_across_heads",
        eps: float = 1e-6,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(return_attention_maps=return_attention_maps),
        )

        # 2. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 3. Curriculum learning parameter for spatial attention
        self.attention_window = -1  # -1 = full attention (default)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_weights = None

        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states.float()).type_as(hidden_states)
        attn_output, attn_weights = self.attn1(norm_hidden_states, None, attention_mask, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output).type_as(hidden_states)

        # 2. Feed-forward
        norm_hidden_states = self.norm3(hidden_states.float()).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float()).type_as(hidden_states)

        return hidden_states, attn_weights


class WanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, False, qk_norm, eps
                )
                for i in range(num_layers)
            ]
        )

        self.gradient_checkpointing = gradient_checkpointing
    
class WanDecoderTransformer(torch.nn.Module):
    def __init__(
        self,
        chunk:int = 2,
        rope_max_seq_len=None,
        patch_size=[(1, 2, 2), (1, 4, 4), (1, 8, 8)],
        num_layers: int = 30,
        num_heads=12,
        head_dim=128,
        channels=[384, 192, 192],
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        reusing: bool = False,
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.chunk = chunk
        self.use_lora = use_lora
        self.attn_weights = []

        # # Initialize the transformer
        if pretrained:
            self.transformer = WanTransformer3DModel.from_pretrained(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                subfolder="transformer",
                num_attention_heads=12,
                attention_head_dim=128,
                num_layers=30,
                ffn_dim=8960,
                eps=1e-6,
                qk_norm="rms_norm_across_heads",
                gradient_checkpointing=gradient_checkpointing,
                torch_dtype=torch.float32,
                device_map=None,
                ignore_mismatched_sizes=True,
                strict=False
            )
        else:
            self.transformer = WanTransformer3DModel(
                num_attention_heads=num_heads,
                attention_head_dim=head_dim,
                num_layers=num_layers,
                ffn_dim=8960,
                eps=1e-6,
                qk_norm="rms_norm_across_heads",
                gradient_checkpointing=gradient_checkpointing,
            )

        # Apply LoRA if requested
        if self.use_lora:
            self._apply_lora(lora_rank, lora_alpha, lora_dropout)

        # Configuration
        self.channels = channels
        self.num_attention_heads = num_heads
        self.attention_head_dim = head_dim
        self.num_layers = num_layers
        self.reusing = reusing
        inner_dim = self.num_attention_heads * self.attention_head_dim

        # Ensure each image has 1560 tokens
        seq_len_per_chunk = 1560
        chunk = self.chunk
        self.patch_size = patch_size
        if rope_max_seq_len is None:
            self.rope_max_seq_len = [seq_len_per_chunk * (chunk + 1), seq_len_per_chunk * (2 * chunk), seq_len_per_chunk * (4 * chunk - 2)]
        else:
            self.rope_max_seq_len = rope_max_seq_len
        eps = 1e-6

        # 1. Patch & position embedding
        self.patch_embeddings = nn.ModuleList([
            nn.Conv3d(channels[0], inner_dim, kernel_size=self.patch_size[0], stride=self.patch_size[0]),  # First upblock output
            nn.Conv3d(channels[1], inner_dim, kernel_size=self.patch_size[1], stride=self.patch_size[1]),  # Second upblock output
            nn.Conv3d(channels[2], inner_dim, kernel_size=self.patch_size[2], stride=self.patch_size[2]),  # Third upblock output
        ])
        
        self.rope = nn.ModuleList([
            WanRotaryPosEmbed(self.attention_head_dim, self.patch_size[i], self.rope_max_seq_len[i]) for i in range(3)
        ])
        
        # Output norms & projections for three resolutions
        self.norm_outs = nn.ModuleList([
            FP32LayerNorm(inner_dim, eps, elementwise_affine=False),
            FP32LayerNorm(inner_dim, eps, elementwise_affine=False),
            FP32LayerNorm(inner_dim, eps, elementwise_affine=False),
        ])
        
        self.proj_outs = nn.ModuleList([
            nn.Linear(inner_dim, channels[0] * math.prod(self.patch_size[0])),
            nn.Linear(inner_dim, channels[1] * math.prod(self.patch_size[1])),
            nn.Linear(inner_dim, channels[2] * math.prod(self.patch_size[2])),
        ])
        
        self.initialize_decoder_components()

    def initialize_decoder_components(self):
        """Initialize patch embeddings and position embeddings"""
        import math
        
        # Initialize patch embeddings
        for patch_embed in self.patch_embeddings:
            patch_embed.reset_parameters()
        
        # # Initialize position embeddings (ViT standard)
        # for pos_embed in self.pos_embeds:
        #     nn.init.trunc_normal_(pos_embed, std=0.02)
        
        # Initialize output projections
        for proj_out in self.proj_outs:
            nn.init.xavier_uniform_(proj_out.weight)
            # nn.init.zeros_(proj_out.weight)
            if proj_out.bias is not None:
                nn.init.zeros_(proj_out.bias)

    def _apply_lora(self, lora_rank, lora_alpha, lora_dropout):
        """Apply LoRA to transformer blocks"""
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",
                "ffn.net.0.proj", "ffn.net.2",
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        self.transformer = get_peft_model(self.transformer, lora_config)
    
    def get_lora_target_modules(self):
        """Return the target modules configured for LoRA on the wrapped transformer."""
        if not self.use_lora:
            return []

        transformer = getattr(self, "transformer", None)
        if transformer is None:
            return []

        peft_config = getattr(transformer, "peft_config", None)
        if not peft_config:
            return []

        active_adapter = getattr(transformer, "active_adapter", None)
        if active_adapter and active_adapter in peft_config:
            config = peft_config[active_adapter]
        else:
            config = next(iter(peft_config.values()))

        target_modules = getattr(config, "target_modules", None)
        if target_modules is None:
            return []

        return list(target_modules)

    def fuse_lora_weights(self):
        """
        Fuse LoRA weights into the base model weights.

        This merges the low-rank adaptation matrices (A and B) with the original weights:
        W' = W + (scaling * B @ A)

        After fusing, the model will have the same behavior but without the LoRA overhead,
        making it more efficient for inference.

        Returns:
            bool: True if fusion was successful, False otherwise
        """
        if not self.use_lora:
            print("⚠ LoRA is not enabled, nothing to fuse")
            return False

        try:
            # PEFT library provides a merge_and_unload method
            print("Fusing LoRA weights into base model...")

            # Get the base model with fused weights
            self.transformer = self.transformer.merge_and_unload()

            # Update the use_lora flag since LoRA is now fused
            self.use_lora = False

            print("✓ Successfully fused LoRA weights into base model")
            return True

        except Exception as e:
            print(f"✗ Error fusing LoRA weights: {e}")
            return False

    def unfuse_lora_weights(self):
        """
        Unfuse/unmerge LoRA weights from the base model.

        This separates the LoRA weights from base weights if they were previously merged.
        Note: This only works if the model still has LoRA adapters loaded.

        Returns:
            bool: True if unfusion was successful, False otherwise
        """
        if not self.use_lora:
            print("⚠ LoRA is not enabled or already unfused")
            return False

        try:
            print("Unfusing LoRA weights from base model...")

            # PEFT library provides an unmerge method
            self.transformer.unmerge_adapter()

            print("✓ Successfully unfused LoRA weights from base model")
            return True

        except Exception as e:
            print(f"✗ Error unfusing LoRA weights: {e}")
            return False
    
    def get_map(self):
        return self.attn_weights
    
    def clear_map(self):
        self.attn_weights = []
        
    def create_spatial_mask(self, attention_window, num_frames, height, width, device):
        """
        Create spatial attention mask for self-attention within frames.

        Restricts each token to attend only to spatially nearby tokens within the same frame.
        Uses Manhattan distance for spatial proximity.

        Args:
            batch_size: Batch size
            num_frames: Number of temporal frames
            height: Spatial height of feature map
            width: Spatial width of feature map
            device: torch device

        Returns:
            Attention mask [1, 1, seq_len, seq_len] or None if full attention
        """
        if attention_window < 0:
            return None  # Full attention

        seq_len = num_frames * height * width

        # Tokens are ordered as [t0_h0_w0, t0_h0_w1, ..., t0_hH_wW, t1_h0_w0, ...]

        # For each query token, compute which key token it should attend to
        # Query token i at (t_q, h_q, w_q) should attend to key token at (t=0, h_q, w_q)

        # Create indices for spatial positions (h, w) - reused across frames
        spatial_size = height * width
        h_indices = torch.arange(height, device=device).repeat_interleave(width)  # [0,0,...,0,1,1,...,1,...]
        w_indices = torch.arange(width, device=device).repeat(height)  # [0,1,2,...,W-1,0,1,2,...,W-1,...]

        # For each query position, find the corresponding key index in first frame
        # Query at frame t, position (h,w) -> Key at frame 0, position (h,w)
        # Key index = h * width + w
        key_indices_per_spatial_pos = h_indices * width + w_indices  # [spatial_size]

        # Repeat this pattern for all frames (each query frame uses same spatial mapping)
        key_indices = key_indices_per_spatial_pos.repeat(num_frames)  # [seq_len]

        # Create sparse mask more efficiently using indexing
        # Initialize with -inf (block all attention)
        attention_mask = torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float32, device=device)

        # For each query position, allow attention to exactly one key position
        query_indices = torch.arange(seq_len, device=device)
        attention_mask[query_indices, key_indices] = 0.0

        # Add batch and head dimensions: [1, 1, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        return attention_mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        stage_idx: int = 0,
        return_dict: bool = True,
        window_size=-1,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: Input tensor (B, C, T, H, W) where C is 384 or 192
            stage_idx: 0 for first stage (384 channels), 1 for second stage (192 channels)
            return_dict: Whether to return dict or tuple
            attention_kwargs: Additional attention arguments
        """
        
        assert stage_idx in [0, 1, 2], f"stage_idx must be 0 or 1, got {stage_idx}"

        # clear previous attention weights
        # self.attn_weights = []

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # Get input dimensions
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size[stage_idx]

        # Keep exact output shape even when T/H/W are not divisible by patch size.
        # We pad before patch embedding and crop back after unpatchify.
        pad_t = (p_t - (num_frames % p_t)) % p_t
        pad_h = (p_h - (height % p_h)) % p_h
        pad_w = (p_w - (width % p_w)) % p_w
        if pad_t or pad_h or pad_w:
            hidden_states = F.pad(hidden_states, (0, pad_w, 0, pad_h, 0, pad_t))

        _, _, padded_num_frames, padded_height, padded_width = hidden_states.shape
        post_patch_num_frames = padded_num_frames // p_t
        post_patch_height = padded_height // p_h
        post_patch_width = padded_width // p_w

        # Select appropriate patch embedding based on stage
        patch_embedding = self.patch_embeddings[stage_idx]
        rotary_emb = self.rope[stage_idx](hidden_states)
        
        # Patch embedding
        hidden_states = patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, seq_len, inner_dim)
        assert hidden_states.shape[1] <= self.rope_max_seq_len[stage_idx], (
            f"Sequence length {hidden_states.shape[1]} is greater than maximum sequence length "
            f"{self.rope_max_seq_len[stage_idx]} for stage {stage_idx}"
        )
        # Select transformer blocks
        if self.reusing:
            transformer_blocks = self.transformer.blocks
        else:
            blocks_per_stage = self.num_layers // 3
            transformer_blocks = self.transformer.blocks[stage_idx * blocks_per_stage : (stage_idx + 1) * blocks_per_stage]

        # Run transformer blocks
        attention_mask = self.create_spatial_mask(
            window_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            hidden_states.device,
        )
        if torch.is_grad_enabled() and getattr(self.transformer, 'gradient_checkpointing', False):
            for block in transformer_blocks:
                hidden_states, attn_weight = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    rotary_emb,
                    attention_mask,
                    use_reentrant=False
                )
                self.attn_weights.append(attn_weight)
        else:
            for block in transformer_blocks:
                hidden_states, attn_weight = block(
                    hidden_states,
                    rotary_emb,
                    attention_mask,
                )
                self.attn_weights.append(attn_weight)

        # Output norm & projection
        norm_out = self.norm_outs[stage_idx]
        proj_out = self.proj_outs[stage_idx]
        
        hidden_states = norm_out(hidden_states.float()).type_as(hidden_states)
        hidden_states = proj_out(hidden_states)

        # Unpatchify
        out_channels = self.channels[stage_idx]
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
            p_t, p_h, p_w, out_channels
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if pad_t or pad_h or pad_w:
            output = output[:, :, :num_frames, :height, :width]

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
