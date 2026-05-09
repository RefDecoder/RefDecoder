import math
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

from src.models.VideoVAEPlus.autoencoder import AutoencoderKL
from src.modules.ae_modules import Normalize, nonlinearity
from src.modules.attention_temporal_videoae import *
from src.modules.t5 import T5Embedder
from src.distributions import DiagonalGaussianDistribution
from src.models.VideoVAEPlus.autoencoder_temporal import EncoderTemporal1DCNN, DecoderTemporal1DCNN
from src.models.Wan.transformer_wan import WanDecoderTransformer
from utils.debug_decoderT import MinimalVideoVaePlusRefTracker
try:
    import xformers
    import xformers.ops as xops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

GRADIENT_CHECKPOINTING = False

class RefConvIn(nn.Module):
    """
    Tokenizes reference videos by converting spatial resolution into channels.
    Uses strided Conv3d patchification.
    Converts [b, c, T, h, w] to [b, c_out, T, h/patch_size, w/patch_size]
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=512,
        patch_size=8,
        gradient_checkpointing=False,
    ):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB)
            out_channels (int): Number of output channels
            patch_size (int): Size of spatial patches for downsampling
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing

        self.patch_channels = in_channels * patch_size * patch_size
        self.proj = nn.Conv3d(
            self.patch_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )
        # self.norm = Normalize(out_channels)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        """
        Tokenize reference input using strided Conv3d patchification.

        Args:
            x: Input tensor [b, in_channels, T, h, w]

        Returns:
            Tokenized tensor [b, out_channels, T, h/patch_size, w/patch_size]
        """
        b, c, t, h, w = x.shape
        patch_size = self.patch_size
        assert h % patch_size == 0, f"Height {h} must be divisible by patch_size {patch_size}"
        assert w % patch_size == 0, f"Width {w} must be divisible by patch_size {patch_size}"

        # patchify: [b, c, t, h, w] -> [b, c*patch^2, t, h/patch, w/patch]
        x = x.view(b, c, t, h // patch_size, patch_size, w // patch_size, patch_size)
        x = x.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
        x = x.view(b, self.patch_channels, t, h // patch_size, w // patch_size)

        if self.gradient_checkpointing and torch.is_grad_enabled():
            x = torch.utils.checkpoint.checkpoint(self.proj, x, use_reentrant=False)
        else:
            x = self.proj(x)
            
        # x = self.norm(x)
        return x

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.conv = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(1, 1, 1),
        )
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        h = x

        h = self.norm(h)
        h = nonlinearity(h)
        h = self.conv(h)
        return h


class ResnetBlock2plus1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        kernel_size_t=3,
        padding_t=1,
        stride_t=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )

        self.conv1_tmp = TemporalConvLayer(out_channels, out_channels)

        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )

        self.conv2_tmp = TemporalConvLayer(out_channels, out_channels)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 3, 3),
                    stride=1,
                    padding=(0, 1, 1),
                )
            else:
                self.nin_shortcut = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1, 1),
                    stride=1,
                    padding=(0, 0, 0),
                )
        self.conv3_tmp = TemporalConvLayer(out_channels, out_channels)

    def forward(self, x, temb, mask_temporal=False):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if not mask_temporal:
            h = self.conv1_tmp(h) + h

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if not mask_temporal:
            h = self.conv2_tmp(h) + h

        # skip connections
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
            if not mask_temporal:
                x = self.conv3_tmp(x) + x

        return x + h


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        # q = q.reshape(b,c,h*w) # bcl
        # q = q.permute(0,2,1)   # bcl -> blc l=hw
        # k = k.reshape(b,c,h*w) # bcl
        q = rearrange(q, "b c t h w -> (b t) (h w) c")  # blc
        k = rearrange(k, "b c t h w -> (b t) c (h w)")  # bcl

        w_ = torch.bmm(q, k)  # b,l,l
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # v = v.reshape(b,c,h*w)
        v = rearrange(v, "b c t h w -> (b t) c (h w)")  # bcl

        # attend to values
        w_ = w_.permute(0, 2, 1)  # bll
        h_ = torch.bmm(v, w_)  # bcl

        # h_ = h_.reshape(b,c,h,w)
        h_ = rearrange(h_, "(b t) c (h w) -> b c t h w", b=b, h=h)

        h_ = self.proj_out(h_)

        return x + h_


# ---------------------------------------------------------------------------------------------------


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        patch_size=1,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.patch_size = patch_size
        patch_dim = query_dim * patch_size * patch_size
        self.norm = nn.LayerNorm(patch_dim)

        self.to_q = nn.Linear(patch_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, patch_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        b, c, t, height, width = x.shape

        # patch: [patch_size, patch_size]
        divide_factor_height = height // self.patch_size
        divide_factor_width = width // self.patch_size
        x = rearrange(
            x,
            "b c t (df1 ph) (df2 pw) -> (b t) (df1 df2) (ph pw c)",
            df1=divide_factor_height,
            df2=divide_factor_width,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.norm(x)

        context = default(context, x)
        context = repeat(context, "b n d -> (b t) n d", b=b, t=t)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), (q, k, v)
        )

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(mask, "b j -> (b t h) () j", t=t, h=self.heads)

        if XFORMERS_IS_AVAILBLE:
            if exists(mask):
                mask = mask.to(q.dtype)
                max_neg_value = -torch.finfo(q.dtype).max

                attn_bias = torch.zeros_like(mask)
                attn_bias.masked_fill_(mask <= 0.5, max_neg_value)

                mask = mask.detach().cpu()
                attn_bias = attn_bias.expand(-1, q.shape[1], -1)

                attn_bias_expansion_q = (attn_bias.shape[1] + 7) // 8 * 8
                attn_bias_expansion_k = (attn_bias.shape[2] + 7) // 8 * 8

                attn_bias_expansion = torch.zeros(
                    (attn_bias.shape[0], attn_bias_expansion_q, attn_bias_expansion_k),
                    dtype=attn_bias.dtype,
                    device=attn_bias.device,
                )
                attn_bias_expansion[:, : attn_bias.shape[1], : attn_bias.shape[2]] = (
                    attn_bias
                )

                attn_bias = attn_bias.detach().cpu()

                out = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=attn_bias_expansion[
                        :, : attn_bias.shape[1], : attn_bias.shape[2]
                    ],
                    scale=self.scale,
                )
            else:
                out = xops.memory_efficient_attention(q, k, v, scale=self.scale)
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
            if exists(mask):
                max_neg_value = -torch.finfo(sim.dtype).max
                sim.masked_fill_(~(mask > 0.5), max_neg_value)
            attn = sim.softmax(dim=-1)
            out = einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)

        ret = self.to_out(out)
        ret = rearrange(
            ret,
            "(b t) (df1 df2) (ph pw c) -> b c t (df1 ph) (df2 pw)",
            b=b,
            t=t,
            df1=divide_factor_height,
            df2=divide_factor_width,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return ret


# ---------------------------------------------------------------------------------------------------


class TemporalAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        max_temporal_length=64,
    ):
        """
        a clean multi-head temporal attention
        """
        super().__init__()

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)
        self.qkv = zero_module(conv_nd(1, channels, channels * 3, 1))
        self.attention = QKVAttention(self.num_heads)
        self.relative_position_k = RelativePosition(
            num_units=channels // self.num_heads,
            max_relative_position=max_temporal_length,
        )
        self.relative_position_v = RelativePosition(
            num_units=channels // self.num_heads,
            max_relative_position=max_temporal_length,
        )
        self.proj_out = zero_module(
            conv_nd(1, channels, channels, 1)
        )  # conv_dim, in_channels, out_channels, kernel_size

    def forward(self, x, mask=None):
        b, c, t, h, w = x.shape
        out = rearrange(x, "b c t h w -> (b h w) c t")

        qkv = self.qkv(self.norm(out))

        len_q = qkv.size()[-1]
        len_k, len_v = len_q, len_q

        k_rp = self.relative_position_k(len_q, len_k)
        v_rp = self.relative_position_v(len_q, len_v)  # [T,T,head_dim]
        out = self.attention(qkv, rp=(k_rp, v_rp))

        out = self.proj_out(out)
        out = rearrange(out, "(b h w) c t -> b c t h w", b=b, h=h, w=w)

        return x + out


# ---------------------------------------------------------------------------------------------------


class Downsample2plus1D(nn.Module):
    """spatial downsample, in a factorized way"""

    def __init__(self, in_channels, with_conv, temp_down):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        self.temp_down = temp_down

        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=0,
            )

    def forward(self, x, mask_temporal):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 0)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
            # print(f'[Encoder-Downsample] after conv={x.shape}')
            # print(f'[Encoder-Downsample] after conv_tmp={x.shape}')
        else:
            raise NotImplementedError
            # x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class Upsample2plus1D(nn.Module):
    def __init__(self, in_channels, with_conv, temp_up):
        super().__init__()
        self.with_conv = with_conv
        self.in_channels = in_channels
        self.temp_up = temp_up
        if self.with_conv:
            self.conv = torch.nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
            )

    def forward(self, x, mask_temporal, is_reference=False):
        # print(f'[Decoder-Upsample] input={x.shape}')
        if not self.temp_up or mask_temporal or is_reference:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> b (c t) h w")
            x = torch.nn.functional.interpolate(
                x, scale_factor=(2.0, 2.0), mode="nearest"
            )
            x = rearrange(x, "b (c t) h w -> b c t h w", t=t)
        else:
            # import pdb;pdb.set_trace()
            x = torch.nn.functional.interpolate(
                x, scale_factor=(2.0, 2.0, 2.0), mode="nearest"
            )
            

        if self.with_conv:
            x = self.conv(x)

        return x


# ---------------------------------------------------------------------------------------------------


class Encoder2plus1D(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        temporal_down_factor,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        mask_temporal=False,
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)  # spatial resolutions
        self.n_temporal_down = int(
            math.log2(temporal_down_factor)
        )  # temporal resolutions
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv3d(
            in_channels, self.ch, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        cur_patch_size = 8

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock2plus1D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        CrossAttention(
                            query_dim=block_in,
                            patch_size=cur_patch_size,
                            context_dim=1024,
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                temp_down = i_level <= self.n_temporal_down - 1
                down.downsample = Downsample2plus1D(
                    block_in, resamp_with_conv, temp_down
                )
                curr_res = curr_res // 2
                cur_patch_size //= 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2plus1D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.attn_1_tmp = TemporalAttention(block_in, num_heads=1)
        self.mid.block_2 = ResnetBlock2plus1D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )

    def forward(
        self, x, text_embeddings=None, text_attn_mask=None, mask_temporal=False
    ):
        # timestep embedding
        temb = None

        # print(f'[Encoder] input={x.shape}')
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb, mask_temporal)
                if len(self.down[i_level].attn) > 0:
                    h = h + self.down[i_level].attn[i_block](
                        h, context=text_embeddings, mask=text_attn_mask
                    )
                # print(f'[Encoder] after down block={h.shape}')
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1], mask_temporal))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb, mask_temporal)
        h = self.mid.attn_1(h)

        if not mask_temporal:
            h = self.mid.attn_1_tmp(h)

        h = self.mid.block_2(h, temb, mask_temporal)
        # print(f'[Encoder] after mid block = {h.shape}')
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # print(f'[Encoder] after conv_out = {h.shape}')

        return h


class Decoder2plus1D(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        temporal_down_factor,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        mask_temporal=False,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)  # spatial resolutions
        self.n_temporal_up = int(
            math.log2(temporal_down_factor)
        )  # temporal resolutions
        self.n_spatial_up = self.num_resolutions - 1  # 3
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv3d(
            z_channels, block_in, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock2plus1D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.attn_1_tmp = TemporalAttention(block_in, num_heads=1)
        self.mid.block_2 = ResnetBlock2plus1D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        # print(f'[Decoder] mid block feature, temporal length={self.input_length//(2 ** self.num_resolutions)}')
        # upsampling
        self.up = nn.ModuleList()

        cur_patch_size = 1

        for i_level in reversed(range(self.num_resolutions)):  # 3210
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock2plus1D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        CrossAttention(
                            query_dim=block_in,
                            patch_size=cur_patch_size,
                            context_dim=1024,
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                temp_up = i_level <= self.num_resolutions - 1 - (
                    self.n_spatial_up - self.n_temporal_up
                )
                up.upsample = Upsample2plus1D(block_in, resamp_with_conv, temp_up)
                curr_res = curr_res * 2
                cur_patch_size *= 2
            self.up.insert(0, up)  # prepend to get consistent order

        # GroupNorm applied to decoder features before each transformer insertion point
        # channels: stage0=512 (i_level=1), stage1=256 (i_level=0), stage2=128 (after loop)
        self.pre_transformer_norms = nn.ModuleList([
            Normalize(512),
            Normalize(256),
            Normalize(128),
        ])
        self.pre_transformer_norms_ref = nn.ModuleList([
            Normalize(512),
            Normalize(256),
            Normalize(128),
        ])

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in, out_ch, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )
        self.gradient_checkpointing = GRADIENT_CHECKPOINTING
        
    def forward(
        self, 
        z, 
        text_embeddings=None, 
        text_attn_mask=None, 
        mask_temporal=False, 
        transformer=None,
        reference_token=None,
    ):
        self.last_z_shape = z.shape

        # print(f'[Decoder] input={z.shape}')
        # timestep embedding
        temb = None

        # z to block_in
        if self.gradient_checkpointing:
            h = torch.utils.checkpoint.checkpoint(self.conv_in, z, use_reentrant=False)
        else:
            h = self.conv_in(z)
        # print(f'[Decoder] after conv_in ={h.shape}')

        # middle
        if self.gradient_checkpointing:
            h = torch.utils.checkpoint.checkpoint(self.mid.block_1, h, temb, mask_temporal, use_reentrant=False)
            h = torch.utils.checkpoint.checkpoint(self.mid.attn_1, h, use_reentrant=False)

            if not mask_temporal:
                h = torch.utils.checkpoint.checkpoint(self.mid.attn_1_tmp, h, use_reentrant=False)
        else:
            h = self.mid.block_1(h, temb, mask_temporal)
            h = self.mid.attn_1(h)

            if not mask_temporal:
                h = self.mid.attn_1_tmp(h)

        if self.gradient_checkpointing:
            h = torch.utils.checkpoint.checkpoint(self.mid.block_2, h, temb, mask_temporal, use_reentrant=False)
        else:
            h = self.mid.block_2(h, temb, mask_temporal)
        # print(f'[Decoder] after mid blocks ={h.shape}')

        i_level2stage_index = {1:0, 0: 1}
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # RefDecoder Transformer Block
            if i_level2stage_index.get(i_level) is not None:
                stage_idx = i_level2stage_index[i_level]
                norm = self.pre_transformer_norms[stage_idx]
                ref_norm = self.pre_transformer_norms_ref[stage_idx]
                h_orig = h
                h = norm(h)
                reference_token = ref_norm(reference_token)
                h = torch.cat([reference_token, h], dim=2)
                h = transformer(
                    hidden_states=h,
                    stage_idx=stage_idx,
                    return_dict=True,
                    window_size=-1,
                ).sample
                reference_token, h = h[:, :, :1], h[:, :, 1:]
                h = h + h_orig

            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, mask_temporal)
                if i_level == 1 or i_level == 0:
                    reference_token = self.up[i_level].block[i_block](reference_token, temb, mask_temporal)
                if len(self.up[i_level].attn) > 0:
                    h = h + self.up[i_level].attn[i_block](
                        h, context=text_embeddings, mask=text_attn_mask
                    )
                # print(f'[Decoder] after up block ={h.shape}')
            if i_level != 0:
                h = self.up[i_level].upsample(h, mask_temporal)
                if i_level == 1:
                    reference_token = self.up[i_level].upsample(reference_token, mask_temporal)
                # print(f'[Decoder] after upsample ={h.shape}')
        
        h_orig = h
        norm = self.pre_transformer_norms[2]
        ref_norm = self.pre_transformer_norms_ref[2]
        h = norm(h)
        reference_token = ref_norm(reference_token)
        h = torch.cat([reference_token, h], dim=2)
        h = transformer(
            hidden_states=h,
            stage_idx=2,
            return_dict=True,
            window_size=-1,
        ).sample
        reference_token, h = h[:, :, :1], h[:, :, 1:]
        h = h + h_orig

        # end
        if self.give_pre_end:
            return h
        if self.gradient_checkpointing:
            h = torch.utils.checkpoint.checkpoint(self.norm_out, h, use_reentrant=False)
            h = torch.utils.checkpoint.checkpoint(nonlinearity, h, use_reentrant=False)
            h = torch.utils.checkpoint.checkpoint(self.conv_out, h, use_reentrant=False)
        else:
            h = self.norm_out(h)
            h = nonlinearity(h)
            h = self.conv_out(h)
        # print(f'[Decoder] after conv_out ={h.shape}')
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class AutoencoderKL2plus1D_1dcnn(AutoencoderKL):
    def __init__(
        self,
        ddconfig,
        ppconfig,
        lossconfig,
        embed_dim=0,
        use_quant_conv=True,
        test=False,
        ckpt_path=None,
        ckpt_path_2d=None,
        ckpt_path_4temporal=None,
        ignore_keys_3d=[],
        img_video_joint_train=False,
        video_key="",
        caption_guide=False,
        t5_model_max_length=120,
        gradient_checkpointing=False,
        # =====Self Attention Layers=====
        reusing=True,
        pretrained=False,
        num_layers=30,
        num_heads=12,
        head_dim=128,
        # ======LoRA OPTIONS=====
        use_lora=False, 
        lora_rank=32, 
        lora_alpha=32,
        lora_dropout=0.1, 
        # ======DEBUG OPTIONS=====
        inference_w_dropout=False,
        dropout_p=0.7,
        debug_tracker=True,
        *args,
        **kwargs,
    ):
        super(AutoencoderKL2plus1D_1dcnn, self).__init__(
            ddconfig,
            lossconfig,
            embed_dim,
            use_quant_conv,
            *args,
            test=False,
            **kwargs,
        )
        self.img_video_joint_train = img_video_joint_train
        self.caption_guide = caption_guide
        self.video_key = video_key
        self.t5_model_max_length = t5_model_max_length
        self.use_quant_conv = use_quant_conv
        self.gradient_checkpointing = gradient_checkpointing

        global GRADIENT_CHECKPOINTING
        GRADIENT_CHECKPOINTING = gradient_checkpointing
        self.encoder_temporal = EncoderTemporal1DCNN(**ppconfig)
        self.decoder_temporal = DecoderTemporal1DCNN(
            gradient_checkpointing=gradient_checkpointing, **ppconfig
        )

        self.inference_w_dropout = inference_w_dropout
        self.dropout_p = dropout_p

        self.encoder = Encoder2plus1D(**ddconfig)
        self.decoder = Decoder2plus1D(**ddconfig)


        self.ref_conv_in = RefConvIn(
            in_channels=3,
            out_channels=512,
            patch_size=2,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.transformer = WanDecoderTransformer(
            rope_max_seq_len=[17 * 1620, 17 * 1620, 17 * 1620],
            patch_size = [(1, 8, 8), (1, 16, 16), (1, 16, 16)],
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            channels=[512, 256, 128],  # stage0 (i_level=2, before blocks): 512ch; stage1 (i_level=1, before blocks): 512ch; stage2 (i_level=0, unused): 256ch
            reusing=reusing,
            pretrained=pretrained,
            # Apply LoRA to transformer if enabled
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

        if use_quant_conv:
            assert embed_dim
            self.embed_dim = embed_dim
            self.quant_conv = torch.nn.Conv3d(
                2 * ddconfig["z_channels"], 2 * embed_dim, 1
            )
            self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)

        # load pretrian vae
        self.init_from_ckpt("ckpt/VideoVAEPlus/sota-4-16z.ckpt", ignore_keys=ignore_keys_3d)

        self.freeze_encoder()
        self.freeze_decoder_temporal()
        self.loss.eval()
        for p in self.loss.parameters():
            p.requires_grad = False

        # self.loss_fn = lpips.LPIPS(net="alex")
        # for p in self.loss_fn.parameters():
        #     p.requires_grad = False
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys_3d)
        if ckpt_path_2d is not None:
            self.init_from_2dckpt(ckpt_path_2d)
        if ckpt_path_4temporal is not None:
            self.init_from_4temporal(ckpt_path_4temporal, ignore_keys=ignore_keys_3d)

        if test:
            self.init_test()

        self.enable_text_embedder = False

        self.tracker = MinimalVideoVaePlusRefTracker(self) if debug_tracker else None

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        
        # Extract the actual state dict
        if "state_dict" in sd:
            self._cur_epoch = sd.get("epoch", "null")
            sd = sd["state_dict"]
        elif "module" in sd:
            self._cur_epoch = sd.get("global_step", "null")
            sd = sd["module"]
        else:
            self._cur_epoch = "null"

        # Clean up keys
        cleaned_sd = {}
        for k, v in sd.items():
            # Skip ignored keys
            if any(k.startswith(ik) for ik in ignore_keys):
                continue
            # Remove DeepSpeed and torch.compile prefixes
            new_key = k.replace("_forward_module.", "").replace("_orig_mod.", "")
            cleaned_sd[new_key] = v

        load_info = self.load_state_dict(cleaned_sd, strict=False)
        
        if "sota" in path:
            return  # skip printing for sota checkpoint which is large and has many missing keys

        print(f"\n====== Loaded checkpoint from {path} ======")
        print("Loaded keys:", len(sd))
        
        print("\n Missing keys (model has, ckpt doesn't):")
        for k in load_info.missing_keys:
            print("  -", k)

        print("\n Unexpected keys (ckpt has, model doesn't):")
        for k in load_info.unexpected_keys:
            print("  -", k)

        print("==========================================\n")


    def init_from_2dckpt(self, ckpt_path_2d):
        sd = torch.load(ckpt_path_2d, map_location="cpu")
        try:
            sd = sd["state_dict"]
        except:
            pass
        sd_new = {}
        for k in sd.keys():
            if k.startswith("first_stage_model."):
                new_key = k.split("first_stage_model.")[-1]
                # print(f"k={k},para={sd[k].shape}")
                v = sd[k]
                if v.dim() == 4:
                    v = v.unsqueeze(2)
                sd_new[new_key] = v

        self.load_state_dict(sd_new, strict=False)
        print(f"Restored from {ckpt_path_2d}")

    def get_text_embeddings(self, captions):
        # print(f"caption is {captions}")
        # print(f"{self.device} enable T5?: {self.enable_text_embedder}")
        if not self.enable_text_embedder:
            self.enable_text_embedder = True
            self.text_embedder = T5Embedder(
                device=self.device, model_max_length=self.t5_model_max_length
            )
        return self.text_embedder.get_text_embeddings(captions)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder_temporal.parameters():
            param.requires_grad = False
        if self.use_quant_conv:
            for param in self.quant_conv.parameters():
                param.requires_grad = False
    
    def freeze_decoder_temporal(self):
        for param in self.decoder_temporal.parameters():
            param.requires_grad = False
        if self.use_quant_conv:
            for param in self.post_quant_conv.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        lr = self.learning_rate
        
        # decoder
        norm_param_ids = (
            {id(p) for p in self.decoder.pre_transformer_norms.parameters()} |
            {id(p) for p in self.decoder.pre_transformer_norms_ref.parameters()}
        )
        ae_params = []
        for param in self.decoder_temporal.parameters():
            if param.requires_grad:
                ae_params.append(param)
        for param in self.decoder.parameters():
            if param.requires_grad and id(param) not in norm_param_ids:
                ae_params.append(param)

        ref_params = []
        for param in self.ref_conv_in.parameters():
            if param.requires_grad:
                ref_params.append(param)
        for param in self.transformer.parameters():
            if param.requires_grad:
                ref_params.append(param)
        for param in self.decoder.pre_transformer_norms.parameters():
            if param.requires_grad:
                ref_params.append(param)
        for param in self.decoder.pre_transformer_norms_ref.parameters():
            if param.requires_grad:
                ref_params.append(param)

        param_groups = []
        if ref_params:
            param_groups.append({"params": ref_params, "lr": lr})
        if ae_params:
            param_groups.append({"params": ae_params, "lr": lr * 0.1})

        opt_ae = torch.optim.AdamW(
            param_groups,
            betas=(0.5, 0.9),
        )

        # Linear warmup from 0 to 1.0
        warmup_steps = 1000
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            opt_ae,
            start_factor=0.01,  # Start at 1% of lr
            end_factor=1.0,     # End at 100% of lr
            total_iters=warmup_steps
        )

        # Cosine annealing after warmup
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_ae,
            T_max=100000 - warmup_steps,  # Remaining steps after warmup
            eta_min=lr * 0.01  # Minimum learning rate (1% of base lr)
        )

        # Combine warmup and cosine
        scheduler_ae = torch.optim.lr_scheduler.SequentialLR(
            opt_ae,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        # opt_disc = torch.optim.Adam(
        #     self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        # )

        # return [opt_ae, opt_disc], []

        return [opt_ae], [{"scheduler": scheduler_ae, "interval": "step", "frequency": 1}]


    def encode_temporal(self, x, text_embeddings=None, text_attn_mask=None):
        # x: [b c t h w] h: [b c t//4 h w]
        # b = x.shape[0]
        moments = self.encoder_temporal(x, text_embeddings, text_attn_mask)
        posterior = DiagonalGaussianDistribution(moments)
        # posterior = rearrange(posterior, '(b t) c h w -> b c t h w', b=b)
        return posterior

    def decode_temporal(self, z, text_embeddings=None, text_attn_mask=None):
        # z: [b c t h w] dec: [b c t//4 h w]
        dec = self.decoder_temporal(z, text_embeddings, text_attn_mask)
        return dec

    def encode_2plus1d(
        self, x, text_embeddings=None, text_attn_mask=None, mask_temporal=False
    ):
        h = self.encoder(
            x, text_embeddings, text_attn_mask, mask_temporal=mask_temporal
        )
        if self.use_quant_conv:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode_2plus1d(
        self, z, text_embeddings=None, text_attn_mask=None, mask_temporal=False, transformer=None, reference_token=None
    ):
        if self.use_quant_conv:
            z = self.post_quant_conv(z)

        z = self._apply_token_dropout(z)
        # reference_token = self._apply_token_dropout(reference_token)

        dec = self.decoder(
            z, text_embeddings, text_attn_mask, mask_temporal=mask_temporal,transformer=transformer, reference_token=reference_token
        )
        return dec

    def _apply_token_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply token dropout to the input tensor.

        Args:
            x: Input tensor of shape [B, C, T, H, W]

        Returns:
            Tensor with random tokens dropped (set to zero)
        """
        if self.inference_w_dropout or self.training:
            # p = self.dropout_p
            p = torch.rand(1).item() * self.dropout_p
            dropped = torch.rand_like(x[:, :1, :1, :, :]) < p
            x = torch.where(dropped, torch.zeros_like(x), x)
        return x

    def encode(
        self,
        x,
        text_embeddings=None,
        text_attn_mask=None,
        sample_posterior=True,
        mask_temporal=False,
    ):
        # [b, c, t, h, w]
        posterior = self.encode_2plus1d(
            x, text_embeddings, text_attn_mask, mask_temporal=mask_temporal
        )
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = z.to(device=self.device, dtype=x.dtype)
        if not mask_temporal:
            posterior = self.encode_temporal(z, text_embeddings, text_attn_mask)
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
            z = z.to(device=self.device, dtype=x.dtype)

        return z, posterior

    def decode(
        self, 
        z, 
        text_embeddings=None, 
        text_attn_mask=None, 
        mask_temporal=False, 
        transformer=None,
        reference_token=None
    ):
        if transformer is None or reference_token is None:
            raise ValueError("transformer and reference_token must be provided")
        
        z = z.to(device=next(self.decoder.parameters()).device, dtype=next(self.decoder.parameters()).dtype)
        
        if not mask_temporal:
            z = self.decode_temporal(z, text_embeddings, text_attn_mask)

        dec = self.decode_2plus1d(
            z, text_embeddings, text_attn_mask, mask_temporal=mask_temporal, transformer=transformer, reference_token=reference_token)
        return dec

    def forward(
        self,
        inputs,
        text_embeddings=None,
        text_attn_mask=None,
        sample_posterior=True,
        mask_temporal=False,
    ):
        # [b, c, t, h, w] input
        z, posterior = self.encode(
            inputs,
            text_embeddings,
            text_attn_mask,
            sample_posterior,
            mask_temporal=mask_temporal,
        )

        ref_idx = 0 if inputs.shape[2] == 1 else torch.randint(0, inputs.shape[2], ()).item()
        reference_frame = (inputs[:, :, ref_idx:ref_idx + 1, :, :].clone()).to(device=inputs.get_device(), dtype=inputs.dtype)
        reference_token = self.ref_conv_in(reference_frame)
        
        dec = self.decode(
            z, text_embeddings, text_attn_mask, mask_temporal=mask_temporal, transformer=self.transformer, reference_token=reference_token
        )
        return dec, posterior, ref_idx

    def training_step(self, *args):
        # if self.img_video_joint_train:
        #     return self.training_step_joint(*args)
        # else:
        return self.training_step_video(*args)

    def training_step_video(self, batch, batch_idx):
        inputs = self.get_input(batch, self.video_key)

        # if self.caption_guide:
        #     text_embeddings, text_attn_mask = self.get_text_embeddings(batch["caption"])
        #     reconstructions, posterior = self(inputs, text_embeddings, text_attn_mask)
        #     del text_embeddings, text_attn_mask
        # else:
        reconstructions, posterior, ref_idx = self(inputs)

        # if optimizer_idx == 0:
        # TODO: logvar
        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log(
           "train/aeloss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        self.log_dict(
            log_dict_ae, 
            prog_bar=False, 
            logger=True, 
            on_step=True, 
            on_epoch=False, 
            sync_dist=True
        )

        if self.global_step % 2 == 0:
            self._compute_and_log_psnr(inputs, reconstructions, prefix="train")
            # self._compute_and_log_ssim(inputs, reconstructions, prefix="train")
            # self._compute_and_log_lpips(inputs, reconstructions, prefix="train")

            inputs_ref = inputs[:, :, ref_idx:ref_idx + 1, :, :].clone()
            rec_ref = reconstructions[:, :, ref_idx:ref_idx + 1, :, :].clone()
            self._compute_and_log_psnr(inputs_ref, rec_ref, prefix="train_ref")

            # log other frames reconsturctions metrics
            if inputs.size(2) != 1:
                inputs_other = torch.cat([inputs[:, :, :ref_idx, :, :], inputs[:, :, ref_idx + 1:, :, :]], dim=2)
                reconstructions_other = torch.cat([reconstructions[:, :, :ref_idx, :, :], reconstructions[:, :, ref_idx + 1:, :, :]], dim=2)
                self._compute_and_log_psnr(inputs_other, reconstructions_other, prefix="train_other")

        if self.tracker is not None:
            tracker_stats = self.tracker.get_stats()
            self.log_dict(
                {f"debug/{k}": v for k, v in tracker_stats.items()},
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.tracker.clear()

        return aeloss

        # if optimizer_idx == 1:
        #     # train the discriminator
        #     discloss, log_dict_disc = self.loss(
        #         inputs,
        #         reconstructions,
        #         posterior,
        #         optimizer_idx,
        #         self.global_step,
        #         last_layer=self.get_last_layer(),
        #         split="train",
        #     )

        #     self.log(
        #         "discloss",
        #         discloss,
        #         prog_bar=True,
        #         logger=True,
        #         on_step=True,
        #         on_epoch=True,
        #     )
        #     self.log_dict(
        #         log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
        #     )
        #     return discloss

    # def training_step_joint(self, batch, batch_idx, optimizer_idx):
    #     opt_ae, opt_disc = self.optimizers()

    #     inputs = self.get_input(batch, self.video_key)
    #     is_video = self.get_input(batch, "is_video")
    #     is_mask = not is_video.all()

    #     if not is_video.all():
    #         # (b, c, 16, h, w) -> (16, c, 1, h, w)
    #         inputs = inputs.permute(2, 1, 0, 3, 4)

    #     if self.caption_guide:
    #         text_embeddings, text_attn_mask = self.get_text_embeddings(batch["caption"])
    #         reconstructions, posterior = self(
    #             inputs, text_embeddings, text_attn_mask, mask_temporal=is_mask
    #         )
    #         del text_embeddings, text_attn_mask
    #     else:
    #         reconstructions, posterior = self(inputs, mask_temporal=is_mask)

    #     # print(f"the temporal is masked: {is_mask}")
    #     if optimizer_idx == 0:
    #         # train encoder+decoder+logvar
    #         aeloss, log_dict_ae = self.loss(
    #             inputs,
    #             reconstructions,
    #             posterior,
    #             global_step=self.global_step,
    #             last_layer=self.get_last_layer(),
    #             split="train",
    #         )

    #         self.log(
    #            "train/aeloss",
    #             aeloss,
    #             prog_bar=True,
    #             logger=True,
    #             on_step=True,
    #             on_epoch=True,
    #             sync_dist=True
    #         )
    #         self.log_dict(
    #             log_dict_ae, 
    #             prog_bar=False, 
    #             logger=True, 
    #             on_step=True, 
    #             on_epoch=False,
    #             sync_dist=True
    #         )
    #         return aeloss

    #     if optimizer_idx == 1:
    #         # train the discriminator
    #         discloss, log_dict_disc = self.loss(
    #             inputs,
    #             reconstructions,
    #             posterior,
    #             global_step=self.global_step,
    #             last_layer=self.get_last_layer(),
    #             split="train",
    #         )

    #         self.log(
    #             "discloss",
    #             discloss,
    #             prog_bar=True,
    #             logger=True,
    #             on_step=True,
    #             on_epoch=True,
    #         )
    #         self.log_dict(
    #             log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
    #         )
    #         return discloss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        inputs = self.get_input(batch, self.video_key)
        is_video = self.get_input(batch, "is_video")
        is_mask = not is_video.all()

        # if self.caption_guide:
        #     text_embeddings, text_attn_mask = self.get_text_embeddings(batch["caption"])
        #     reconstructions, posterior = self(
        #         inputs, text_embeddings, text_attn_mask, mask_temporal=is_mask
        #     )
        #     del text_embeddings, text_attn_mask
        # else:
        reconstructions, posterior, ref_idx = self(inputs, mask_temporal=is_mask)

        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        del reconstructions

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        torch.cuda.empty_cache()
        return self.log_dict

    def _compute_and_log_psnr(self, inputs, reconstructions, prefix="train"):
        with torch.no_grad():
            inputs_np = inputs.detach().cpu().float().numpy()
            reconstructions_np = reconstructions.detach().cpu().float().numpy()

            inputs_np = (inputs_np + 1.0) / 2.0
            reconstructions_np = (reconstructions_np + 1.0) / 2.0

            psnr_value = psnr(inputs_np, reconstructions_np, data_range=1.0)
            self.log(
                f"{prefix}/psnr",
                psnr_value,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            return psnr_value

    def _compute_and_log_ssim(self, inputs, reconstructions, prefix="train"):
        with torch.no_grad():
            inputs_np = inputs.detach().cpu().float().numpy()
            reconstructions_np = reconstructions.detach().cpu().float().numpy()

            inputs_np = (inputs_np + 1.0) / 2.0
            reconstructions_np = (reconstructions_np + 1.0) / 2.0

            b, c, t, h, w = inputs_np.shape
            ssim_values = []
            for batch_idx in range(b):
                for time_idx in range(t):
                    img1 = inputs_np[batch_idx, :, time_idx, :, :].transpose(1, 2, 0)
                    img2 = reconstructions_np[batch_idx, :, time_idx, :, :].transpose(1, 2, 0)
                    min_dim = min(h, w)
                    win_size = 3 if min_dim < 7 else (7 if min_dim < 11 else 11)
                    frame_ssim = ssim(
                        img1,
                        img2,
                        data_range=1.0,
                        win_size=win_size,
                        channel_axis=-1,
                        multichannel=True,
                    )
                    ssim_values.append(frame_ssim)

            ssim_value = sum(ssim_values) / len(ssim_values)
            self.log(
                f"{prefix}/ssim",
                ssim_value,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            return ssim_value

    def _compute_and_log_lpips(self, inputs, reconstructions, prefix="train"):
        with torch.no_grad():
            self.loss_fn = self.loss_fn.to(inputs.device)
            b, c, t, h, w = inputs.shape
            lpips_values = []
            for frame_idx in range(t):
                input_frame = inputs[:, :, frame_idx, :, :]
                recon_frame = reconstructions[:, :, frame_idx, :, :]
                lpips_frame = self.loss_fn(input_frame, recon_frame).mean()
                lpips_values.append(lpips_frame.item())

            lpips_value = sum(lpips_values) / len(lpips_values)
            self.log(
                f"{prefix}/lpips",
                lpips_value,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            return lpips_value

    @torch.no_grad()
    def log_images_joint(self, batch, only_inputs=False, **kwargs):
        raise NotImplementedError("Only accep videos training")
        log = dict()
        x = self.get_input(batch, self.video_key)
        x = x.to(self.device)
        is_video = self.get_input(batch, "is_video")
        is_mask = not is_video.all()

        if not is_video.all():
            # (b, c, 16, h, w) -> (16, c, 1, h, w)
            x = x.permute(2, 1, 0, 3, 4)

        if self.caption_guide:
            text_embeddings, text_attn_mask = self.get_text_embeddings(batch["caption"])
            text_embeddings = text_embeddings.to(device=self.device, dtype=x.dtype)
            text_attn_mask = text_attn_mask.to(device=self.device, dtype=x.dtype)

        if not only_inputs:
            if self.caption_guide:
                xrec, posterior, _ = self(
                    x, text_embeddings, text_attn_mask, mask_temporal=is_mask
                )
            else:
                xrec, posterior, _ = self(x, mask_temporal=is_mask)

            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)

            if self.caption_guide:
                log["samples"] = self.decode(
                    torch.randn_like(posterior.sample()),
                    text_embeddings,
                    text_attn_mask,
                    mask_temporal=is_mask,
                )
            else:
                log["samples"] = self.decode(
                    torch.randn_like(posterior.sample()), mask_temporal=is_mask
                )
            xrec = xrec.cpu().detach()
            log["reconstructions"] = xrec

        x = x.cpu().detach()
        log["inputs"] = x
        return log

    @torch.no_grad()
    def log_video(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.video_key)
        x = x.to(self.device, dtype=next(self.ref_conv_in.parameters()).dtype)

        if self.caption_guide:
            text_embeddings, text_attn_mask = self.get_text_embeddings(batch["caption"])
            text_embeddings = text_embeddings.to(device=self.device, dtype=x.dtype)
            text_attn_mask = text_attn_mask.to(device=self.device, dtype=x.dtype)

        if not only_inputs:
            if self.caption_guide:
                xrec, posterior, _  = self(x, text_embeddings, text_attn_mask)
            else:
                xrec, posterior, _  = self(x)

            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            # if self.caption_guide:
            #     log["samples"] = self.decode(
            #         torch.randn_like(posterior.sample()),
            #         text_embeddings,
            #         text_attn_mask,
            #     )
            # else:
            #     log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            xrec = xrec.cpu().detach()
            log["reconstructions"] = xrec

        x = x.cpu().detach()
        log["inputs"] = x
        return log

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        # if self.img_video_joint_train:
        #     return self.log_images_joint(batch, only_inputs=False, **kwargs)
        # else:
        return self.log_video(batch, only_inputs=False, **kwargs)
