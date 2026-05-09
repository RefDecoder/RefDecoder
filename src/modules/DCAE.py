import torch
import torch.nn as nn

from typing import Optional
from typing import Tuple, Dict


def DCAEDownsampleBlock(
        block_type: str,
        in_channels: int,
        out_channels: int,
        shortcut: Optional[str],
        factor: int = 2,
        kernel_size: int = 3
        ) -> nn.Module:
    """
    Factory function to create a downsampling block for spatial compression in video processing.

    This function constructs a downsampling module that reduces spatial dimensions (H, W)
    while preserving the temporal dimension (T). It supports different downsampling strategies
    and optional residual shortcut connections for better gradient flow.

    Args:
        block_type: Type of downsampling operation to use.
                   - "Conv": Standard strided convolution downsampling
                   - "ConvPixelUnshuffle": Convolution followed by pixel unshuffle (channel expansion)
        in_channels: Number of input channels
        out_channels: Number of output channels
        shortcut: Optional residual connection type for skip connections.
                 - None: No shortcut connection
                 - "averaging": Use pixel unshuffle with channel averaging as shortcut
        factor: Spatial downsampling factor (default=2).
               factor=2 reduces H,W to H/2,W/2
        kernel_size: Convolution kernel size (default=3, should be odd)

    Returns:
        nn.Module: A downsampling block module, optionally wrapped in ResidualBlock if
                  shortcut is specified

    Raises:
        ValueError: If block_type or shortcut is not supported
    """
    
    params = get_sampling_params(factor=factor, kernel_size=kernel_size, operation="downsample")

    if block_type == "Conv":
        block = DCAEConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=params["kernel_size"],
            stride=params["stride"],  # stride = factor
            use_bias=True,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=params["kernel_size"],
            factor=params["factor"]
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, 
            out_channels=out_channels, 
            factor=params["factor"]
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block

def DCAEUpsampleBlock(
        block_type: str,
        in_channels: int,
        out_channels: int,
        shortcut: Optional[str],
        factor: int = 2,  # NEW: expansion factor
        kernel_size: int = 3  # NEW: configurable kernel size
        ) -> nn.Module:
    """
    Factory function to create an upsampling block for spatial expansion in video processing.

    This function constructs an upsampling module that increases spatial dimensions (H, W)
    while preserving the temporal dimension (T). It supports different upsampling strategies
    and optional residual shortcut connections for better gradient flow.

    Args:
        block_type: Type of upsampling operation to use.
                   - "ConvPixelShuffle": Convolution with pixel shuffle (sub-pixel convolution)
                   - "InterpolateConv": Interpolation-based upsampling followed by convolution
        in_channels: Number of input channels
        out_channels: Number of output channels
        shortcut: Optional residual connection type for skip connections.
                 - None: No shortcut connection
                 - "duplicating": Use channel duplication with pixel shuffle as shortcut
        factor: Spatial upsampling factor (default=2).
               factor=2 expands H,W to H*2,W*2
        kernel_size: Convolution kernel size (default=3, should be odd)

    Returns:
        nn.Module: An upsampling block module, optionally wrapped in ResidualBlock if
                  shortcut is specified

    Raises:
        ValueError: If block_type or shortcut is not supported
    """
    
    params = get_sampling_params(factor=factor, kernel_size=kernel_size, operation="upsample")

    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=params["kernel_size"],
            factor=params["factor"]
        )
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=params["kernel_size"],
            factor=params["factor"]
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, 
            out_channels=out_channels, 
            factor=params["factor"]
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


#################################################################################
#                             Basic Layers                                      #
#################################################################################

def get_sampling_params(
    factor: int = 2,
    kernel_size: int = 3,
    operation: str = "downsample"  # or "upsample"
) -> Dict[str, int]:
    """
    Calculate parameters for spatial sampling operations.
    
    Args:
        factor: Compression/expansion factor for spatial dimensions (H, W)
                factor=2 means H,W -> H/2,W/2 (downsample) or H,W -> H*2,W*2 (upsample)
        kernel_size: Size of convolution kernel (should be odd, typically 3, 5, or 7)
        operation: "downsample" or "upsample"
    
    Returns:
        Dictionary with 'kernel_size', 'stride', 'padding', 'factor'
    """
    assert kernel_size % 2 == 1, "kernel_size should be odd number for symmetric padding"
    assert factor >= 1, "factor should be >= 1"
    
    # Calculate padding for "same" convolution
    padding = kernel_size // 2
    
    if operation == "downsample":
        stride = factor  # stride = factor for direct downsampling
    elif operation == "upsample":
        stride = 1  # upsampling typically uses stride=1 with pixel shuffle or interpolation
    else:
        raise ValueError(f"operation must be 'downsample' or 'upsample', got {operation}")
    
    return {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "factor": factor
    }

class DCAEConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
    ):
        super(DCAEConv3d, self).__init__()

        padding = self.get_same_padding(kernel_size) * dilation
        # Keep temporal dimension causal by avoiding padding along T.
        padding = (0, padding, padding)

        self.dropout = nn.Dropout3d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        return x
    
    def get_same_padding(self, kernel_size: int) -> int:
            assert kernel_size % 2 > 0, "kernel size should be odd number"
            return kernel_size // 2
    
class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = DCAEConv3d(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = pixel_unshuffle_3d(x, self.factor)
        return x

class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pixel_unshuffle_3d(x, self.factor)
        B, C, T, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, T, H, W)
        x = x.mean(dim=2) 
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
    ):
        super(ResidualBlock, self).__init__()

        self.main = main
        self.shortcut = shortcut

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
        return res

class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = DCAEConv3d(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = pixel_shuffle_3d(x, self.factor)
        return x

class InterpolateConvUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = DCAEConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        
        # Reshape to (B*T, C, H, W) to apply 2D interpolation on spatial dims only
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Apply 2D spatial interpolation
        x = torch.nn.functional.interpolate(
            x, scale_factor=self.factor, mode=self.mode
        )
        
        # Reshape back to (B, C, T, H*factor, W*factor)
        x = x.view(B, T, C, H * self.factor, W * self.factor)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H*factor, W*factor)
        
        x = self.conv(x)
        return x

class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = pixel_shuffle_3d(x, self.factor)
        return x
    
def pixel_unshuffle_3d(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Spatial pixel unshuffle for 3D tensors (keeps temporal dimension unchanged)

    Args:
        x: Input tensor of shape (B, C, T, H, W)
        factor: Downsampling factor

    Returns:
        Output tensor of shape (B, C * factor^2, T, H // factor, W // factor)
    """
    b, c, t, h, w = x.shape

    # Reshape to group spatial pixels
    # (B, C, T, H, W) -> (B, C, T, H//factor, factor, W//factor, factor)
    x = x.view(b, c, t, h // factor, factor, w // factor, factor)

    # Rearrange to move spatial blocks into channels
    # (B, C, T, H//factor, factor, W//factor, factor) -> (B, C, T, H//factor, W//factor, factor, factor)
    x = x.permute(0, 1, 2, 3, 5, 4, 6).contiguous()

    # Flatten spatial blocks into channels
    # (B, C, T, H//factor, W//factor, factor, factor) -> (B, C*factor^2, T, H//factor, W//factor)
    x = x.view(b, c * factor * factor, t, h // factor, w // factor)

    return x


def pixel_shuffle_3d(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Spatial pixel shuffle for 3D tensors (keeps temporal dimension unchanged)

    Args:
        x: Input tensor of shape (B, C * factor^2, T, H, W)
        factor: Upsampling factor

    Returns:
        Output tensor of shape (B, C, T, H * factor, W * factor)
    """
    b, c_total, t, h, w = x.shape
    c = c_total // (factor * factor)

    # Reshape channels into spatial blocks
    # (B, C*factor^2, T, H, W) -> (B, C, factor, factor, T, H, W)
    x = x.view(b, c, factor, factor, t, h, w)

    # Rearrange to interleave spatial blocks
    # (B, C, factor, factor, T, H, W) -> (B, C, T, H, factor, W, factor)
    x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()

    # Merge spatial blocks
    # (B, C, T, H, factor, W, factor) -> (B, C, T, H*factor, W*factor)
    x = x.view(b, c, t, h * factor, w * factor)

    return x