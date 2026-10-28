"""ResNet and Swin-Unet encoders/decoders/bottleneck that plug into the UNet pipeline.

ResNet variants are drop-in replacements for BaselineEncoder/BaselineDecoder: ``skips[i]`` has ``hidden_features[i]`` channels at
1/2**i resolution and the encoder output has ``hidden_features[-1]`` channels at 1/2**N resolution.

The Swin variants follow Swin-Unet (Cao et al., 2021): 4x4 patch embedding, Swin stages with patch merging, a Swin bottleneck,
and a symmetric decoder with patch expanding, skip fusion and a final 4x expansion. In both cases the decoder returns
``hidden_features[0]`` channels at full resolution, so the output heads and fuel/auxiliary conditioning are unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoders import DecoderBase
from src.models.encoders import EncoderBase, append_coord_channels


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ResNetEncoder(EncoderBase):
    """ResNet-style encoder. Stage 0 keeps full resolution; every later stage halves it with a strided residual block."""

    def __init__(self, in_channels: int, hidden_features: list[int] | None, use_coordconv: bool = False, blocks_per_stage: int = 2):
        super().__init__()
        if hidden_features is None:
            raise ValueError("Hidden features cannot be None")
        self.in_channels = in_channels
        self.hidden_features = hidden_features
        self.use_coordconv = use_coordconv

        stem_in = in_channels + 2 if use_coordconv else in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, hidden_features[0], 3, padding=1, bias=False), nn.BatchNorm2d(hidden_features[0]), nn.ReLU(inplace=True)
        )
        # stage i produces skips[i]; the extra last stage produces the deepest feature map.
        stage_channels = list(hidden_features) + [hidden_features[-1]]
        self.stages = nn.ModuleList()
        prev = hidden_features[0]
        for i, ch in enumerate(stage_channels):
            stride = 1 if i == 0 else 2
            blocks = [ResidualBlock(prev, ch, stride)] + [ResidualBlock(ch, ch) for _ in range(blocks_per_stage - 1)]
            self.stages.append(nn.Sequential(*blocks))
            prev = ch

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.use_coordconv:
            x = append_coord_channels(x)
        x = self.stem(x)
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                skips.append(x)
        return x, skips


def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def _window_reverse(w: torch.Tensor, ws: int, B: int, H: int, W: int) -> torch.Tensor:
    C = w.shape[-1]
    x = w.view(B, H // ws, W // ws, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (window_size - 1)
        self.rel_index: torch.Tensor
        self.register_buffer("rel_index", rel[..., 0] * (2 * window_size - 1) + rel[..., 1], persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.rel_bias[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(Bw, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(Bw, N, C))


class SwinBlock(nn.Module):
    """(Shifted-)window transformer block operating on (B, H, W, C) tokens."""

    def __init__(self, dim: int, num_heads: int, window_size: int, shift: bool, mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window_size
        self.shift = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def _attn_mask(self, Hp: int, Wp: int, H: int, W: int, device: torch.device) -> torch.Tensor | None:
        ws, s = self.window_size, self.shift
        ids = torch.zeros(1, Hp, Wp, 1, device=device)
        if s:
            cnt = 0
            for h in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                for w in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                    ids[:, h, w, :] = cnt
                    cnt += 1
        win = _window_partition(ids, ws).squeeze(-1)
        blocked = win.unsqueeze(1) != win.unsqueeze(2)
        if Hp != H or Wp != W:
            valid = torch.zeros(1, Hp, Wp, 1, device=device)
            valid[:, :H, :W] = 1.0
            if s:
                valid = torch.roll(valid, (-s, -s), dims=(1, 2))
            blocked = blocked | (_window_partition(valid, ws).squeeze(-1).unsqueeze(1) == 0)
        return torch.zeros(blocked.shape, device=device).masked_fill(blocked, -100.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        ws = self.window_size
        shortcut = x
        x = self.norm1(x)
        pad_h, pad_w = (-H) % ws, (-W) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        mask = None
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
        if self.shift or pad_h or pad_w:
            mask = self._attn_mask(Hp, Wp, H, W, x.device)
        x = _window_reverse(self.attn(_window_partition(x, ws), mask), ws, B, Hp, Wp)
        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        x = shortcut + x[:, :H, :W, :]
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """2x2 patch merge: (B, H, W, C_in) -> (B, H/2, W/2, C_out)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_dim)
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        if H % 2 or W % 2:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x))


class SwinEncoder(EncoderBase):
    """Swin-Unet encoder (Cao et al., 2021): 4x4 patch embedding, then Swin stages joined by patch merging.

    Stage i has ``hidden_features[i]`` channels at 1/(4*2**i) resolution and produces ``skips[i]``. The last entry of
    ``hidden_features`` is the width of the Swin bottleneck, so the returned ``x`` has that width at 1/(4*2**(N-1)) resolution.
    """

    patch_size = 4

    def __init__(
        self,
        in_channels: int,
        hidden_features: list[int] | None,
        use_coordconv: bool = False,
        window_size: int = 8,
        depth: int = 2,
        head_dim: int = 32,
    ):
        super().__init__()
        if hidden_features is None:
            raise ValueError("Hidden features cannot be None")
        self.in_channels = in_channels
        self.hidden_features = hidden_features
        self.use_coordconv = use_coordconv

        stem_in = in_channels + 2 if use_coordconv else in_channels
        self.patch_embed = nn.Conv2d(stem_in, hidden_features[0], kernel_size=self.patch_size, stride=self.patch_size)
        self.embed_norm = nn.LayerNorm(hidden_features[0])
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        for i, dim in enumerate(hidden_features[:-1]):
            self.stages.append(_swin_stage(dim, window_size, depth, head_dim))
            self.merges.append(PatchMerging(dim, hidden_features[i + 1]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.use_coordconv:
            x = append_coord_channels(x)
        tokens = self.embed_norm(self.patch_embed(x).permute(0, 2, 3, 1))
        skips = []
        for stage, merge in zip(self.stages, self.merges, strict=True):
            tokens = stage(tokens)
            skips.append(tokens.permute(0, 3, 1, 2).contiguous())
            tokens = merge(tokens)
        return tokens.permute(0, 3, 1, 2).contiguous(), skips


def _swin_stage(dim: int, window_size: int, depth: int, head_dim: int) -> nn.Sequential:
    heads = max(1, dim // head_dim)
    while dim % heads:
        heads -= 1
    return nn.Sequential(*[SwinBlock(dim, heads, window_size, shift=(d % 2 == 1)) for d in range(depth)])


class SwinBottleneck(nn.Module):
    """Swin bottleneck with the same (x, x_tabular, x_wind) interface and auxiliary fusion as MultiSourceBottleneck."""

    def __init__(self, channels: int, aux_dims: dict[str, int], window_size: int = 8, depth: int = 2, head_dim: int = 32):
        super().__init__()
        self.blocks = _swin_stage(channels, window_size, depth, head_dim)
        self.fusion_projector: nn.Module | None = None
        if aux_dims:
            self.fusion_projector = nn.Sequential(
                nn.Conv2d(channels + sum(aux_dims.values()), channels, kernel_size=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
            )

    def forward(self, x: torch.Tensor, x_tabular: torch.Tensor | None = None, x_wind: torch.Tensor | None = None) -> torch.Tensor:
        x = self.blocks(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        if (x_tabular is None and x_wind is None) or self.fusion_projector is None:
            return x
        B, _, H, W = x.shape
        parts = [x]
        if x_tabular is not None:
            parts.append(x_tabular.view(B, -1, 1, 1).expand(-1, -1, H, W))
        if x_wind is not None:
            parts.append(x_wind if x_wind.shape[-2:] == (H, W) else F.interpolate(x_wind, size=(H, W), mode="bilinear", align_corners=True))
        return self.fusion_projector(torch.cat(parts, dim=1))


def _merge_skip(x: torch.Tensor, skip: torch.Tensor | None, use_coordconv: bool) -> torch.Tensor:
    if skip is not None:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
    if use_coordconv:
        x = append_coord_channels(x)
    return x


class _BackboneDecoder(DecoderBase):
    """Shared decoder plumbing: per level, upsample 2x, concatenate the skip, then fuse with a level-specific block."""

    def __init__(self, hidden_features: list[int] | None, use_skip_connections: bool, use_coordconv: bool):
        super().__init__()
        if hidden_features is None:
            raise ValueError("Hidden features cannot be None")
        self.hidden_features = list(hidden_features)
        self.use_skip_connections = use_skip_connections
        self.use_coordconv = use_coordconv
        self.out_channels = self.hidden_features[0]
        self.upsamples = nn.ModuleList()
        self.fuse = nn.ModuleList()

    def _fused_channels(self, ch: int) -> int:
        return (2 * ch if self.use_skip_connections else ch) + (2 if self.use_coordconv else 0)

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        skips = skip_connections[::-1]
        for i, (up, fuse) in enumerate(zip(self.upsamples, self.fuse, strict=True)):
            x = _merge_skip(up(x), skips[i] if self.use_skip_connections else None, self.use_coordconv)
            x = self._fuse(fuse, x)
        return x

    def _fuse(self, fuse: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return fuse(x)


class ResNetDecoder(_BackboneDecoder):
    """Residual decoder: bilinear upsample + conv, skip concat, then residual blocks."""

    def __init__(
        self, hidden_features: list[int] | None, use_skip_connections: bool = True, use_coordconv: bool = False, blocks_per_stage: int = 2
    ):
        super().__init__(hidden_features, use_skip_connections, use_coordconv)
        in_ch = self.hidden_features[-1] * 2  # bottleneck width
        for ch in reversed(self.hidden_features):
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            blocks = [ResidualBlock(self._fused_channels(ch), ch)] + [ResidualBlock(ch, ch) for _ in range(blocks_per_stage - 1)]
            self.fuse.append(nn.Sequential(*blocks))
            in_ch = ch


class _PatchExpand(nn.Module):
    """Swin-Unet patch expanding: linear to scale**2 * out channels, then pixel shuffle by ``scale``."""

    def __init__(self, in_dim: int, out_dim: int, scale: int = 2):
        super().__init__()
        self.expand = nn.Conv2d(in_dim, scale**2 * out_dim, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(scale)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shuffle(self.expand(x)).permute(0, 2, 3, 1)
        return self.norm(x).permute(0, 3, 1, 2).contiguous()


class _SwinFuse(nn.Module):
    """Linear fusion of the concatenated skip, followed by Swin blocks."""

    def __init__(self, in_dim: int, dim: int, window_size: int, depth: int, head_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, dim)
        self.blocks = _swin_stage(dim, window_size, depth, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.linear(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2).contiguous()


class SwinDecoder(DecoderBase):
    """Swin-Unet decoder: per level a 2x patch expanding, skip concat + linear fusion and Swin blocks, then a final 4x expansion.

    Takes the Swin bottleneck output (``hidden_features[-1]`` channels) and returns ``hidden_features[0]`` channels at full resolution.
    """

    def __init__(
        self,
        hidden_features: list[int] | None,
        use_skip_connections: bool = True,
        window_size: int = 8,
        depth: int = 2,
        head_dim: int = 32,
    ):
        super().__init__()
        if hidden_features is None:
            raise ValueError("Hidden features cannot be None")
        self.hidden_features = list(hidden_features)
        self.use_skip_connections = use_skip_connections
        self.out_channels = self.hidden_features[0]
        self.upsamples = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in range(len(self.hidden_features) - 2, -1, -1):
            dim = self.hidden_features[i]
            self.upsamples.append(_PatchExpand(self.hidden_features[i + 1], dim))
            self.fuse.append(_SwinFuse(2 * dim if use_skip_connections else dim, dim, window_size, depth, head_dim))
        self.final_expand = _PatchExpand(self.hidden_features[0], self.hidden_features[0], scale=SwinEncoder.patch_size)

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        skips = skip_connections[::-1]
        for i, (up, fuse) in enumerate(zip(self.upsamples, self.fuse, strict=True)):
            x = up(x)
            if self.use_skip_connections:
                x = torch.cat([skips[i], x], dim=1)
            x = fuse(x)
        return self.final_expand(x)
