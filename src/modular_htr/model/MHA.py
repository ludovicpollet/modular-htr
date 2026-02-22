import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_rel_features(rel: torch.Tensor, dim: int) -> torch.Tensor:
    """
    rel: [...], integer relative distances (e.g. j-i), can be negative.
    returns: [..., dim] sinusoidal features
    """
    device = rel.device
    half = dim // 2

    rel = rel.to(torch.float32).unsqueeze(-1)  # [..., 1]
    div = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / half)
    )  # [half]

    angles = rel * div  # [..., half]
    feat = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [..., 2*half]
    if dim % 2 == 1:
        feat = torch.nn.functional.pad(feat, (0, 1))
    return feat  # [..., dim]


class RelSinusoidalPositionBias(nn.Module):
    """
    Bias-only relative positional encoding using sinusoidal features.
    Produces bias [1, H, T, T] to add to attention logits.
    """

    def __init__(self, num_heads: int, max_rel: int = 256, rel_dim: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.max_rel = max_rel
        self.rel_dim = rel_dim
        self.proj = nn.Linear(rel_dim, num_heads, bias=False)

    def forward(self, T: int, device=None) -> torch.Tensor:
        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        rel = (j - i).clamp(-self.max_rel, self.max_rel)  # [T,T]

        feat = sinusoidal_rel_features(rel, self.rel_dim)  # [T,T,rel_dim]
        bias = self.proj(feat)  # [T,T,H]
        return bias.permute(2, 0, 1).unsqueeze(0)  # [1,H,T,T]


class MHAWithRelSinBias(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_rel: int = 256,
        rel_dim: int = 32,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        self.rel_bias = RelSinusoidalPositionBias(
            n_heads, max_rel=max_rel, rel_dim=rel_dim
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: [B,T,D]
        key_padding_mask: [B,T] (True=PAD)
        """
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,dh]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,T,T]
        logits = logits + self.rel_bias(T, device=x.device)  # add bias

        # AMP-friendly masking + fp32 softmax
        NEG = -1e4
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], NEG)

        attn = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        attn = self.drop(attn)

        y = attn @ v  # [B,H,T,dh]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


class RelBiasEncoderLayer(nn.Module):
    """
    Mirrors nn.TransformerEncoderLayer with norm_first=True and GELU FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_rel: int,
        rel_dim: int,
    ):
        super().__init__()
        self.self_attn = MHAWithRelSinBias(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            max_rel=max_rel,
            rel_dim=rel_dim,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)
        self.lin1 = nn.Linear(d_model, dim_feedforward)
        self.lin2 = nn.Linear(dim_feedforward, d_model)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # norm_first=True style
        x = x + self.drop(
            self.self_attn(self.norm1(x), key_padding_mask=src_key_padding_mask)
        )
        ff = self.lin2(self.drop(F.gelu(self.lin1(self.norm2(x)))))
        x = x + self.drop(ff)
        return x


class RelBiasTransformerStack(nn.Module):
    """
    A transformer stack with relative sinusoidal positional encoding (bias only),
    to try and match the architecture proposed by Diaz et al. in the
    "Rethinking Text Line Recognition Models" (2021) paper.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        max_rel: int,
        rel_dim: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelBiasEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, max_rel, rel_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)
