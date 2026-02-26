import torch
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.basic_var import FFN, AdaLNBeforeHead
from models.embed_rope_faceray import apply_rotary_emb
from models.helpers import DropPath


__all__ = ["FFN", "Attention", "AttnBlock", "AdaLNBeforeHead"]


dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = (
    flash_attn_func
) = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError:
    pass
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    pass
try:
    from flash_attn import flash_attn_func
except ImportError:
    pass


def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
    attn = query.mul(scale) @ key.transpose(-2, -1)
    if attn_mask is not None:
        attn.add_(attn_mask)
    if dropout_p > 0:
        attn = F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
    else:
        attn = attn.softmax(dim=-1)
    return attn @ value


class Attention(nn.Module):
    def __init__(
        self,
        block_idx,
        embed_dim=768,
        is_cross=False,
        in_dim_cross=77,
        num_heads=12,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_l2_norm=False,
        flash_if_available=True,
        rotary_pos_emb=False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.attn_l2_norm = attn_l2_norm
        self.is_cross = is_cross
        self.rotary_pos_emb = rotary_pos_emb
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                requires_grad=True,
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        if is_cross:
            self.mat_kv = nn.Linear(in_dim_cross, embed_dim * 2, bias=False)
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=False)
        else:
            self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer("zero_k_bias", torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = (
            flash_if_available
            and memory_efficient_attention is not None
            and not self.is_cross
        )
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(
        self, x, attn_bias=None, encoder_hidden_states=None, freqs_cis=None, layer_id=0
    ):
        B, L, C = x.shape
        if freqs_cis is not None:
            freqs_cis = freqs_cis.to(x.device)

        if not self.is_cross:
            qkv = F.linear(
                input=x,
                weight=self.mat_qkv.weight,
                bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
            ).view(B, L, 3, self.num_heads, self.head_dim)
            using_flash = (
                self.using_flash and attn_bias is None and qkv.dtype != torch.float32
            )
            using_xform = self.using_xform and attn_bias is None
            if using_flash or using_xform:
                q, k, v = qkv.unbind(dim=2)
                if self.rotary_pos_emb and freqs_cis is not None:
                    q, k = apply_rotary_emb(
                        q.transpose(1, 2), k.transpose(1, 2), freqs_cis=freqs_cis
                    )
                    q = q.transpose(1, 2)
                    k = k.transpose(1, 2)
            else:
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
                if self.rotary_pos_emb and freqs_cis is not None:
                    q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        else:
            assert encoder_hidden_states is not None
            q = (
                F.linear(input=x, weight=self.mat_q.weight, bias=self.q_bias)
                .view(B, L, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            kv = F.linear(
                input=encoder_hidden_states,
                weight=self.mat_kv.weight,
                bias=torch.cat((self.zero_k_bias, self.v_bias)),
            ).view(B, -1, 2, self.num_heads, self.head_dim)
            using_flash = (
                self.using_flash and attn_bias is None and kv.dtype != torch.float32
            )
            using_xform = self.using_xform and attn_bias is None
            if using_flash or using_xform:
                k, v = kv.unbind(dim=2)
                q = q.permute(0, 2, 1, 3)
            else:
                k, v = kv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        if self.attn_l2_norm:
            scale_mul = torch.exp(self.scale_mul_1H11).clamp(max=self.max_scale_mul)
            if using_flash or using_xform:
                scale_mul = scale_mul.transpose(1, 2)
            q = F.normalize(q, dim=-1).mul(scale_mul).to(v.dtype)
            k = F.normalize(k, dim=-1).to(v.dtype)

        if using_flash:
            attn_out = flash_attn_func(q, k, v, dropout_p=self.attn_drop)
            self.last_attn_impl = "flash"
        elif using_xform:
            if not self.is_cross:
                bias = (
                    None
                    if attn_bias is None
                    else attn_bias.to(dtype=q.dtype).expand(B, self.num_heads, -1, -1)
                )
                attn_out = memory_efficient_attention(
                    q, k, v, attn_bias=bias, p=self.attn_drop
                )
                self.last_attn_impl = "xform"
            else:
                bias = (
                    attn_bias.expand(B, self.num_heads, q.shape[1], -1)
                    if attn_bias is not None
                    else None
                )
                attn_out = memory_efficient_attention(
                    q, k, v, attn_bias=bias, p=self.attn_drop
                )
                self.last_attn_impl = "xform"
        else:
            attn_out = slow_attn(
                q, k, v, scale=self.scale, attn_mask=attn_bias, dropout_p=self.attn_drop
            )
            self.last_attn_impl = "sdpa"

        self.last_attn_bias = attn_bias is not None

        if using_flash or self.using_xform:
            attn_out = attn_out.view(B, L, self.num_heads * self.head_dim)
        else:
            attn_out = attn_out.transpose(1, 2).reshape(B, L, C)

        return self.proj_drop(self.proj(attn_out))


class AttnBlock(nn.Module):
    def __init__(
        self,
        block_idx,
        last_drop_p,
        embed_dim,
        cond_dim,
        shared_aln: bool,
        norm_layer,
        num_heads,
        in_dim_cross=1024,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        attn_l2_norm=False,
        enable_cross=True,
        cross_attn_ln=True,
        flash_if_available=False,
        fused_if_available=True,
        rotary_pos_emb=True,
        faceray_adapter=None,
        use_checkpoint=True,
    ):
        super().__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = Attention(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_l2_norm=attn_l2_norm,
            flash_if_available=flash_if_available,
            rotary_pos_emb=rotary_pos_emb,
        )
        if enable_cross:
            self.cross_attn = Attention(
                block_idx=block_idx,
                embed_dim=embed_dim,
                is_cross=True,
                in_dim_cross=in_dim_cross,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available,
            )
        else:
            self.cross_attn = None
        self.enable_cross = enable_cross
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=round(embed_dim * mlp_ratio),
            drop=drop,
            fused_if_available=fused_if_available,
        )

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(
                torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5
            )
        self.cross_attn_ln = cross_attn_ln

        self.fused_add_norm_fn = None
        self.faceray_adapter = faceray_adapter
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        x,
        cond_BD,
        attn_bias,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        freqs_cis=None,
        layer_id=0,
        face_group=1,
        log_adapter_delta: bool = False,
    ):
        if freqs_cis is not None:
            freqs_cis = freqs_cis.to(x.device)
        if self.shared_aln and cond_BD is not None:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_gss + cond_BD
            ).unbind(2)
        else:
            gamma1 = 1
            gamma2 = 1
            scale1 = 0
            scale2 = 0
            shift1 = 0
            shift2 = 0

        x_in = self.ln_wo_grad(x).mul(scale1 + 1).add_(shift1)

        def _self_attn_forward(x_input):
            return self.attn(
                x_input,
                attn_bias=attn_bias,
                freqs_cis=freqs_cis,
                layer_id=layer_id,
            ).mul_(gamma1)

        if self.use_checkpoint and x_in.requires_grad:
            sa_out = checkpoint(_self_attn_forward, x_in, use_reentrant=False)
        else:
            sa_out = _self_attn_forward(x_in)

        x = x + self.drop_path(sa_out)

        if self.faceray_adapter is not None:
            delta = self.faceray_adapter(x_in, face_group=face_group)
            if log_adapter_delta:
                delta_mean_abs = delta.detach().abs().mean().item()
                print(
                    f"[FaceRay] block={layer_id} face_group={face_group} delta_mean_abs={delta_mean_abs:.6f}"
                )
            x = x + delta

        def _cross_ffn_forward(x_input):
            x_out = x_input
            if self.cross_attn is not None and encoder_hidden_states is not None:
                if self.cross_attn_ln:
                    out = self.cross_attn(
                        self.ln_wo_grad(x_out),
                        encoder_hidden_states=encoder_hidden_states,
                        attn_bias=encoder_attention_mask,
                        layer_id=layer_id,
                    )
                else:
                    out = self.cross_attn(
                        x_out,
                        encoder_hidden_states=encoder_hidden_states,
                        attn_bias=encoder_attention_mask,
                    )

                out = out[0] if isinstance(out, (tuple, list)) else out
                x_out = out + x_out

            x_out = x_out + self.drop_path(
                self.ffn(self.ln_wo_grad(x_out).mul(scale2 + 1).add_(shift2)).mul(
                    gamma2
                )
            )
            return x_out

        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(_cross_ffn_forward, x, use_reentrant=False)
        else:
            x = _cross_ffn_forward(x)
        return x

    def extra_repr(self) -> str:
        return f"shared_aln={self.shared_aln}"
