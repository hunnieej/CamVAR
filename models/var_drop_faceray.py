import math
import os
import random
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dist
from models.basic_var_faceray import AttnBlock, Attention
from models.embed_rope import compute_axial_cis
from models.quant import VectorQuantizer2
from models.vqvae import VQVAE
from models.face_ray_attention import FaceRayAttentionAdapter
from models.cubemap_geometry import CubemapGeometry
from utils.mask_utils import Scheduler
from models.helpers import sample_with_top_k_top_p_


def prepare_attn_mask(encoder_attention_mask):
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = torch.where(encoder_attention_mask == 1, 0, -torch.inf)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(1)
    return encoder_attention_mask


class VAR(nn.Module):
    def __init__(
        self,
        vae_local: VQVAE,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        enable_cross=True,
        in_dim_cross=1024,
        flash_if_available=True,
        fused_if_available=True,
        rotary_pos_emb=True,
        rope_norm=32,
        absolute_lvl_emb=True,
        drop_scale_length=None,
        enable_logit_norm=True,
        enable_adaptive_norm=True,
        train_mode="head_only",
        rope_theta=10000.0,
        sample_from_idx=9,
        noise_sampling=False,
    ):
        super().__init__()
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_local: VQVAE = vae_local
        self.quantize_local: VectorQuantizer2 = vae_local.quantize
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (self.quantize_local,)
        self.num_classes = num_classes
        self.depth = depth
        self.C = embed_dim
        self.D = embed_dim
        self.Cvae = self.vae_local.Cvae
        self.V = self.vae_local.vocab_size
        self.num_heads = num_heads
        self.attn_l2_norm = attn_l2_norm
        self.shared_aln = shared_aln
        self.cond_drop_rate = cond_drop_rate
        self.patch_nums = tuple(patch_nums)
        self.num_stages = len(self.patch_nums)
        self.num_stages_minus_1 = self.num_stages - 1
        self.first_l = 1
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.noise_sampling = noise_sampling

        self.lvl_1L = []
        self.begin_ends = []
        cur = 0
        for pn in self.patch_nums:
            self.lvl_1L += [len(self.begin_ends)] * (pn * pn)
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn
        self.lvl_1L = torch.tensor(self.lvl_1L, dtype=torch.long).view(1, -1)
        self.L = cur

        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, embed_dim))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=0.02)

        self.rotary_pos_emb = rotary_pos_emb
        self.absolute_lvl_emb = absolute_lvl_emb
        if self.rotary_pos_emb:
            self.compute_cis = partial(
                compute_axial_cis,
                dim=embed_dim // num_heads,
                theta=rope_theta,
                normalize=rope_norm,
            )
            freqs_cis = []
            for pn in self.patch_nums:
                freqs_cis.append(self.compute_cis(end_x=pn, end_y=pn))
            freqs_cis = torch.cat(freqs_cis, dim=0)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        else:
            self.freqs_cis = None
            self.pos_1LC = nn.Parameter(torch.empty(1, self.L, embed_dim))
            nn.init.trunc_normal_(self.pos_1LC.data, mean=0, std=0.02)

        if self.absolute_lvl_emb:
            self.lvl_embed = nn.Embedding(self.num_stages, embed_dim)
            nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=0.02)

        dpr = torch.linspace(0, drop_path_rate, depth)

        self.blocks = nn.ModuleList([])
        self.faceray_precompute = None
        self.faceray_adapters = nn.ModuleList([])
        self.faceray_sampler_adapters = nn.ModuleList([])

        self.enable_cross = enable_cross

        self._init_faceray_geometry()
        for block_idx in range(depth):
            adapter = FaceRayAttentionAdapter(
                embed_dim=self.C,
                adapter_dim=self.C // 8,
                num_heads=8,
                ray_subdim=24,
                precompute=self.faceray_precompute,
                seq_offset=self.begin_ends[-1][0],
            )
            self.faceray_adapters.append(adapter)
            self.blocks.append(
                AttnBlock(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    in_dim_cross=in_dim_cross,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    enable_cross=enable_cross,
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                    rotary_pos_emb=rotary_pos_emb,
                    faceray_adapter=adapter,
                    use_checkpoint=True,
                )
            )

        self.attn_bias_for_masking = self._build_causal_mask(self.L)

        self.word_embed = nn.Linear(self.Cvae, self.C)
        self.head_logits = nn.Linear(self.C, self.V)
        self.encoder_proj = nn.Linear(in_dim_cross, embed_dim)
        self.enable_logit_norm = enable_logit_norm
        self.enable_adaptive_norm = enable_adaptive_norm
        if self.enable_logit_norm:
            self.logit_norm = nn.LayerNorm(
                embed_dim, elementwise_affine=False, eps=norm_eps
            )

        if self.enable_adaptive_norm:
            self.word_embed_head = nn.Linear(self.Cvae, self.C)
            encoder_depth = 3
            for block_idx in range(encoder_depth):
                adapter = FaceRayAttentionAdapter(
                    embed_dim=self.C,
                    adapter_dim=self.C // 8,
                    num_heads=8,
                    ray_subdim=24,
                    precompute=self.faceray_precompute,
                    seq_offset=self.begin_ends[-1][0],
                )
                self.faceray_sampler_adapters.append(adapter)
            self.feat_extract_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        cond_dim=self.D,
                        shared_aln=shared_aln,
                        in_dim_cross=in_dim_cross,
                        block_idx=block_idx,
                        embed_dim=self.C,
                        norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[block_idx],
                        last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                        enable_cross=False,
                        attn_l2_norm=attn_l2_norm,
                        flash_if_available=flash_if_available,
                        fused_if_available=fused_if_available,
                        rotary_pos_emb=rotary_pos_emb,
                        faceray_adapter=self.faceray_sampler_adapters[block_idx],
                        use_checkpoint=True,
                    )
                    for block_idx in range(encoder_depth)
                ]
            )
            self.from_idx = sample_from_idx
            self.bg_last, _ = self.begin_ends[self.from_idx]
            _, self.ed_last = self.begin_ends[-1]
            length_ = self.ed_last - self.bg_last
            self.attn_mask = self._build_stage_mask(self.bg_last, self.ed_last)
            self.pos_start_last = nn.Parameter(torch.empty(1, self.first_l, embed_dim))
            nn.init.trunc_normal_(self.pos_start_last.data, mean=0, std=0.02)
            self.logit_norm = nn.LayerNorm(
                embed_dim, elementwise_affine=False, eps=norm_eps
            )
            self.mask_scheduler = Scheduler()
            self.head_logits2 = nn.Linear(self.C, self.V)
            self.encoder_proj2 = nn.Linear(in_dim_cross, embed_dim)
            self.head_proj = nn.Linear(2 * embed_dim, embed_dim)
            self.lvl_embed_2 = nn.Embedding(len(self.patch_nums), self.C)
            nn.init.trunc_normal_(self.lvl_embed_2.weight.data, mean=0, std=0.02)
        else:
            self.from_idx = math.inf

        self.train_mode = train_mode

    def _init_faceray_geometry(self):
        geom = CubemapGeometry(self.patch_nums, self.begin_ends, strip_w=2, k=2)
        self.faceray_precompute = geom.precompute()
        self.faceray_geom = geom

    @staticmethod
    def _build_causal_mask(length: int) -> torch.Tensor:
        d = torch.arange(length, device=dist.get_device()).view(1, -1)
        dT = d.transpose(0, 1)
        return (
            torch.where(d >= dT, 0.0, -torch.inf)
            .reshape(1, 1, length, length)
            .contiguous()
            .to(dist.get_device())
        )

    def _build_stage_mask(self, bg: int, ed: int) -> torch.Tensor:
        length_ = ed - bg
        return torch.zeros((1, 1, length_ + 1, length_ + 1), device=dist.get_device())

    def forward(
        self,
        x_BLCv_wo_first_l: torch.Tensor,
        encoder_hidden_states=None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_pool_feat=None,
        face_group: int = 1,
    ):
        B = x_BLCv_wo_first_l.shape[0]

        if encoder_attention_mask is not None:
            encoder_attention_mask = prepare_attn_mask(encoder_attention_mask)

        if not self.rotary_pos_emb:
            lvl_pos = self.pos_1LC
        else:
            lvl_pos = 0

        if self.absolute_lvl_emb:
            lvl_pos = lvl_pos + self.lvl_embed(self.lvl_1L.to(x_BLCv_wo_first_l.device))
            cond_lvl_emb = None
        else:
            cond_lvl_emb = None

        if not self.rotary_pos_emb or self.absolute_lvl_emb:
            x_BLC = (
                self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, : self.first_l]
            )
        else:
            x_BLC = self.pos_start.expand(B, self.first_l, -1)

        if x_BLCv_wo_first_l is not None:
            x_BLC = torch.cat(
                [x_BLC, self.word_embed(x_BLCv_wo_first_l.float())], dim=1
            )

        attn_bias = self.attn_bias_for_masking.to(x_BLC.device)
        freqs_cis = (
            self.freqs_cis.to(x_BLC.device) if self.freqs_cis is not None else None
        )

        for i, b in enumerate(self.blocks):
            x_BLC = b(
                x=x_BLC,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                cond_BD=cond_lvl_emb,
                attn_bias=attn_bias,
                freqs_cis=freqs_cis,
                layer_id=i,
                face_group=face_group,
                log_adapter_delta=False,
            )

        logits_BLV = self.get_logits(x_BLC.float())
        return logits_BLV, x_BLC, None

    def forward_sampler(
        self,
        x_BLCv_wo_first_l: torch.Tensor,
        encoder_hidden_states=None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_pool_feat=None,
        embed_Cvae=None,
        face_group: int = 6,
    ):
        with torch.no_grad():
            logits_BLV, feat_BlC, _ = self.forward(
                x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_pool_feat=encoder_pool_feat,
                face_group=face_group,
            )

        b6 = feat_BlC.shape[0]
        b = b6 // face_group
        bg, ed = self.bg_last, self.ed_last
        last_len = ed - bg

        logits_last = logits_BLV.view(b, face_group, -1, self.V)[:, :, bg:ed].reshape(
            b, face_group * last_len, self.V
        )
        feat_last = feat_BlC.view(b, face_group, -1, self.C)[:, :, bg:ed].reshape(
            b, face_group * last_len, self.C
        )

        mask = self.mask_scheduler.add_mask_for_training(embed_Cvae[..., 0])
        embed_Cvae = embed_Cvae * mask[..., None]

        encoder_pool_feat = encoder_pool_feat.view(b, face_group, -1)[:, 0]
        text_pool_feat = (
            self.encoder_proj2(encoder_pool_feat.unsqueeze(1)) + self.pos_start_last
        )

        freqs_cis = None
        if self.freqs_cis is not None:
            freqs_last = self.freqs_cis[bg:ed].to(feat_last.device)
            freqs_last = freqs_last.repeat(face_group, 1)
            freqs_cis = torch.cat(
                [self.freqs_cis[0:1].to(feat_last.device), freqs_last], dim=0
            )

        lvl_last = self.lvl_1L[:, bg:ed].repeat(1, face_group).to(feat_last.device)
        sampler_attn = self._build_sampler_attn(face_group, last_len)

        logits_BLV_ = self.sample(
            feature=torch.cat(
                [
                    text_pool_feat,
                    self.word_embed_head(embed_Cvae) + self.lvl_embed_2(lvl_last),
                ],
                dim=1,
            ),
            prev_emb=torch.cat([text_pool_feat, feat_last], dim=1),
            attn_bias=sampler_attn,
            freqs_cis=freqs_cis,
            face_group=face_group,
        )
        return logits_BLV_[:, 1:], mask

    def autoregressive_infer_cubemap(
        self,
        encoder_hidden_states,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_pool_feat=None,
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        w_mask=False,
        sample_version="new",
        face_group: int = 6,
        face_order=(0, 1, 2, 3, 4, 5),
        log_face_group: bool = True,
        log_adapter_delta: bool = True,
    ) -> torch.Tensor:
        """Cubemap-joint inference entrypoint.

        Expects per-scene text embeddings and generates 6 faces jointly.
        Output order is enforced to [F, R, B, L, U, D] by default.
        """

        if face_group != 6:
            raise ValueError("autoregressive_infer_cubemap expects face_group=6")

        B_scene = encoder_hidden_states.shape[0]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(
            face_group, dim=0
        )
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.repeat_interleave(
                face_group, dim=0
            )
        if encoder_pool_feat is not None:
            encoder_pool_feat = encoder_pool_feat.repeat_interleave(face_group, dim=0)

        imgs = self.autoregressive_infer_cfg(
            B=B_scene * face_group,
            label_B=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_pool_feat=encoder_pool_feat,
            g_seed=g_seed,
            cfg=cfg,
            top_k=top_k,
            top_p=top_p,
            more_smooth=more_smooth,
            w_mask=w_mask,
            sample_version=sample_version,
            face_group=face_group,
            log_face_group=log_face_group,
            log_adapter_delta=log_adapter_delta,
        )

        b, c, h, w = imgs.shape
        imgs = imgs.view(B_scene, face_group, c, h, w)
        imgs = imgs[:, list(face_order), ...]
        return imgs.reshape(B_scene * face_group, c, h, w)

    def autoregressive_infer_cfg(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        encoder_hidden_states=None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_pool_feat=None,
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        w_mask=False,
        sample_version="new",
        face_group: int = 1,
        log_face_group: bool = False,
        log_adapter_delta: bool = False,
    ) -> torch.Tensor:
        encoder_attention_mask = prepare_attn_mask(
            encoder_attention_mask=encoder_attention_mask
        )

        if g_seed is None:
            rng = None
        else:
            rng = torch.Generator(device=encoder_pool_feat.device)
            rng.manual_seed(g_seed)

        if not self.noise_sampling:
            sos = self.encoder_proj(encoder_pool_feat)
        else:
            sos = torch.randn([2 * B, self.D], device=encoder_pool_feat.device)

        if not self.rotary_pos_emb:
            lvl_pos = self.pos_1LC
        else:
            lvl_pos = 0

        if self.absolute_lvl_emb:
            lvl_pos = lvl_pos + self.lvl_embed(self.lvl_1L.to(sos.device))
            cond_lvl_emb = None
        if self.shared_aln:
            cond_lvl_emb = self.lvl_embed_proj(
                torch.cat(
                    [
                        self.lvl_embed(self.lvl_1L),
                        self.lvl_embed(
                            torch.full(
                                self.lvl_1L.shape,
                                self.lvl_1L[0, -1],
                                device=self.lvl_1L.device,
                            )
                        ),
                    ],
                    dim=-1,
                )
            )
            cond_lvl_emb = self.lvl_embed_adaln(cond_lvl_emb)

        if (not self.rotary_pos_emb) or (self.absolute_lvl_emb):
            next_token_map = (
                sos.unsqueeze(1).expand(2 * B, self.first_l, -1)
                + self.pos_start.expand(2 * B, self.first_l, -1)
                + lvl_pos[:, : self.first_l]
            )
        else:
            next_token_map = sos.unsqueeze(1).expand(
                2 * B, self.first_l, -1
            ) + self.pos_start.expand(2 * B, self.first_l, -1)

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        if log_face_group:
            for bi, _ in enumerate(self.blocks):
                print(f"[FaceRay] block={bi} face_group={face_group}")

        for b in self.blocks:
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            t = cfg * ratio
            cur_L += pn * pn
            freqs_cis_cur = (
                self.freqs_cis[cur_L - pn * pn : cur_L, :]
                if self.freqs_cis is not None
                else None
            )
            cond_lvl_emb_cur = (
                cond_lvl_emb[:, cur_L - pn * pn : cur_L, ...]
                if cond_lvl_emb is not None
                else None
            )
            x = next_token_map
            for i, b in enumerate(self.blocks):
                x = b(
                    x=x,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    cond_BD=cond_lvl_emb_cur,
                    attn_bias=None,
                    freqs_cis=freqs_cis_cur,
                    layer_id=i,
                    face_group=face_group,
                    log_adapter_delta=log_adapter_delta and si == 0,
                )

            if w_mask and si >= self.from_idx:
                embed_Cvae = None
                h_BChw = torch.zeros((B, pn * pn, self.Cvae), device=x.device)
                mask = torch.zeros((B, pn * pn), device=x.device)

                if sample_version == "1024":
                    step_thresh = 5
                    si_thresh = 13
                    self.mask_scheduler.step = 15 if si > 12 else 8
                else:
                    step_thresh = 10000
                    si_thresh = -1
                    self.mask_scheduler.step = 8

                self.mask_scheduler._create_scheduler(patch_size=pn)
                for step in range(self.mask_scheduler.step):
                    _, t_maskratio = self.mask_scheduler.get_mask(step, x[..., 0])
                    if step < step_thresh and si < si_thresh:
                        logits_BlV = self.get_logits(x)
                        embed_Cvae, conf_Bl = self.from_logit2emb(
                            logits_BlV, t=t, rng=rng, top_k=top_k, top_p=top_p, B=B
                        )
                        tresh_conf, indice_mask = torch.topk(
                            conf_Bl.view(B, -1), k=t_maskratio, dim=-1
                        )
                        mask_0 = mask.clone().detach()
                        for i_mask, ind_mask in enumerate(indice_mask):
                            mask[i_mask, ind_mask] = 1
                        h_BChw += embed_Cvae * (mask[..., None] - mask_0[..., None])
                    else:
                        text_pool_feat = (
                            self.encoder_proj2(encoder_pool_feat.unsqueeze(1))
                            + self.pos_start_last
                        )
                        cur_feature = torch.cat(
                            [
                                text_pool_feat,
                                self.word_embed_head(h_BChw * mask[..., None]).repeat(
                                    2, 1, 1
                                )
                                + self.lvl_embed_2(
                                    self.lvl_1L[:, cur_L - pn * pn : cur_L]
                                ),
                            ],
                            dim=1,
                        )

                        logits_BlV = self.sample(
                            feature=cur_feature,
                            prev_emb=torch.cat([text_pool_feat, x], dim=1),
                            attn_bias=None,
                            freqs_cis=torch.cat(
                                [self.freqs_cis[0, None], freqs_cis_cur], dim=0
                            ),
                            face_group=face_group,
                        )
                        logits_BlV = logits_BlV[:, 1:]
                        embed_Cvae, conf_Bl = self.from_logit2emb(
                            logits_BlV, t=cfg, rng=rng, top_k=top_k, top_p=top_p, B=B
                        )
                    conf_Bl = torch.rand_like(conf_Bl) if step < 5 else conf_Bl
                    conf_Bl = conf_Bl * (1 - mask[..., None])
                    tresh_conf, indice_mask = torch.topk(
                        conf_Bl.view(B, -1), k=t_maskratio, dim=-1
                    )
                    mask_0 = mask.clone().detach()
                    for i_mask, ind_mask in enumerate(indice_mask):
                        mask[i_mask, ind_mask] = 1
                    h_BChw += embed_Cvae * (mask[..., None] - mask_0[..., None])
            else:
                logits_BlV = self.get_logits(x)
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
                if not more_smooth:
                    idx_Bl = sample_with_top_k_top_p_(
                        logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
                    )
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
                else:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                    h_BChw = torch.nn.functional.gumbel_softmax(
                        logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1
                    ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                if (not self.rotary_pos_emb) or (self.absolute_lvl_emb):
                    next_token_map = (
                        self.word_embed(next_token_map)
                        + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
                    )
                else:
                    next_token_map = self.word_embed(next_token_map)
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in self.blocks:
            b.attn.kv_caching(False)
        with torch.autocast(
            "cuda", enabled=False, dtype=torch.float32, cache_enabled=True
        ):
            return self.vae_proxy[0].fhat_to_img(f_hat.float()).add_(1).mul_(0.5)

    def _build_sampler_attn(self, face_group: int, last_len: int) -> torch.Tensor:
        total = 1 + face_group * last_len
        mask = torch.full((total, total), float("-inf"), device=dist.get_device())
        mask[0, :] = 0
        mask[:, 0] = 0
        for f in range(face_group):
            start = 1 + f * last_len
            end = start + last_len
            mask[start:end, start:end] = 0
        return mask.view(1, 1, total, total)

    def sample(
        self,
        feature,
        prev_emb=None,
        attn_bias=None,
        freqs_cis=None,
        face_group: int = 6,
    ):
        if prev_emb is not None:
            feat_ = torch.cat([feature, prev_emb], dim=-1)
            feat_ = self.head_proj(feat_)
            for block in self.feat_extract_blocks:
                feat_ = block(
                    x=feat_,
                    cond_BD=None,
                    attn_bias=attn_bias,
                    freqs_cis=freqs_cis,
                    face_group=face_group,
                )
        return self.head_logits2(self.logit_norm(feat_.float())).float()

    def get_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
        if self.enable_logit_norm:
            logits_feature = self.logit_norm(h.float())
        else:
            logits_feature = h.float()
        return self.head_logits(logits_feature).float()

    def init_weights(
        self,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=0.02,
        conv_std_or_gain=0.02,
    ):
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5
        for m in self.modules():
            with_weight = hasattr(m, "weight") and m.weight is not None
            with_bias = hasattr(m, "bias") and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(
                m,
                (
                    nn.LayerNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(
                m,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head_logits, nn.Linear):
                self.head_logits.weight.data.mul_(init_head)
                self.head_logits.bias.data.zero_()
            elif isinstance(self.head_logits, nn.Sequential):
                self.head_logits[-1].weight.data.mul_(init_head)
                self.head_logits[-1].bias.data.zero_()

        self.reset_faceray_zero_init()

    def reset_faceray_zero_init(self) -> None:
        for adapter in self.faceray_adapters:
            if hasattr(adapter, "out_proj"):
                nn.init.zeros_(adapter.out_proj.weight)
                nn.init.zeros_(adapter.out_proj.bias)
        for adapter in self.faceray_sampler_adapters:
            if hasattr(adapter, "out_proj"):
                nn.init.zeros_(adapter.out_proj.weight)
                nn.init.zeros_(adapter.out_proj.bias)


def build_vae_var_faceray(
    device,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    V=4096,
    Cvae=32,
    ch=160,
    share_quant_resi=4,
    depth=16,
    shared_aln=False,
    attn_l2_norm=True,
    enable_cross=True,
    in_dim_cross=1024,
    flash_if_available=True,
    fused_if_available=True,
    init_adaln=0.5,
    init_adaln_gamma=1e-5,
    init_head=0.02,
    init_std=-1,
    rope_emb=True,
    lvl_emb=True,
    rope_norm=32,
    drop_scale_length=None,
    enable_logit_norm=True,
    enable_adaptive_norm=True,
    train_mode="head_only",
    rope_theta=10000.0,
    vae_ada=False,
    sample_from_idx=9,
):
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    for clz in (
        nn.Linear,
        nn.LayerNorm,
        nn.BatchNorm2d,
        nn.SyncBatchNorm,
        nn.Conv1d,
        nn.Conv2d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
    ):
        setattr(clz, "reset_parameters", lambda self: None)

    vae_local = VQVAE(
        vocab_size=V,
        z_channels=Cvae,
        ch=ch,
        test_mode=True,
        share_quant_resi=share_quant_resi,
        v_patch_nums=patch_nums,
        vae_ada=vae_ada,
    ).to(device)

    var_wo_ddp = VAR(
        vae_local=vae_local,
        depth=depth,
        embed_dim=width,
        num_heads=heads,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=dpr,
        norm_eps=1e-6,
        shared_aln=shared_aln,
        cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        enable_cross=enable_cross,
        in_dim_cross=in_dim_cross,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available,
        fused_if_available=fused_if_available,
        rotary_pos_emb=rope_emb,
        rope_norm=rope_norm,
        absolute_lvl_emb=lvl_emb,
        drop_scale_length=drop_scale_length,
        enable_logit_norm=enable_logit_norm,
        enable_adaptive_norm=enable_adaptive_norm,
        train_mode=train_mode,
        rope_theta=rope_theta,
        sample_from_idx=sample_from_idx,
    ).to(device)

    if train_mode != "head_only":
        var_wo_ddp.init_weights(
            init_adaln=init_adaln,
            init_adaln_gamma=init_adaln_gamma,
            init_head=init_head,
            init_std=init_std,
        )

    return vae_local, var_wo_ddp
