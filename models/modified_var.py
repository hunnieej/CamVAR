import math
from functools import partial
from typing import Optional, Tuple, Union
from utils.mask_utils import Scheduler
import pdb

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.basic_var import AttnBlock, Attention
from models.modified_basic_var import ModifiedAttnBlock
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from models.quant import VectorQuantizer2
from models.embed_rope import compute_axial_cis
import numpy as np
from utils.sample_subset import prob_subset_selection

# Ray adaptation modules
from models.token_grid import TokenGridBuffers
from models.camera_system import CameraEmbedder, RayConstructor, CamRoPE
from models.memory_system import MemoryUpdater


def prepare_attn_mask(encoder_attention_mask):
    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = torch.where(encoder_attention_mask == 1, 0, -torch.inf)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(
            1
        )  # (B,1,1,77)(b,h,c,l)
    return encoder_attention_mask


class SharedAdaLin(nn.Linear):  # 1,L,C
    def forward(self, cond_BD):
        B, L, C_ = cond_BD.shape
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(B, L, 6, C)  # B16C


class RayConstructor:
    """Constructs world-space ray directions from camera parameters and token grid."""

    @staticmethod
    def construct_rays(
        token_grid_buffers: TokenGridBuffers,
        cam_dir: torch.Tensor,  # (B, 3) - unit vector
        fov_deg: float = 100.0,
        patch_size: int = 16,
    ) -> torch.Tensor:
        """
        Construct world-space ray directions for each token.

        Returns:
            r_world: (B, L, 3) - unit ray directions in world space
        """
        B = cam_dir.shape[0]
        device = cam_dir.device
        L = token_grid_buffers.L

        # Get token grid coordinates - all (L,) tensors on device
        u_coords = token_grid_buffers.u_coords.to(device)  # (L,) - patch u coordinate
        v_coords = token_grid_buffers.v_coords.to(device)  # (L,) - patch v coordinate
        patch_sizes = token_grid_buffers.patch_sizes.to(
            device
        )  # (L,) - patch size for this token

        # Compute pixel coordinates for each token's center
        # Each token represents a patch_size x patch_size region
        pixel_u = (u_coords.float() + 0.5) * patch_sizes.float() * patch_size  # (L,)
        pixel_v = (v_coords.float() + 0.5) * patch_sizes.float() * patch_size  # (L,)

        # Image dimensions (assume square for simplicity)
        # Largest scale: 32x32 patches * 16 pixels/patch = 512x512 image
        H = W = 32 * patch_size

        # Normalize to NDC [-1, 1]
        ndc_x = (pixel_u / W) * 2 - 1  # (L,)
        ndc_y = (pixel_v / H) * 2 - 1  # (L,)

        # Compute ray directions in camera space
        fov_rad = math.radians(fov_deg)
        focal_length = 0.5 / math.tan(fov_rad / 2)

        # Camera space rays (pointing along +Z, X right, Y up)
        r_cam_x = ndc_x / focal_length  # (L,)
        r_cam_y = -ndc_y / focal_length  # (L,) - flip Y for image coordinates
        r_cam_z = torch.ones_like(r_cam_x)  # (L,)

        # Normalize camera-space rays
        r_cam = torch.stack([r_cam_x, r_cam_y, r_cam_z], dim=-1)  # (L, 3)
        r_cam = r_cam / r_cam.norm(dim=-1, keepdim=True)  # (L, 3)

        # Build rotation matrix from camera direction to world space
        # Camera looks along cam_dir, with up=[0,1,0]
        cam_dir_norm = cam_dir / (cam_dir.norm(dim=-1, keepdim=True) + 1e-8)  # (B, 3)

        # World up vector
        world_up = torch.tensor([0.0, 1.0, 0.0], device=device).expand(B, 3)  # (B, 3)

        # Camera right = cam_dir × world_up
        cam_right = torch.cross(cam_dir_norm, world_up, dim=-1)  # (B, 3)
        cam_right = cam_right / (cam_right.norm(dim=-1, keepdim=True) + 1e-8)  # (B, 3)

        # Camera up = right × cam_dir
        cam_up = torch.cross(cam_right, cam_dir_norm, dim=-1)  # (B, 3)
        cam_up = cam_up / (cam_up.norm(dim=-1, keepdim=True) + 1e-8)  # (B, 3)

        # Rotation matrix: [right | up | forward]
        # R @ r_cam = r_world
        # r_world = r_cam.x * right + r_cam.y * up + r_cam.z * forward
        r_world = (
            r_cam[:, 0:1].unsqueeze(0) * cam_right.unsqueeze(1)  # (1, L, 1) * (B, 1, 3)
            + r_cam[:, 1:2].unsqueeze(0) * cam_up.unsqueeze(1)  # (1, L, 1) * (B, 1, 3)
            + r_cam[:, 2:3].unsqueeze(0)
            * cam_dir_norm.unsqueeze(1)  # (1, L, 1) * (B, 1, 3)
        )  # (B, L, 3)

        # Normalize (should already be normalized, but ensure numerical stability)
        r_world = r_world / (r_world.norm(dim=-1, keepdim=True) + 1e-8)  # (B, L, 3)

        return r_world


class ModifiedVAR(nn.Module):
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
        enable_cross=True,
        in_dim_cross=1024,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        flash_if_available=True,
        fused_if_available=True,
        noise_sampling=False,
        rotary_pos_emb=True,
        absolute_lvl_emb=True,
        rope_theta=100.0,
        rope_norm=32,
        drop_scale_length=None,
        enable_logit_norm=True,
        enable_adaptive_norm=True,
        train_mode="head_only",
        sample_from_idx=9,
        # Ray adaptation parameters
        enable_ray_adaptation=False,
        adapter_dim=128,
        num_memory_tokens=32,
        ray_adapter_num_heads=4,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = (
            depth,
            embed_dim,
            embed_dim,
            num_heads,
        )

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # progressive training
        self.rotary_pos_emb = rotary_pos_emb
        self.absolute_lvl_emb = absolute_lvl_emb
        self.shared_aln = shared_aln

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn**2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        self.drop_scale_length = drop_scale_length
        if drop_scale_length == None:
            print("no drop, using full self-attention for training...")
        else:
            print("force self-attention map to size ", drop_scale_length)
            self.drop_start_idx = 13
            self.drop_start = self.begin_ends[self.drop_start_idx][0]
            self.num_tokens_to_drop = self.L - self.drop_scale_length

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full(
            (1, num_classes),
            fill_value=1.0 / num_classes,
            dtype=torch.float32,
            device=dist.get_device(),
        )
        self.class_emb = nn.Embedding(
            self.num_classes + 1, self.C
        )  # 每个class的embed：torch.Size([1001, 1024])
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(
            torch.empty(1, self.first_l, self.C)
        )  # 第一层token的初始偏置：torch.Size([1, 1, 1024])，影响不大，仍能产生合理结果
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        if not self.rotary_pos_emb:
            print("using absolute positional encoding...")
            # 3. absolute position embedding
            pos_1LC = []
            for i, pn in enumerate(self.patch_nums):
                pe = torch.empty(1, pn * pn, self.C)
                nn.init.trunc_normal_(pe, mean=0, std=init_std)
                pos_1LC.append(pe)
            pos_1LC = torch.cat(
                pos_1LC, dim=1
            )  # 1, L, C：所有层的token的偏置：torch.Size([1, 680, 1024]),去掉会对模型性能有很大影响
            assert tuple(pos_1LC.shape) == (1, self.L, self.C)
            self.pos_1LC = nn.Parameter(pos_1LC)
            self.freqs_cis = None

        else:
            # -----------RoPE----------------------
            # RoPE axiel（TODO:他们还有一个mixed的版本性能更好，似乎是rope+ape插值）
            print("using rotary positional encoding...")
            self.freqs_cis = []
            self.rope_norm = rope_norm
            self.compute_cis = partial(
                compute_axial_cis,
                dim=embed_dim // num_heads,
                theta=rope_theta,
                normalize=self.rope_norm,
            )
            for i, pn in enumerate(self.patch_nums):
                freqs_cis = self.compute_cis(end_x=pn, end_y=pn)
                self.freqs_cis.append(freqs_cis)
            self.freqs_cis = torch.cat(self.freqs_cis, dim=0).to(
                dist.get_device()
            )  # (L,C//h)
            # ---------------------------------------

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat(
            [torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]
        ).view(1, self.L, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        self.lvl_1L = dT[:, 0].contiguous().to(dist.get_device())

        if self.absolute_lvl_emb:
            print("using absolute level embeding (lvl_embed)...")
            # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
            self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
            nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        if self.shared_aln:
            print("using shared_adaln...")
            self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
            self.lvl_embed_proj = nn.Linear(self.C * 2, self.C)
            self.lvl_embed_adaln = nn.Sequential(
                nn.SiLU(inplace=False), SharedAdaLin(self.D, 6 * self.C)
            )

        # Ray adaptation setup
        self.enable_ray_adaptation = enable_ray_adaptation
        if self.enable_ray_adaptation:
            print(
                f"\n[Ray Adaptation] Enabled with adapter_dim={adapter_dim}, num_memory_tokens={num_memory_tokens}"
            )

            # Token grid buffers
            self.token_grid = TokenGridBuffers(patch_nums=patch_nums)

            # Camera system
            self.camera_embedder = CameraEmbedder(
                input_dim=3, hidden_dim=64, output_dim=64
            )
            self.cam_rope = CamRoPE(
                d_cam=32  # Rotary embedding dimension (16 pairs)
            )

            # Memory system (model-level, shared across all blocks)
            self.memory_updater = MemoryUpdater(
                adapter_dim=adapter_dim, mem_size=num_memory_tokens
            )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule (linearly increasing)

        # Build blocks - use ModifiedAttnBlock if ray adaptation is enabled
        if self.enable_ray_adaptation:
            print("[Ray Adaptation] Using ModifiedAttnBlock for all transformer blocks")
            self.blocks = nn.ModuleList(
                [
                    ModifiedAttnBlock(
                        cond_dim=self.D,
                        shared_aln=shared_aln,
                        in_dim_cross=in_dim_cross,
                        block_idx=block_idx,
                        embed_dim=self.C,
                        norm_layer=norm_layer,
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
                        # Ray adaptation parameters
                        enable_ray_adaptation=True,
                    )
                    for block_idx in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    AttnBlock(
                        cond_dim=self.D,
                        shared_aln=shared_aln,
                        in_dim_cross=in_dim_cross,
                        block_idx=block_idx,
                        embed_dim=self.C,
                        norm_layer=norm_layer,
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
                    )
                    for block_idx in range(depth)
                ]
            )

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f"\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n"
            f"    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n"
            f"    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})",
            end="\n\n",
            flush=True,
        )

        # 这种情况下attn map是正常的
        self.attn_bias_for_masking = (
            torch.where(d >= dT, 0.0, -torch.inf)
            .reshape(1, 1, self.L, self.L)
            .contiguous()
            .to(dist.get_device())
        )
        print("using casual attention...")

        # 6. classifier head
        self.head_logits = nn.Linear(self.C, self.V)
        self.encoder_proj = nn.Linear(in_dim_cross, embed_dim)
        self.noise_sampling = noise_sampling
        print("if using noise sampling: ", noise_sampling)

        # True for 1024 only
        self.enable_logit_norm = enable_logit_norm
        self.enable_adaptive_norm = enable_adaptive_norm
        if self.enable_logit_norm:
            print("enable norm in getting logits...")
            self.logit_norm = norm_layer(embed_dim, elementwise_affine=False)
        if self.enable_adaptive_norm:
            print("enable adaptive norm in getting logits...")
            self.word_embed_head = nn.Linear(self.Cvae, self.C)
            encoder_depth = 3
            self.feat_extract_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        cond_dim=self.D,
                        shared_aln=shared_aln,
                        in_dim_cross=in_dim_cross,
                        block_idx=block_idx,
                        embed_dim=self.C,
                        norm_layer=norm_layer,
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
                    )
                    for block_idx in range(encoder_depth)
                ]
            )
            self.from_idx = sample_from_idx
            self.bg_last, _ = self.begin_ends[self.from_idx]
            _, self.ed_last = self.begin_ends[-1]
            length_ = self.ed_last - self.bg_last
            self.attn_mask = (
                torch.where(
                    d[:, self.bg_last - 1 : self.ed_last]
                    == dT[..., self.bg_last - 1 : self.ed_last],
                    0,
                    -torch.inf,
                )
                .reshape(1, 1, length_ + 1, length_ + 1)
                .contiguous()
                .to(dist.get_device())
            )
            self.attn_mask[:, :, 0] = 0
            self.attn_mask[..., 0] = 0
            self.pos_start_last = nn.Parameter(torch.empty(1, self.first_l, embed_dim))
            nn.init.trunc_normal_(self.pos_start_last.data, mean=0, std=init_std)
            self.logit_norm = norm_layer(embed_dim, elementwise_affine=False)

            self.mask_scheduler = Scheduler()
            self.head_logits2 = nn.Linear(self.C, self.V)
            self.encoder_proj2 = nn.Linear(in_dim_cross, embed_dim)
            self.head_proj = nn.Linear(2 * embed_dim, embed_dim)
            self.feat_drop_enabled = False
            self.stage_2_faster = True
            self.drop_thresh = 0.8
            self.lvl_embed_2 = nn.Embedding(len(self.patch_nums), self.C)
            nn.init.trunc_normal_(self.lvl_embed_2.weight.data, mean=0, std=init_std)

        else:
            self.from_idx = math.inf

        self.train_mode = train_mode
        if self.train_mode == "head_only":
            print("train_mode: head_only")
            [p.requires_grad_(False) for p in self.parameters()]
            [p.requires_grad_(False) for p in self.head_logits.parameters()]
            [p.requires_grad_(True) for p in self.lvl_embed_2.parameters()]

            [p.requires_grad_(True) for p in self.word_embed_head.parameters()]
            [p.requires_grad_(True) for p in self.feat_extract_blocks.parameters()]
            self.pos_start_last.requires_grad_(True)
            [p.requires_grad_(True) for p in self.head_logits2.parameters()]
            [p.requires_grad_(True) for p in self.encoder_proj2.parameters()]
            [p.requires_grad_(True) for p in self.head_proj.parameters()]
            self.init_weights(
                init_adaln=0.5, init_adaln_gamma=5e-5, init_head=0.02, init_std=-1
            )
        else:
            print("train_mode: all")

    def sample(self, feature, prev_emb=None, attn_bias=None, freqs_cis=None):
        if not prev_emb == None:
            feat_ = torch.cat([feature, prev_emb], dim=-1)
            feat_ = self.head_proj(feat_)
            AttnBlock.forward
            for block in self.feat_extract_blocks:
                feat_ = block(
                    x=feat_, cond_BD=None, attn_bias=attn_bias, freqs_cis=freqs_cis
                )
        return self.head_logits2(self.logit_norm(feat_.float())).float()

    def from_logit2emb(self, logits_BlV, t, rng, top_k, top_p, B):
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
        idx_Bl, conf = sample_with_top_k_top_p_(
            logits_BlV,
            rng=rng,
            top_k=top_k,
            top_p=top_p,
            num_samples=1,
            return_conf=True,
        )
        h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
        return h_BChw, conf

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

    @torch.no_grad()
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
        # Ray adaptation parameters
        cam_dir: Optional[torch.Tensor] = None,
        fov_deg: float = 100.0,
        patch_size: int = 16,
        # Metrics collection flag
        collect_metrics: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """Autoregressive inference with optional ray adaptation.

        Returns:
            If collect_metrics=False: torch.Tensor (generated image)
            If collect_metrics=True: tuple (generated image, metrics dict)
        """
        encoder_attention_mask = prepare_attn_mask(
            encoder_attention_mask=encoder_attention_mask
        )

        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng

        # Prepare camera inputs if ray adaptation is enabled
        if self.enable_ray_adaptation and cam_dir is not None:
            # For CFG inference, we need to duplicate cam_dir for both conditional and unconditional
            # cam_dir is (B, 3), we need (2*B, 3) for CFG
            cam_dir_cfg = torch.cat([cam_dir, cam_dir], dim=0)  # (2*B, 3)

            # Construct rays and compute theta
            r_world = RayConstructor.construct_rays(
                self.token_grid, cam_dir_cfg, fov_deg, patch_size
            )  # (2*B, L, 3)
            # Note: cam_dir_cfg is kept as raw (2*B, 3) - blocks will embed it themselves
            theta = self.cam_rope(r_world)  # (2*B, L, 16)

            # Reset memory (Phase 1: zeros) - needs 2*B for CFG
            memory = self.memory_updater.reset_memory(
                2 * B, device=cam_dir.device
            )  # (2*B, M, adapter_dim)
        else:
            cam_dir_cfg = None
            theta = None
            memory = None

        if not self.noise_sampling:
            sos = self.encoder_proj(encoder_pool_feat)
        else:
            sos = torch.randn([2 * B, self.D])

        if not self.rotary_pos_emb:
            lvl_pos = self.pos_1LC
        else:
            lvl_pos = 0

        if self.absolute_lvl_emb:
            lvl_pos = lvl_pos + self.lvl_embed(self.lvl_1L)
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

        # Initialize metrics collection
        all_gate_values = [] if collect_metrics else None

        for b in self.blocks:
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            t = cfg * ratio
            cur_L += pn * pn
            freqs_cis_cur = (
                self.freqs_cis[cur_L - pn * pn : cur_L, :]
                if not self.freqs_cis == None
                else None
            )
            cond_lvl_emb_cur = (
                cond_lvl_emb[:, cur_L - pn * pn : cur_L, ...]
                if not cond_lvl_emb == None
                else None
            )
            # Slice theta for current tokens (same as freqs_cis and cond_lvl_emb)
            theta_cur = (
                theta[:, cur_L - pn * pn : cur_L, :] if theta is not None else None
            )
            x = next_token_map

            for i, b in enumerate(self.blocks):
                if self.enable_ray_adaptation:
                    result = b(
                        x=x,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        cond_BD=cond_lvl_emb_cur,
                        attn_bias=None,
                        freqs_cis=freqs_cis_cur,
                        layer_id=i,
                        # Ray adaptation inputs
                        memory=memory,
                        theta=theta_cur,
                        cam_dir=cam_dir_cfg,
                        camera_embedder=self.camera_embedder,
                        # Metrics collection
                        collect_metrics=collect_metrics,
                    )

                    # Handle return value (tuple if metrics collected, else just x)
                    if collect_metrics:
                        x, block_metrics = result
                        # Only store gate values (Option 3: lightweight)
                        if (
                            block_metrics is not None
                            and "gate_abs_mean" in block_metrics
                        ):
                            all_gate_values.append(block_metrics["gate_abs_mean"])
                    else:
                        x = result
                else:
                    x = b(
                        x=x,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        cond_BD=cond_lvl_emb_cur,
                        attn_bias=None,
                        freqs_cis=freqs_cis_cur,
                        layer_id=i,
                    )

            # For ray adaptation, update memory after all blocks (but not implemented in inference yet)
            # Memory update would go here if needed

            if w_mask and si >= self.from_idx:
                # ... (omitted for brevity, same as original)
                pass
            else:
                logits_BlV = self.get_logits(x)
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

                if not more_smooth:
                    if sample_version == "old":
                        if si < 12:
                            top_k = top_k if si < 9 else 300
                        else:
                            top_k = 100
                        top_p = 0.8
                    else:
                        pass
                    idx_Bl = sample_with_top_k_top_p_(
                        logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
                    )
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
                else:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                    h_BChw = gumbel_softmax_with_rng(
                        logits_BlV.mul(1 + ratio),
                        tau=gum_t,
                        hard=False,
                        dim=-1,
                        rng=rng,
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

        # Generate final image
        with torch.autocast(
            "cuda", enabled=False, dtype=torch.float32, cache_enabled=True
        ):
            generated_img = (
                self.vae_proxy[0].fhat_to_img(f_hat.float()).add_(1).mul_(0.5)
            )

        # Aggregate metrics if collected
        if collect_metrics and all_gate_values:
            # all_gate_values contains Python floats, convert to numpy for aggregation
            gate_array = np.array(all_gate_values)  # (num_blocks,)
            aggregated_metrics = {
                "gate_mean": float(gate_array.mean()),
                "gate_std": float(gate_array.std()),
            }
            return generated_img, aggregated_metrics
        else:
            return generated_img

    def drop_scale(
        self, x_BLC, attn_bias, freqs_cis, start_idx, num_tokens_to_drop=200
    ):
        B, L, C = x_BLC.shape
        all_indices = torch.arange(start_idx, L)
        dropped_indices = torch.randperm(len(all_indices))[:num_tokens_to_drop]
        dropped_indices = all_indices[dropped_indices]

        keep_indices = torch.ones(L, dtype=torch.bool)
        keep_indices[dropped_indices] = False

        x_BLC_dropped = x_BLC[:, keep_indices, :]
        attn_bias_dropped = attn_bias[:, :, keep_indices, :][:, :, :, keep_indices]
        freqs_cis_dropped = freqs_cis[keep_indices, :]

        length_dropped = x_BLC_dropped.shape[1]

        return x_BLC_dropped, attn_bias_dropped, freqs_cis_dropped, keep_indices

    def select_square_region(self, x_BCHW, x_BLC, k, patch_nums, ranges, edge=3):
        B, C, H, W = x_BCHW.shape
        assert k <= H and k <= W, "k should be less than or equal to H and W"

        H_1 = round(ranges[0])
        H_2 = round(ranges[1])
        W_1 = round(ranges[2])
        W_2 = round(ranges[3])
        top_left_x = torch.randint(H_1, H_2 - k + 1, (1,)).item()
        top_left_y = torch.randint(W_1, W_2 - k + 1, (1,)).item()

        selected_region = (
            x_BCHW[:, :, top_left_x : top_left_x + k, top_left_y : top_left_y + k]
            .reshape(B, C, -1)
            .transpose(1, 2)
        )

        row_indices = torch.arange(top_left_x, top_left_x + k).unsqueeze(1) * patch_nums
        col_indices = torch.arange(top_left_y, top_left_y + k)
        grid_indices = (row_indices + col_indices).flatten()

        selected_indices = grid_indices.view(-1)

        return (
            selected_region,
            selected_indices,
            [
                top_left_x + edge,
                top_left_x + k - edge,
                top_left_y + edge,
                top_left_y + k - edge,
            ],
        )

    def drop_scale_v2(
        self, x_BLC, attn_bias, freqs_cis, start_idx, k_list=[5, 5, 5, 5, 5]
    ):
        B, L, C = x_BLC.shape
        keep_indices = torch.ones(L, dtype=torch.bool)

        ranges = [0, 1, 0, 1]
        for idx in range(min(len(k_list), len(self.patch_nums) - self.drop_start_idx)):
            bg, ed = self.begin_ends[idx + self.drop_start_idx]
            pn = self.patch_nums[idx + self.drop_start_idx]
            x_BCHW = x_BLC[:, bg:ed].transpose(1, 2).reshape(B, C, pn, pn)
            x_BLC_dropped, dropped_indices, ranges = self.select_square_region(
                x_BCHW, x_BLC, k_list[idx], pn, [i * pn for i in ranges]
            )
            ranges = [i / pn for i in ranges]
            dropped_indices = dropped_indices + bg
            keep_indices[bg:ed] = False
            keep_indices[dropped_indices] = True

        x_BLC_dropped = x_BLC[:, keep_indices, :]
        attn_bias_dropped = attn_bias[:, :, keep_indices, :][:, :, :, keep_indices]
        freqs_cis_dropped = freqs_cis[keep_indices, :]

        length_dropped = x_BLC_dropped.shape[1]

        return x_BLC_dropped, attn_bias_dropped, freqs_cis_dropped, keep_indices

    def forward(
        self,
        x_BLCv_wo_first_l: torch.Tensor,
        encoder_hidden_states=None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_pool_feat=None,
        embed_Cvae=None,
        # Ray adaptation parameters
        cam_dir: Optional[torch.Tensor] = None,
        fov_deg: float = 100.0,
        patch_size: int = 16,
        # Metrics collection
        collect_metrics: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional ray adaptation.

        Args:
            cam_dir: (B, 3) camera direction unit vectors (only used if enable_ray_adaptation=True)
            fov_deg: Field of view in degrees
            patch_size: Patch size for ray construction
            collect_metrics: Whether to collect ray adapter metrics

        Returns:
            logits_BLV: Output logits
            x_BLC: Final feature map
            drop_idxs: Indices of dropped tokens (if any)
            aggregated_metrics: Dict with aggregated metrics (if collect_metrics=True), else None
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]

        # Prepare camera inputs if ray adaptation is enabled
        if self.enable_ray_adaptation and cam_dir is not None:
            # Construct rays and compute theta
            r_world = RayConstructor.construct_rays(
                self.token_grid, cam_dir, fov_deg, patch_size
            )  # (B, L, 3)
            theta = self.cam_rope(r_world)  # (B, L, 16)

            # Reset memory (Phase 1: zeros each forward pass)
            memory = self.memory_updater.reset_memory(
                B, device=cam_dir.device
            )  # (B, M, adapter_dim)
        else:
            theta = None
            memory = None

        with torch.amp.autocast("cuda", enabled=False):
            sos = cond_BD = self.encoder_proj(encoder_pool_feat)

            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(
                B, self.first_l, -1
            )

            if self.prog_si == 0:
                x_BLC = sos
            else:
                x_BLC = torch.cat(
                    (sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1
                )

            if not self.rotary_pos_emb:
                x_BLC += self.pos_1LC[:, :ed]
            if self.absolute_lvl_emb:
                x_BLC += self.lvl_embed(self.lvl_1L[:, :ed])
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

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        freqs_cis = self.freqs_cis

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        encoder_attention_mask = prepare_attn_mask(
            encoder_attention_mask=encoder_attention_mask
        )

        if not self.drop_scale_length == None:
            x_BLC, attn_bias, freqs_cis, drop_idxs = self.drop_scale(
                x_BLC,
                attn_bias,
                freqs_cis,
                start_idx=self.drop_start,
                num_tokens_to_drop=self.num_tokens_to_drop,
            )
        else:
            drop_idxs = None

        # Transformer blocks with ray adaptation
        # Collect metrics per block if requested
        block_metrics_list = (
            [] if (collect_metrics and self.enable_ray_adaptation) else None
        )

        for i, b in enumerate(self.blocks):
            if self.enable_ray_adaptation:
                result = b(
                    x=x_BLC,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    cond_BD=cond_lvl_emb,
                    attn_bias=attn_bias,
                    freqs_cis=freqs_cis,
                    layer_id=i,
                    # Ray adaptation inputs (read-only)
                    memory=memory,
                    theta=theta,
                    cam_dir=cam_dir,
                    camera_embedder=self.camera_embedder,
                    collect_metrics=collect_metrics,
                )

                # Handle return value (may be tuple if metrics collected)
                if collect_metrics:
                    x_BLC, block_metrics = result
                    block_metrics_list.append(block_metrics)
                else:
                    x_BLC = result
            else:
                x_BLC = b(
                    x=x_BLC,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    cond_BD=cond_lvl_emb,
                    attn_bias=attn_bias,
                    freqs_cis=freqs_cis,
                    layer_id=i,
                )

        # Memory update happens OUTSIDE transformer loop (Phase 1: not used yet)
        # In Phase 2, we would update memory here based on x_BLC
        if self.enable_ray_adaptation and memory is not None:
            # Phase 1: Memory remains zeros (no update)
            # Phase 2 would be: memory = self.memory_updater.update_memory(memory, x_BLC)
            pass

        logits_BLV = self.get_logits(x_BLC.float())

        # Aggregate metrics if collected
        aggregated_metrics = None
        if collect_metrics and block_metrics_list:
            aggregated_metrics = {}
            metric_keys = block_metrics_list[0].keys()

            # Compute mean and std across blocks for each metric
            for key in metric_keys:
                values = [m[key] for m in block_metrics_list]
                aggregated_metrics[f"{key}_mean"] = sum(values) / len(values)

                # Compute std for gate metrics only (most important)
                if "gate" in key:
                    mean_val = aggregated_metrics[f"{key}_mean"]
                    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                    aggregated_metrics[f"{key}_std_across_blocks"] = variance**0.5

        return logits_BLV, x_BLC, drop_idxs, aggregated_metrics

    def forward_sampler(
        self,
        x_BLCv_wo_first_l: torch.Tensor,
        encoder_hidden_states=None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_pool_feat=None,
        embed_Cvae=None,
    ):
        with torch.no_grad():
            logits_BLV, feat_BlC, _, _ = self.forward(
                x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_pool_feat=encoder_pool_feat,
            )
        B = embed_Cvae.shape[0]
        bg, ed = self.bg_last, self.ed_last
        logits_last = logits_BLV[:, bg:ed]
        feat_last = feat_BlC[:, bg:ed]
        pn = self.patch_nums[-1]

        total_logits = []
        mask = self.mask_scheduler.add_mask_for_training(embed_Cvae[..., 0])
        embed_Cvae = embed_Cvae * mask[..., None]

        text_pool_feat = (
            self.encoder_proj2(encoder_pool_feat.unsqueeze(1)) + self.pos_start_last
        )

        freqs_cis = torch.cat([self.freqs_cis[0, None], self.freqs_cis[bg:ed]], dim=0)
        logits_BLV_ = self.sample(
            feature=torch.cat(
                [
                    text_pool_feat,
                    self.word_embed_head(embed_Cvae)
                    + self.lvl_embed_2(self.lvl_1L[:, bg:ed]),
                ],
                dim=1,
            ),
            prev_emb=torch.cat([text_pool_feat, feat_last], dim=1),
            attn_bias=self.attn_mask,
            freqs_cis=freqs_cis,
        )
        return logits_BLV_[:, 1:], mask

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

        print(f"[init_weights] {type(self).__name__} with {init_std=:g}")
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

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            if isinstance(sab, AttnBlock):
                sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
                sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
                if hasattr(sab.ffn, "fcg") and sab.ffn.fcg is not None:
                    nn.init.ones_(sab.ffn.fcg.bias)
                    nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
                if hasattr(sab, "ada_lin"):
                    sab.ada_lin[-1].weight.data[2 * self.C :].mul_(init_adaln)
                    sab.ada_lin[-1].weight.data[: 2 * self.C].mul_(init_adaln_gamma)
                    if (
                        hasattr(sab.ada_lin[-1], "bias")
                        and sab.ada_lin[-1].bias is not None
                    ):
                        sab.ada_lin[-1].bias.data.zero_()
                elif hasattr(sab, "ada_gss"):
                    sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                    sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
            # ModifiedAttnBlock uses the same structure, apply same init
            elif hasattr(sab, "attn"):
                sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
                sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))

    def extra_repr(self):
        return f"drop_path_rate={self.drop_path_rate:g}"

    def freeze_pretrained_parameters(self):
        """Freeze all pretrained VAR parameters, only train ray adaptation modules."""
        if not self.enable_ray_adaptation:
            print(
                "[freeze_pretrained_parameters] Ray adaptation not enabled, nothing to freeze"
            )
            return

        # Freeze all parameters first
        for p in self.parameters():
            p.requires_grad_(False)

        # Unfreeze ray adaptation parameters
        trainable_params = []

        # Camera system
        for p in self.camera_embedder.parameters():
            p.requires_grad_(True)
            trainable_params.append(p)

        for p in self.cam_rope.parameters():
            p.requires_grad_(True)
            trainable_params.append(p)

        # Memory updater
        for p in self.memory_updater.parameters():
            p.requires_grad_(True)
            trainable_params.append(p)

        # Ray adapters and gates in each block
        for block in self.blocks:
            if hasattr(block, "ray_adapter"):
                for p in block.ray_adapter.parameters():
                    p.requires_grad_(True)
                    trainable_params.append(p)
            if hasattr(block, "gate"):
                for p in block.gate.parameters():
                    p.requires_grad_(True)
                    trainable_params.append(p)

        num_trainable = sum(p.numel() for p in trainable_params)
        num_total = sum(p.numel() for p in self.parameters())
        print(
            f"[freeze_pretrained_parameters] Trainable: {num_trainable:,} / {num_total:,} parameters ({100 * num_trainable / num_total:.2f}%)"
        )
        print(
            f"    - camera_embedder: {sum(p.numel() for p in self.camera_embedder.parameters()):,}"
        )
        print(f"    - cam_rope: {sum(p.numel() for p in self.cam_rope.parameters()):,}")
        print(
            f"    - memory_updater: {sum(p.numel() for p in self.memory_updater.parameters()):,}"
        )
        print(
            f"    - ray_adapters (all blocks): {sum(p.numel() for b in self.blocks if hasattr(b, 'ray_adapter') for p in b.ray_adapter.parameters()):,}"
        )
        print(
            f"    - gates (all blocks): {sum(p.numel() for b in self.blocks if hasattr(b, 'gate') for p in b.gate.parameters()):,}"
        )


class VARHF(ModifiedVAR, PyTorchModelHubMixin):
    def __init__(
        self,
        vae_kwargs,
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
        flash_if_available=True,
        fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_eps=norm_eps,
            shared_aln=shared_aln,
            cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available,
            fused_if_available=fused_if_available,
        )
