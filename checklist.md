# FaceRay-STAR Cubemap Extension: Implementation Checklist (v1)

## 0) High-level goal
Extend pretrained STAR d30 512x512 to cubemap panorama generation by adding a UCPE-inspired sparse inter-face camera attention adapter (FaceRayAttentionAdapter) and training the masked sampler. Keep STAR backbone intact (normalized RoPE on Q/K only, lvl_embed unchanged, head init scaling init_head=0.02 unchanged).

Face order (canonical): [F, R, B, L, U, D]  
Image size: 512 (no resize)  
Boundary strip width: 2  
Neighbor window: k=2 (tangent window) + corner union  
Adapter: C=1920, C’=240, H_ray=8, head_dim=30, ray_subdim=24 (r=8)  

---

## 1) Files to exist
### New files
- models/cubemap_geometry.py
- models/face_ray_attention.py
- data/cubemap_scene_dataset.py

### Copied + modified files (originals unchanged)
- models/var_drop_faceray.py  (copy of models/var_drop.py)
- (optional) models/basic_var_faceray.py (copy of models/basic_var.py if needed)

### Training scripts
- train_faceray.py
- train_sampler_faceray.py

---

## 2) Dataset loader acceptance
Loader: data/cubemap_scene_dataset.py

Given a scene folder:
sceneXXXX/
  faces/F.png R.png B.png L.png U.png D.png
  prompts.json

__getitem__ returns:
- faces: float tensor [6,3,512,512] in [F,R,B,L,U,D] order
- prompt: str
- scene_id: str

No augmentation. dice.png is ignored in v1.

---

## 3) Geometry utilities acceptance
File: models/cubemap_geometry.py

Must provide:
- Face frames (n_f, a_f, b_f) for F/R/B/L/U/D
- Token center coordinates (u,v) given (h_s,w_s) and (i,j)
- Ray direction d = normalize(n + u*a + v*b)
- Ray-frame basis:
  e_z=d
  e_y = normalize(b_f - (b_f·d)d) with fallback to a_f when degenerate
  e_x = e_y x e_z
  R_ray=[e_x e_y e_z], D_ray=R_ray^T (world->ray)
- Neighbor mapping pi(t) via canonical cubemap projection (no flip tables)
- Boundary strip indices for strip_w=2
- Precompute buffers per scale for boundary tokens:
  - boundary indices
  - neighbor indices (with tangent window k=2 and corner union)
  - D_ray for boundary tokens

Sanity checks:
- Orthonormality: max|R^T R - I| < 1e-3 on sampled boundary tokens
- Neighbor mapping counts: boundary tokens map primarily to cube-adjacent faces (not random)

---

## 4) FaceRayAttentionAdapter acceptance
File: models/face_ray_attention.py

Config must match:
- C=1920, C’=240, H_ray=8, head_dim=30
- ray_subdim=24 (r=8), remaining 6 dims passthrough per head
- Apply D_ray to Q/K/V ray_subspace; inverse on output (query token)
- Sparse attention ONLY for boundary tokens using precomputed neighbors
- Zero-init output projection back to C (weights and bias = 0 at init)

Sanity checks:
- With zero-init, adapter delta ||Δx|| is near 0 before training (e.g., mean abs < 1e-6)

---

## 5) Backbone integration acceptance
File: models/var_drop_faceray.py

Requirement:
- Insert FaceRayAdapter as parallel residual branch to self-attn:
  x_in = x
  x = x + self_attn(x_in)              # STAR path, unchanged RoPE/lvl_embed
  x = x + zero_proj(adapter(x_in, s))  # adapter path
  then cross-attn(text) and FFN unchanged

Keep:
- normalized RoPE per-scale on Q/K only
- lvl_embed unchanged
- init_head scaling on head_logits (0.02) unchanged

Checkpoint:
- Load official d30 512 ckpt with strict=False
- Missing keys must be only for new FaceRay modules / faceray sampler blocks

---

## 6) Masked sampler (included) acceptance
File: models/var_drop_faceray.py

- forward_sampler must flatten last-scale tokens to [B, 6N, ...]
- Repeat freqs_cis per face for last scale (prepend start token)
- Build block-diagonal attention bias for sampler self-attn:
  - text_pool token attends to all
  - intra-face allowed
  - inter-face disallowed inside sampler attention (face-separated)
- Use FaceRayAdapter inside sampler feat_extract_blocks (faceray copies)

Masking:
- Seam-coupled masking OFF in v1 (random mask only)

L_mask:
- Cross entropy on masked positions only

---

## 7) Seam loss acceptance
Compute seam loss on last-scale pre-logit features feat_last:
- Reshape feat_last -> [B,6,pn,pn,1920]
- Extract boundary strips (strip_w=2)
- Pair tokens using same neighbor mapping pi(t)
- L_seam = SmoothL1 on paired features

---

## 8) Training scripts acceptance
### train_faceray.py
- Loads official d30 512 ckpt (strict=False)
- Trains generator (FaceRayAdapter + optional unfrozen modules), backbone can be frozen for v1
- Loss: L_face + lambda_seam_eff * L_seam
- lambda_seam default 0.1
- seam_warmup_ratio default 0.10
  lambda_seam_eff = lambda_seam * min(1, step / (total_steps*seam_warmup_ratio))

### train_sampler_faceray.py
- Trains masked sampler only (lambda_mask=1.0)
- Should accept --gen_ckpt to load the generator (preferably Stage1 checkpoint)
- Loss: L_mask

---

## 9) Minimal runbook (expected passes)
1) Dataset loads 10 samples and prints shapes OK.
2) Geometry sanity checks pass (orthonormality + neighbor counts).
3) Forward pass runs with adapter zero-init (delta ~0).
4) One training step runs for train_faceray.py without NaNs.
5) forward_sampler runs and returns logits + mask with correct shapes.