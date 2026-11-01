"""K1 vlm-dedupe: CPU proof that encoding only the unique images of a micro-batch is exact.

Builds a tiny random-weight Qwen2.5-VL and Qwen3-VL (transformers >= 5.x), a micro-batch of
8 samples drawn from 2 unique images (4 siblings each, shuffled, distinct response tokens),
and compares
  verl  : verl's own wrapper (_get_input_embeds in verl/models/transformers/qwen2_vl.py:342-402,
          qwen3_vl.py:206-298), i.e. what the FSDP engine runs today
  naive : model.visual on the concatenated pixel_values of all 8 samples (same thing, inline)
  dedupe: model.visual on the 2 unique images, index_select back to every occurrence
on inputs_embeds (bitwise) and on the final logits of the language model (bitwise / max|diff|).

Usage: <repo>/.venv-a9/bin/python poc_dedupe_cpu.py
"""

import os
import sys
import time
import traceback

import torch

torch.manual_seed(0)
torch.set_num_threads(max(1, os.cpu_count() // 2))

from transformers import (  # noqa: E402
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

IMAGE_TOKEN_ID = 5
VISION_START_ID = 3
VISION_END_ID = 4
VIDEO_TOKEN_ID = 6
VOCAB = 256


def unpack_visual_output(out):
    """Mirror of verl.utils.transformers_compat.unpack_visual_output (transformers_compat.py:77-92)."""
    if hasattr(out, "pooler_output"):
        return out.pooler_output, getattr(out, "deepstack_features", None)
    if isinstance(out, tuple):
        return out
    return out, None


def set_mrope(text_cfg, head_dim):
    half = head_dim // 2
    section = [half // 2, half // 4, half - half // 2 - half // 4]
    for key in ("rope_parameters", "rope_scaling"):
        cur = getattr(text_cfg, key, None)
        if isinstance(cur, dict):
            cur = dict(cur)
            cur["mrope_section"] = section
            setattr(text_cfg, key, cur)
            return key, section
    return None, section


def build_q25():
    cfg = Qwen2_5_VLConfig(
        text_config=dict(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=VOCAB,
            max_position_embeddings=512,
        ),
        vision_config=dict(
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            in_channels=3,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            window_size=28,  # 28 / 14 / 2 = 1 merged token per window -> exercises the window path
            fullatt_block_indexes=[1],
            out_hidden_size=128,
        ),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
        vocab_size=VOCAB,
        tie_word_embeddings=False,
    )
    set_mrope(cfg.text_config, 128 // 4)
    return cfg, Qwen2_5_VLForConditionalGeneration


def build_q3():
    cfg = Qwen3VLConfig(
        text_config=dict(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=VOCAB,
            max_position_embeddings=512,
        ),
        vision_config=dict(
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            in_channels=3,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=128,
            num_position_embeddings=64,  # 8x8 grid for the learned pos-embed interpolation
            deepstack_visual_indexes=[0],
        ),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
        vocab_size=VOCAB,
        tie_word_embeddings=False,
    )
    set_mrope(cfg.text_config, 128 // 4)
    return cfg, Qwen3VLForConditionalGeneration


def base_of(model):
    """transformers >= 5 nests the tower: ForConditionalGeneration.model.{visual,language_model}.
    verl's wrappers receive that inner model (qwen2_vl.py:421-436 passes self.model)."""
    inner = getattr(model, "model", None)
    return inner if inner is not None and hasattr(inner, "visual") else model


def make_images(cfg):
    """Two unique images with different grids; returns list of (pixel_rows, grid_thw)."""
    v = cfg.vision_config
    patch_dim = v.in_channels * v.temporal_patch_size * v.patch_size**2
    grids = [(1, 4, 4), (1, 4, 6)]  # 16 patches / 4 tokens, 24 patches / 6 tokens
    images = []
    for g in grids:
        n = g[0] * g[1] * g[2]
        images.append((torch.randn(n, patch_dim), torch.tensor([g], dtype=torch.long)))
    return images


def make_micro_batch(cfg, images, order):
    """Samples in `order` (list of unique-image indices), one image per sample,
    rmpad-style: input_ids concatenated into (1, total_nnz) as verl's rmpad path does."""
    merge = cfg.vision_config.spatial_merge_size**2
    ids, pv, grid, keys = [], [], [], []
    for s, img_idx in enumerate(order):
        rows, g = images[img_idx]
        n_tok = int(g[0, 0] * g[0, 1] * g[0, 2]) // merge
        prompt = torch.randint(10, VOCAB, (3,), generator=torch.Generator().manual_seed(100 + img_idx))
        response = torch.randint(10, VOCAB, (5,), generator=torch.Generator().manual_seed(1000 + s))
        seq = torch.cat(
            [
                torch.tensor([1, VISION_START_ID]),
                torch.full((n_tok,), IMAGE_TOKEN_ID),
                torch.tensor([VISION_END_ID]),
                prompt,
                response,
            ]
        )
        ids.append(seq)
        pv.append(rows)
        grid.append(g)
        keys.append(f"uid{img_idx}:0")
    return torch.cat(ids).unsqueeze(0), torch.cat(pv, 0), torch.cat(grid, 0), keys


def naive_image_embeds(base, pixel_values, grid_thw):
    pixel_values = pixel_values.type(base.visual.dtype)
    embeds, deep = unpack_visual_output(base.visual(pixel_values, grid_thw=grid_thw))
    return embeds, (list(deep) if deep is not None else None)


def dedupe_image_embeds(base, pixel_values, grid_thw, image_keys):
    """The proposed change: encode each unique key once, index_select back to every occurrence."""
    merge = base.config.vision_config.spatial_merge_size**2
    patches = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
    tokens = [p // merge for p in patches]
    patch_off = [0]
    for p in patches:
        patch_off.append(patch_off[-1] + p)
    uniq, first = {}, []
    for i, k in enumerate(image_keys):
        if k not in uniq:
            uniq[k] = len(first)
            first.append(i)
    if len(first) == len(image_keys):
        return naive_image_embeds(base, pixel_values, grid_thw) + (len(first),)
    pv_u = torch.cat([pixel_values[patch_off[i] : patch_off[i + 1]] for i in first], 0)
    grid_u = grid_thw[first]
    embeds_u, deep_u = naive_image_embeds(base, pv_u, grid_u)
    tok_off_u = [0]
    for i in first:
        tok_off_u.append(tok_off_u[-1] + tokens[i])
    gather = torch.cat(
        [torch.arange(tok_off_u[uniq[k]], tok_off_u[uniq[k]] + tokens[i]) for i, k in enumerate(image_keys)]
    )
    embeds = embeds_u.index_select(0, gather)
    deep = [d.index_select(0, gather) for d in deep_u] if deep_u is not None else None
    return embeds, deep, len(first)


def scatter(base, input_ids, image_embeds):
    inputs_embeds = base.get_input_embeddings()(input_ids)
    mask = (input_ids == base.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    assert int((input_ids == base.config.image_token_id).sum()) == image_embeds.shape[0]
    return inputs_embeds.masked_scatter(mask, image_embeds.to(inputs_embeds.dtype)), mask[..., 0]


def lm_logits(model, base, inputs_embeds, visual_pos_masks=None, deepstack=None):
    kwargs = {}
    if deepstack is not None:
        kwargs = {"visual_pos_masks": visual_pos_masks, "deepstack_visual_embeds": deepstack}
    out = base.language_model(inputs_embeds=inputs_embeds, **kwargs)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return model.lm_head(hidden)


def verl_inputs_embeds(name, base, input_ids, pixel_values, grid_thw):
    """Run verl's real wrapper; returns (inputs_embeds, deepstack or None) or None if not importable."""
    try:
        if name.startswith("qwen2"):
            from verl.models.transformers.qwen2_vl import _get_input_embeds

            ie, _ = _get_input_embeds(base, input_ids, None, pixel_values, None, grid_thw, None)
            return ie, None
        from verl.models.transformers.qwen3_vl import _get_input_embeds

        out = _get_input_embeds(base, input_ids, None, pixel_values, None, grid_thw, None)
        return out["inputs_embeds"], out["deepstack_visual_embeds"]
    except Exception:
        traceback.print_exc()
        return None


def run(name, author, attn_impl):
    cfg, cls = author()
    cfg._attn_implementation = attn_impl
    try:
        cfg.text_config._attn_implementation = attn_impl
        cfg.vision_config._attn_implementation = attn_impl
    except Exception:
        pass
    model = cls(cfg).eval()
    base = base_of(model)
    images = make_images(cfg)
    order = [0, 1, 0, 1, 1, 0, 0, 1]
    input_ids, pixel_values, grid_thw, keys = make_micro_batch(cfg, images, order)
    with torch.no_grad():
        t0 = time.perf_counter()
        emb_n, deep_n = naive_image_embeds(base, pixel_values, grid_thw)
        t1 = time.perf_counter()
        emb_d, deep_d, n_uniq = dedupe_image_embeds(base, pixel_values, grid_thw, keys)
        t2 = time.perf_counter()
        ie_n, vmask = scatter(base, input_ids, emb_n)
        ie_d, _ = scatter(base, input_ids, emb_d)
        lg_n = lm_logits(model, base, ie_n, vmask, deep_n)
        lg_d = lm_logits(model, base, ie_d, vmask, deep_d)
        verl_out = verl_inputs_embeds(name, base, input_ids, pixel_values, grid_thw)
    deep_eq = None
    if deep_n is not None:
        deep_eq = all(torch.equal(a, b) for a, b in zip(deep_n, deep_d))
    verl_eq = None
    if verl_out is not None:
        ie_v, deep_v = verl_out
        verl_eq = torch.equal(ie_v, ie_d)
        if deep_v is not None and deep_d is not None:
            verl_eq = verl_eq and all(torch.equal(a, b) for a, b in zip(deep_v, deep_d))
    print(
        f"{name:10s} attn={attn_impl:5s} images={len(order)} unique={n_uniq} "
        f"patches={pixel_values.shape[0]} tokens={emb_n.shape[0]} | "
        f"image_embeds bitwise={torch.equal(emb_n, emb_d)} max|d|={(emb_n - emb_d).abs().max().item():.3e} "
        f"deepstack bitwise={deep_eq} | inputs_embeds bitwise={torch.equal(ie_n, ie_d)} "
        f"verl-wrapper==dedupe bitwise={verl_eq} | "
        f"logits bitwise={torch.equal(lg_n, lg_d)} max|d|={(lg_n - lg_d).abs().max().item():.3e} | "
        f"visual naive {1e3 * (t1 - t0):.1f} ms, dedupe {1e3 * (t2 - t1):.1f} ms"
    )
    return (lg_n - lg_d).abs().max().item(), verl_eq


def processor_determinism():
    """Siblings of one prompt get bit-identical pixel rows: a (uid, image_idx) key is exact."""
    try:
        from PIL import Image
        from transformers import Qwen2VLImageProcessor

        proc = Qwen2VLImageProcessor(min_pixels=4 * 28 * 28, max_pixels=64 * 28 * 28)
        rng = torch.Generator().manual_seed(7)
        arr = (torch.rand(90, 120, 3, generator=rng) * 255).to(torch.uint8).numpy()
        img = Image.fromarray(arr)
        a = proc(images=[img], return_tensors="pt")
        b = proc(images=[img, img], return_tensors="pt")
        pa, pb = a["pixel_values"], b["pixel_values"]
        n = pa.shape[0]
        print(
            f"processor   grid={a['image_grid_thw'].tolist()} rows={n} "
            f"same-image-twice rows==2x: {pb.shape[0] == 2 * n} "
            f"copy1 bitwise={torch.equal(pa, pb[:n])} copy2 bitwise={torch.equal(pa, pb[n:])}"
        )
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    import transformers

    print(f"torch {torch.__version__} transformers {transformers.__version__} threads {torch.get_num_threads()}")
    ok = True
    for name, author in (("qwen2.5-vl", build_q25), ("qwen3-vl", build_q3)):
        for attn in ("eager", "sdpa"):
            try:
                d, verl_eq = run(name, author, attn)
                ok &= d < 1e-5 and verl_eq is not False
            except Exception:
                ok = False
                print(f"{name} attn={attn} FAILED")
                traceback.print_exc()
    processor_determinism()
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
