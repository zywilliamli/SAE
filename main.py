#!/usr/bin/env python3
"""
sae_feature_pipeline.py  –  robust to both *Sae* (new API) and *SparseCoder* (legacy)
Author: William Li   •   2025-05-25

▪ Discovers + labels sparse-autoencoder features for EleutherAI/pythia-160m
▪ SAE repo: EleutherAI/sae-pythia-160m-32k
▪ Dataset : neelnanda/pile-10k
▪ Builds two FAISS indexes:
      1) label-text embeddings (OpenAI)    – semantic search
      2) decoder weight vectors            – geometric search & steering lookup
▪ Two steering modes:  "bias" (fast)  or  "latent" (boost-and-reconstruct)

To install deps:

    pip install "transformers>=4.40" "datasets>=2.19" faiss-cpu \
                sparsify openai tqdm

and set:

    export OPENAI_API_KEY="sk-…"
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import pickle
from dataclasses import dataclass
from typing import Dict, List, Sequence, Any

import faiss
import numpy as np
import openai
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ───────────────────────────── CONFIG ───────────────────────────── #
@dataclass
class Config:
    # models
    lm_name: str = "EleutherAI/pythia-160m"
    sae_repo: str = "EleutherAI/sae-pythia-160m-32k"
    embed_model: str = "text-embedding-3-small"
    gpt_label_model: str = "gpt-4o-mini"
    # data
    dataset: str = "neelnanda/pile-10k"
    n_rows: int = 50  # dataset rows to scan
    max_len: int = 512
    batch_size: int = 8
    layers: Sequence[int] | None = None  # None → all SAEs; else list, e.g. [5]
    topk_per_token: int = 5
    examples_per_feat: int = 20
    # steering
    default_strength: float = 5.0
    # system
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # persistence
    workdir: pathlib.Path = pathlib.Path("sae_feature_work")
    text_index_name: str = "label_text.faiss"
    vec_index_name: str = "decoder_vec.faiss"
    meta_name: str = "meta.pkl"


cfg = Config()
cfg.layers = [10]
cfg.workdir.mkdir(exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # suppress libomp clash
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence HF fork warning
faiss.omp_set_num_threads(1)  # optional: single-thread FAISS

openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key, "Set OPENAI_API_KEY in your environment!"

# ────────────────────────── LOAD MODELS ────────────────────────── #
print("Loading tokenizer & language model …")
tok = AutoTokenizer.from_pretrained(cfg.lm_name)
tok.pad_token = tok.eos_token
lm = AutoModelForCausalLM.from_pretrained(
    cfg.lm_name,
    torch_dtype=torch.bfloat16,
    device_map={"": cfg.device},
).eval()

print("Loading SAEs …")
from sparsify import Sae  # both classes may appear

saes_raw: Dict[str, Any] = Sae.load_many(cfg.sae_repo)  # returns Sae or SparseCoder

# keep only specific layers if configured
if cfg.layers is not None:
    saes_raw = {k: v for k, v in saes_raw.items()
                if k.startswith("layers.") and int(k.split(".")[1]) in cfg.layers}


# ───────── util: unified access to decoder weight & dims ───────── #
def decoder_weight(sae_obj):
    """
    Return decoder weight matrix (latent_dim, d_model) as a detached CPU tensor.
    Works for both new Sae and legacy SparseCoder checkpoints.
    """
    if hasattr(sae_obj, "decoder"):  # new API
        return sae_obj.decoder.weight.detach().cpu()
    for field in ("W_dec", "W_decoder", "reconstruction_weight"):
        if hasattr(sae_obj, field):  # legacy API
            return getattr(sae_obj, field).detach().cpu()
    raise AttributeError("Decoder weight matrix not found")


def latent_acts(output):
    """
    Return the latent-activation tensor no matter what the SAE/SparseCoder
    gives back – plain Tensor, EncoderOutput, or a tuple.
    Expected final shape: (batch, seq, latent_dim)
    """
    if isinstance(output, torch.Tensor):
        return output
    # EncoderOutput dataclass (newer *sparsify*)
    if hasattr(output, "sae_acts"):
        return output.sae_acts
    # legacy (acts, recon) tuple
    if isinstance(output, (list, tuple)) and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Unrecognised SAE encode output type")


first_sae = next(iter(saes_raw.values()))
W_dec0 = decoder_weight(first_sae)
latent_dim, d_model = W_dec0.shape
print(f"Using {len(saes_raw)} hookpoints × {latent_dim} latents each")

# ────────────────────── DATASET & HELPERS ─────────────────────── #
ds = load_dataset(cfg.dataset, split="train")
row_iter = (ds[i]["text"] for i in range(min(cfg.n_rows, len(ds))))


def batched(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield buf;
            buf = []
    if buf: yield buf


def embed_text(texts: List[str]) -> np.ndarray:
    resp = openai.embeddings.create(model=cfg.embed_model, input=texts)
    return np.asarray([r.embedding for r in resp.data], dtype="float32")


def label_feature(tokens: List[str], contexts: List[str]) -> tuple[str, str]:
    prompt = (f"Analyze this SAE feature.\n\n"
              f"Top tokens: {', '.join(tokens[:10])}\n\n"
              f"Top contexts:\n" + "\n".join(f"- {c}" for c in contexts[:5]) +
              "\n\nReturn:\nLABEL: <2-5 words>\nDESCRIPTION: <1-2 sentences>")
    ans = openai.chat.completions.create(
        model=cfg.gpt_label_model,
        temperature=0.3,
        max_tokens=120,
        messages=[{"role": "system", "content": "You are an expert at analysing latent features."},
                  {"role": "user", "content": prompt}]
    ).choices[0].message.content
    label, descr = "", ""
    for line in ans.splitlines():
        if line.strip().startswith("LABEL:"):
            label = line.split(":", 1)[1].strip()
        elif line.strip().startswith("DESCRIPTION:"):
            descr = line.split(":", 1)[1].strip()
    return (label or "Unlabelled", descr or "–")


# ───────────────────────── DISCOVERY ──────────────────────────── #
print("Scanning dataset for activations …")
meta: Dict[str, dict] = {}
for batch_text in tqdm(batched(row_iter, cfg.batch_size),
                       total=cfg.n_rows // cfg.batch_size):
    toks = tok(batch_text, return_tensors="pt",
               padding=True, truncation=True, max_length=cfg.max_len).to(cfg.device)
    with torch.no_grad():
        outs = lm(**toks, output_hidden_states=True)

    # iterate SAEs
    for hook, sae in saes_raw.items():
        # figure out which hidden state slice to use
        if hook.startswith("layers."):
            layer_idx = int(hook.split(".")[1])
            h_state = outs.hidden_states[layer_idx + 1]  # (b, s, d)
        else:  # embedding table SAE
            h_state = outs.hidden_states[0]  # embedding stream

        # encode → activations
        acts = latent_acts(sae.encode(h_state))  # (b, s, latent_dim)
        topk = min(cfg.topk_per_token, latent_dim)
        vals, idxs = torch.topk(acts, topk, dim=-1)  # (b,s,k)

        for bi in range(acts.shape[0]):
            for ti in range(acts.shape[1]):
                if toks.attention_mask[bi, ti] == 0: continue
                tok_id = toks.input_ids[bi, ti].item()
                tok_str = tok.decode([tok_id])
                # 5-token context
                s0, s1 = max(0, ti - 5), min(acts.shape[1], ti + 6)
                ctx = tok.decode(toks.input_ids[bi, s0:s1])
                for latent, val in zip(idxs[bi, ti], vals[bi, ti]):
                    if val <= 0: continue
                    fkey = f"{hook}/{latent.item()}"
                    item = meta.setdefault(fkey, {"tokens": [], "contexts": [], "strengths": []})
                    item["tokens"].append(tok_str)
                    item["contexts"].append(ctx)
                    item["strengths"].append(float(val))

# ─────────────────── LABEL & TEXT INDEX ───────────────────────── #
print("Labelling features …")
all_feats = list(meta.keys())
labels = []
for fk in tqdm(all_feats):
    entry = meta[fk]
    toks = entry["tokens"][:cfg.examples_per_feat]
    ctxs = entry["contexts"][:cfg.examples_per_feat]
    lbl, desc = label_feature(toks, ctxs)
    entry["label"], entry["description"] = lbl, desc
    labels.append(lbl)

print("Building label-text FAISS index …")
text_index = faiss.IndexFlatL2(1536)
text_index.add(embed_text(labels))

# ─────────────── BUILD DECODER-VECTOR INDEX ───────────────────── #
print("Building decoder-vector FAISS index …")
vec_index = faiss.IndexFlatL2(d_model)
vec_mat = np.zeros((len(all_feats), d_model), dtype="float32")
for i, fk in enumerate(all_feats):
    h, l = fk.split("/")
    l = int(l)
    vec_mat[i] = decoder_weight(saes_raw[h])[l].numpy()
vec_index.add(vec_mat)

# ───────────────────── SAVE ARTIFACTS ─────────────────────────── #
faiss.write_index(text_index, str(cfg.workdir / cfg.text_index_name))
faiss.write_index(vec_index, str(cfg.workdir / cfg.vec_index_name))
(cfg.workdir / cfg.meta_name).write_bytes(pickle.dumps(meta))
print(f"Saved indexes + metadata to {cfg.workdir}")


# ───────────────────── SEARCH & GENERATE API ─────────────────── #
def search_by_text(query: str, k: int = 5):
    qvec = embed_text([query])
    D, I = text_index.search(qvec, k)
    return [{**meta[all_feats[i]], "feature": all_feats[i], "score": float(D[0][j])}
            for j, i in enumerate(I[0])]


def search_by_vector(prompt: str, layer: int | None = None, k: int = 5):
    toks = tok(prompt, return_tensors="pt").to(cfg.device)
    with torch.no_grad():
        outs = lm(**toks, output_hidden_states=True)
        if layer is None:
            h = outs.hidden_states[-1]
        else:
            h = outs.hidden_states[layer + 1]
        vec = h.mean(dim=1).squeeze().cpu().float().numpy()[None, :]
    D, I = vec_index.search(vec.astype("float32"), k)
    return [{**meta[all_feats[i]], "feature": all_feats[i], "score": float(D[0][j])}
            for j, i in enumerate(I[0])]


# ────────────────────── STEERING UTILITIES ───────────────────── #
def bias_hook(basis: torch.Tensor, strength: float):
    def fn(_mod, _inp, output): return output + strength * basis

    return fn


def latent_hook(sae_obj, latent: int, strength: float):
    W_dec = decoder_weight(sae_obj).to(cfg.device)  # (L,d)

    def fn(_mod, _inp, output):
        acts = sae_obj.encode(output)  # (b,s,L)
        acts[:, :, latent] += strength
        return torch.einsum("bsl,ld->bsd", acts, W_dec)  # decode

    return fn


@contextlib.contextmanager
def steer(feature_key: str, strength: float | None = None, mode: str = "bias"):
    """mode ∈ {'bias','latent'}"""
    if strength is None: strength = cfg.default_strength
    hook, latent = feature_key.split("/")
    latent = int(latent)
    sae_obj = saes_raw[hook]

    handles = []
    basis = None
    if mode == "bias":
        basis = decoder_weight(sae_obj)[latent].to(cfg.device)

    for name, module in lm.named_modules():
        if name == hook:
            if mode == "bias":
                handles.append(module.register_forward_hook(bias_hook(basis, strength)))
            else:
                handles.append(module.register_forward_hook(latent_hook(sae_obj, latent, strength)))
            break
    try:
        yield
    finally:
        for h in handles: h.remove()


def generate(prompt: str, feature_key: str | None = None, strength: float | None = None,
             mode: str = "bias", **gkw):
    gkw.setdefault("max_new_tokens", 120)
    gkw.setdefault("temperature", 0.8)
    toks = tok(prompt, return_tensors="pt").to(cfg.device)
    with steer(feature_key, strength, mode) if feature_key else contextlib.nullcontext():
        out_ids = lm.generate(**toks, **gkw)[0]
    return tok.decode(out_ids, skip_special_tokens=True)


# ────────────────────────── DEMO ─────────────────────────────── #
if __name__ == "__main__":
    term = "mathematics"
    print(f"\nTop semantic features for '{term}':")
    for r in search_by_text(term, 3):
        print(f"· {r['label']}: {r['description'][:60]}…")

    chosen = search_by_text(term, 1)[0]["feature"]
    print(f"\nGenerating with feature {chosen} boosted (latent mode):\n")
    print(generate("The scientist discovered", feature_key=chosen,
                   strength=8.0, mode="latent"))
