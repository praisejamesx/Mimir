"""t5_targets.py - Semantic word targets for the SigLIP objective.

Recipe (d'Ascoli et al. 2024 / Jayalath et al. 2025a): predict T5 middle-layer
word embeddings with a SigLIP-style contrastive loss instead of classifying
one-hot labels. Semantics give rare words meaning for free (a rare word's
embedding sits near related frequent words), which is exactly what balanced
Top-10 rewards.

Model: t5-small (6 encoder layers, d=512, ~230MB download). Middle layer =
layer index 2. Each vocab word is encoded alone; the target is the mean over
its token positions. 100 words take ~1 min on GPU, a few min on CPU.

Built ON KAGGLE (vocabulary order must come from pnpl.load_vocabulary there
-- it defines the submission columns). Vectors are cached next to this file
(t5_vocab.pt); rebuilds never re-download. All HF imports are lazy so this
module imports anywhere (no transformers needed to read the cache).

If the build fails for any reason, callers must fall back to the supcon
objective -- never crash a run over targets.
"""

from __future__ import annotations

from pathlib import Path

import torch

CACHE_NAME = "t5_vocab.pt"
T5_MODEL = "t5-small"
T5_LAYER = 2          # middle layer of 6
T5_DIM = 512


def cache_path(around: str | Path = None) -> Path:
    base = Path(around).resolve().parent if around else Path(__file__).resolve().parent
    return base / CACHE_NAME


def load_cached(cache: Path = None):
    """Load cached (primary, moses) T5 matrices or return None."""
    path = Path(cache) if cache else cache_path()
    if not path.exists():
        return None
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
        tp = torch.as_tensor(obj["primary"], dtype=torch.float32)
        tm = torch.as_tensor(obj["moses"], dtype=torch.float32)
        if tp.shape[1] == T5_DIM and tm.shape[1] == T5_DIM and len(tp) == len(tm) == 50:
            return tp, tm
    except Exception:
        pass
    return None


def build_t5_matrix(words: list[str], model_name: str = T5_MODEL,
                    layer: int = T5_LAYER) -> "torch.Tensor":
    """Encode words with T5 encoder layer `layer`, mean over token positions."""
    import torch as _torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name).encoder
    mdl.eval()
    vecs = []
    with _torch.no_grad():
        for w in words:
            ids = tok(w, return_tensors="pt").input_ids
            out = mdl(ids, output_hidden_states=True)
            h = out.hidden_states[layer].squeeze(0)      # (tokens, 512)
            vecs.append(h.mean(dim=0))
    mat = _torch.stack(vecs).float()
    return mat / mat.norm(dim=1, keepdim=True).clamp(min=1e-9)


def ensure_t5_vocab(primary_words: list[str], moses_words: list[str],
                    cache: Path = None):
    """Return L2-normalized (primary, moses) T5 matrices (50, 512) each.

    Uses the cache when valid, else builds via HF (needs transformers +
    sentencepiece + network) and caches. Raises RuntimeError on failure so
    callers can fall back to supcon.
    """
    path = Path(cache) if cache else cache_path()
    hit = load_cached(path)
    if hit is not None:
        print(f"[t5] cache hit: {path}")
        return hit
    print(f"[t5] building {T5_MODEL} layer-{T5_LAYER} targets "
          f"({len(primary_words)}+{len(moses_words)} words) ...")
    try:
        tp = build_t5_matrix(list(primary_words))
        tm = build_t5_matrix(list(moses_words))
    except Exception as exc:
        raise RuntimeError(f"t5 target build failed ({type(exc).__name__}: "
                           f"{exc}); fall back to supcon") from exc
    torch.save({"primary": tp, "moses": tm, "model": T5_MODEL,
                "layer": T5_LAYER}, str(path))
    print(f"[t5] cached: {path}")
    return tp, tm


if __name__ == "__main__":
    # Offline self-test with fake vocabularies (no HF, no pnpl).
    import torch.nn.functional as F
    torch.manual_seed(0)
    fake_p = [f"word{i}" for i in range(50)]
    fake_m = [f"mword{i}" for i in range(50)]
    assert load_cached(Path("/nonexistent/t5_vocab.pt")) is None
    # Geometry sanity on random stand-ins: distinct rows, unit norm.
    stand = F.normalize(torch.randn(50, T5_DIM), dim=1)
    assert abs(float(stand.norm(dim=1).mean()) - 1.0) < 1e-5
    off = (stand @ stand.T)[~torch.eye(50, dtype=torch.bool)]
    assert float(off.abs().mean()) < 0.2, "random 512-dim rows should be ~orthogonal"
    print(f"t5_targets OK: cache-miss path clean, stand-in off-diag "
          f"|cos|={float(off.abs().mean()):.3f}")
