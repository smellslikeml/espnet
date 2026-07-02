# Copyright 2025 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Semantic-retrieval-guided KV-cache compression for SpeechLM inference.

Adapted from CompressKV (Semantic-Retrieval-Guided KV-Cache Compression for
Resource-Efficient Long-Context LLM Inference, https://arxiv.org/abs/2606.24467).

Long-context decoding is dominated by the KV cache. Rather than scoring tokens
with every attention head, CompressKV argues that only a small set of
*Semantic Retrieval Heads* (SRHs) actually track which prompt tokens matter,
and that the attention sink (initial tokens) plus the recent window must always
be retained. This module ports that retention policy so it can prune the
``DynamicCache`` threaded through ``ParallelLLM.inference_segment`` to a fixed
token budget while preserving those structurally critical positions.

Scope notes (intentionally not ported):
  * Token importance here is the per-token key-vector L2 norm pooled over the
    selected heads, used as a cache-only proxy for SRH attention mass (the HF
    forward in this repo does not expose attention maps). The faithful part is
    *which* heads contribute: passing ``retrieval_head_ids`` restricts scoring
    to the SRH subset, the paper's central thesis.
  * Layer-wise budget allocation from offline eviction-error estimates is out
    of scope; a single budget is applied uniformly so the cache stays
    rectangular and position bookkeeping for appended tokens is unaffected.

The retention policy shares CompressKV's caveat with every KV-eviction method:
positions are not re-encoded after pruning, so keep ``num_recent`` large enough
to cover the local window the model relies on. Compression is opt-in and a
no-op whenever the sequence already fits the budget, so default generation is
unchanged.
"""

import torch


def token_importance(cache, retrieval_head_ids=None):
    """Score every cached token by pooled key magnitude across layers.

    Args:
        cache: a ``DynamicCache``-like object exposing ``layers``, where each
            layer has ``keys``/``values`` tensors shaped
            ``(batch, num_kv_heads, seq_len, head_dim)``.
        retrieval_head_ids: optional iterable of KV-head indices (the Semantic
            Retrieval Heads). When given, only these heads contribute to the
            score; out-of-range ids are ignored. ``None`` uses all heads.

    Returns:
        A 1-D tensor of length ``seq_len`` with per-token importance, or
        ``None`` when the cache is empty.
    """
    importance = None
    for layer in cache.layers:
        keys = layer.keys  # (batch, num_kv_heads, seq_len, head_dim)
        if retrieval_head_ids is not None:
            head_index = torch.as_tensor(
                list(retrieval_head_ids), dtype=torch.long, device=keys.device
            )
            head_index = head_index[head_index < keys.shape[1]]
            if head_index.numel() > 0:
                keys = keys.index_select(1, head_index)

        # L2 norm of each token's key vector, pooled over batch and heads.
        layer_score = keys.float().norm(dim=-1).mean(dim=(0, 1))  # (seq_len,)
        # Normalize per layer so deep/shallow layers contribute on equal footing.
        layer_score = layer_score / (layer_score.amax() + 1e-6)
        importance = layer_score if importance is None else importance + layer_score
    return importance


def select_retained_indices(importance, budget, num_sink, num_recent):
    """Pick the token positions to keep under a fixed budget.

    Always retains the first ``num_sink`` (attention sink) and last
    ``num_recent`` (local window) tokens, then fills the remaining budget with
    the highest-importance tokens from the middle region.

    Args:
        importance: 1-D per-token importance tensor.
        budget: maximum number of tokens to retain.
        num_sink: number of leading tokens always kept.
        num_recent: number of trailing tokens always kept.

    Returns:
        A sorted 1-D ``LongTensor`` of retained positions. When ``budget``
        covers the whole sequence, all positions are returned.
    """
    seq_len = int(importance.shape[0])
    device = importance.device
    if budget is None or budget >= seq_len:
        return torch.arange(seq_len, device=device)
    budget = max(0, int(budget))

    keep_sink = min(max(0, int(num_sink)), budget, seq_len)
    keep_recent = min(max(0, int(num_recent)), budget - keep_sink, seq_len - keep_sink)
    middle_budget = budget - keep_sink - keep_recent

    sink = torch.arange(keep_sink, device=device)
    recent = torch.arange(seq_len - keep_recent, seq_len, device=device)

    selected = [sink, recent]
    mid_lo, mid_hi = keep_sink, seq_len - keep_recent
    if middle_budget > 0 and mid_hi > mid_lo:
        mid_scores = importance[mid_lo:mid_hi]
        top = torch.topk(mid_scores, min(middle_budget, mid_scores.shape[0])).indices
        selected.append(top + mid_lo)

    retained = torch.cat(selected).unique()
    return retained.sort().values


def compress_kv_cache(
    cache,
    budget,
    num_sink=4,
    num_recent=64,
    retrieval_head_ids=None,
):
    """Prune a KV cache to ``budget`` tokens using the CompressKV policy.

    A single retained-index set is computed from importance aggregated across
    layers and applied to every layer, so all layers (and, under CFG, both the
    conditional and unconditional batch halves) stay aligned. Returns the cache
    unchanged when it already fits the budget.

    Args:
        cache: ``DynamicCache``-like object (see :func:`token_importance`).
        budget: target number of retained tokens. ``None`` disables pruning.
        num_sink: leading tokens always kept (attention sink).
        num_recent: trailing tokens always kept (local window).
        retrieval_head_ids: optional Semantic Retrieval Head indices to restrict
            importance scoring to.

    Returns:
        A new cache of the same type holding the retained KV pairs (or the
        original cache when no compression is needed).
    """
    layers = getattr(cache, "layers", None)
    if not layers:
        return cache
    seq_len = layers[0].keys.shape[2]
    if budget is None or budget >= seq_len:
        return cache

    importance = token_importance(cache, retrieval_head_ids=retrieval_head_ids)
    keep = select_retained_indices(importance, budget, num_sink, num_recent)

    compressed = type(cache)()
    for idx, layer in enumerate(layers):
        layer_keep = keep.to(layer.keys.device)
        compressed.update(
            key_states=layer.keys.index_select(2, layer_keep),
            value_states=layer.values.index_select(2, layer_keep),
            layer_idx=idx,
        )
    return compressed
