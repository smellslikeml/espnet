"""Tests for CompressKV KV-cache compression.

Covers the standalone retention policy in
``espnet2/speechlm/model/speechlm/lm/kv_cache_compression.py`` plus the wiring
into ``ParallelLLM.inference_segment`` (``lm/parallel.py``). The whole file is
skipped (via the ``model/conftest.py`` ``collect_ignore_glob``) when
``liger_kernel`` is not importable, matching ``test_parallel.py``.
"""

import torch

from espnet2.speechlm.model.speechlm.lm.kv_cache_compression import (
    compress_kv_cache,
    select_retained_indices,
    token_importance,
)


# ---------------------------------------------------------------------------
# Minimal DynamicCache stand-in: just enough of the interface that
# kv_cache_compression duck-types against (``layers``, ``update``).
# ---------------------------------------------------------------------------
class _FakeLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self):
        self.layers = []

    def update(self, key_states=None, value_states=None, layer_idx=None):
        while len(self.layers) <= layer_idx:
            self.layers.append(None)
        self.layers[layer_idx] = _FakeLayer(key_states, value_states)
        return key_states, value_states


def _make_cache(num_layers=2, batch=1, heads=2, seq_len=40, dim=8):
    cache = _FakeCache()
    for idx in range(num_layers):
        keys = torch.randn(batch, heads, seq_len, dim)
        values = torch.randn(batch, heads, seq_len, dim)
        cache.update(key_states=keys, value_states=values, layer_idx=idx)
    return cache


# ---------------------------------------------------------------------------
# select_retained_indices
# ---------------------------------------------------------------------------
class TestSelectRetainedIndices:
    def test_no_compression_when_budget_covers_sequence(self):
        importance = torch.arange(10).float()
        keep = select_retained_indices(importance, budget=10, num_sink=2, num_recent=2)
        assert torch.equal(keep, torch.arange(10))

    def test_respects_budget(self):
        importance = torch.randn(50)
        keep = select_retained_indices(importance, budget=12, num_sink=4, num_recent=4)
        assert keep.shape[0] == 12

    def test_keeps_sink_and_recent(self):
        importance = torch.zeros(30)  # middle has no signal; structure must win
        keep = select_retained_indices(importance, budget=8, num_sink=3, num_recent=3)
        keep_set = set(keep.tolist())
        assert {0, 1, 2}.issubset(keep_set)  # sink
        assert {27, 28, 29}.issubset(keep_set)  # recent
        assert keep.shape[0] == 8

    def test_prefers_high_importance_middle_tokens(self):
        importance = torch.zeros(20)
        importance[10] = 5.0  # a salient mid-context token
        keep = select_retained_indices(importance, budget=5, num_sink=2, num_recent=2)
        assert 10 in keep.tolist()

    def test_sorted_and_unique(self):
        importance = torch.randn(40)
        keep = select_retained_indices(importance, budget=15, num_sink=4, num_recent=4)
        assert torch.equal(keep, keep.sort().values)
        assert keep.unique().shape[0] == keep.shape[0]


# ---------------------------------------------------------------------------
# token_importance — Semantic Retrieval Head restriction
# ---------------------------------------------------------------------------
class TestTokenImportance:
    def test_shape(self):
        cache = _make_cache(seq_len=24)
        imp = token_importance(cache)
        assert imp.shape == (24,)

    def test_retrieval_head_subset_changes_scores(self):
        cache = _make_cache(heads=4, seq_len=24)
        full = token_importance(cache)
        subset = token_importance(cache, retrieval_head_ids=[0])
        # Restricting to a single Semantic Retrieval Head yields a different
        # token ranking than pooling over all heads.
        assert not torch.allclose(full, subset)

    def test_out_of_range_heads_ignored(self):
        cache = _make_cache(heads=2, seq_len=16)
        # Heads >= num_kv_heads are dropped; falls back to all available heads.
        imp = token_importance(cache, retrieval_head_ids=[99])
        assert imp.shape == (16,)
        assert torch.isfinite(imp).all()


# ---------------------------------------------------------------------------
# compress_kv_cache
# ---------------------------------------------------------------------------
class TestCompressKVCache:
    def test_prunes_to_budget(self):
        cache = _make_cache(num_layers=3, seq_len=40)
        out = compress_kv_cache(cache, budget=10, num_sink=2, num_recent=4)
        for layer in out.layers:
            assert layer.keys.shape[2] == 10
            assert layer.values.shape[2] == 10

    def test_all_layers_share_length(self):
        cache = _make_cache(num_layers=4, seq_len=32)
        out = compress_kv_cache(cache, budget=8)
        lengths = {layer.keys.shape[2] for layer in out.layers}
        assert lengths == {8}

    def test_noop_when_within_budget(self):
        cache = _make_cache(seq_len=12)
        out = compress_kv_cache(cache, budget=100)
        assert out is cache

    def test_noop_when_budget_none(self):
        cache = _make_cache(seq_len=12)
        out = compress_kv_cache(cache, budget=None)
        assert out is cache

    def test_retained_kv_pairs_are_original_columns(self):
        cache = _make_cache(num_layers=1, seq_len=20)
        original = cache.layers[0].keys.clone()
        out = compress_kv_cache(cache, budget=6, num_sink=2, num_recent=2)
        # Sink columns are retained verbatim (no re-encoding of KV pairs).
        assert torch.equal(out.layers[0].keys[:, :, 0], original[:, :, 0])


# ---------------------------------------------------------------------------
# Integration: inference_segment compresses the prompt cache when configured.
# Imports the real (non-new) call site module to exercise the wiring edit.
# ---------------------------------------------------------------------------
class TestInferenceSegmentWiring:
    def _build_model(self):
        import numpy as np
        import torch.nn as nn

        from espnet2.speechlm.model.speechlm.multimodal_io.abs_io import AbsIO

        class _IO(AbsIO):
            def __init__(self):
                super().__init__(modality="text", is_discrete=True)

            def num_stream(self):
                return 1

            def get_vocabulary(self):
                return [f"text_tok_{i}" for i in range(100)]

            def get_stream_interval(self):
                return [(0, 100)]

            def get_stream_weight(self):
                return [1.0]

            def preprocess(self, data):
                return (
                    np.zeros((3, 1), dtype=np.int64),
                    None,
                    np.ones((3, 1), dtype=np.float32),
                )

            def find_length(self, data):
                return 3

            def copy_for_worker(self):
                return self

            def feature_dim(self):
                return None

            def encode_batch(self, feats, lengths):
                return torch.zeros(feats.shape[0], 3, 1, dtype=torch.long)

            def decode_batch(self, codes, lengths):
                return [None] * codes.shape[0]

            def dummy_forward(self, ref_tensor=None):
                return torch.zeros(1, requires_grad=True)

        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )

        text_io = _IO()
        multimodal_io = nn.ModuleDict({"text": text_io})
        vocab = ["<|pad|>", "<|eos|>", "<|eot|>"]
        while len(vocab) < 256:
            vocab.append(f"<|unused_{len(vocab)}|>")
        text_start = len(vocab)
        vocab.extend(text_io.get_vocabulary())
        vocab_meta = {
            "vocab": vocab,
            "vocab_intervals": {
                "special_token": [(0, 256)],
                "text": [(text_start, text_start + 100)],
            },
            "vocab_weight": torch.ones(len(vocab)),
            "vocab_size": len(vocab),
            "mm_start": text_start,
            "mm_end": len(vocab),
            "text_start": text_start,
            "text_end": text_start + 100,
            "num_stream": 1,
        }
        cls = build_parallel_hf_class("mock-model")
        return cls.from_pretrained(
            "mock-model", multimodal_io=multimodal_io, vocab_meta=vocab_meta
        )

    def test_compress_called_from_inference_segment(self, monkeypatch):
        model = self._build_model()

        captured = {}

        def _fake_compress(cache, **kwargs):
            captured.update(kwargs)
            captured["called"] = True
            return cache

        # Patch the symbol bound inside parallel.py's module namespace.
        monkeypatch.setattr(
            "espnet2.speechlm.model.speechlm.lm.parallel.compress_kv_cache",
            _fake_compress,
        )

        cache = _make_cache(num_layers=1, seq_len=30)
        config = {
            "max_step": 1,
            "min_step": 1,
            "temperature": 0,
            "topk": 1,
            "kv_cache_budget": 8,
            "kv_num_sink": 2,
            "kv_num_recent": 4,
            "kv_retrieval_heads": [0],
        }
        mask = torch.zeros(1, 1, 1, len(model.vocab)).bool()
        prev_token = torch.zeros(1, 1, 1, dtype=torch.long)

        model.eos_token_id = 1
        model.eot_token_id = 2
        model.inference_segment(
            config=config, cache=cache, prev_token=prev_token, mask=mask
        )

        assert captured.get("called") is True
        assert captured["budget"] == 8
        assert captured["num_sink"] == 2
        assert captured["num_recent"] == 4
        assert captured["retrieval_head_ids"] == [0]

    def test_compression_skipped_without_budget(self, monkeypatch):
        model = self._build_model()

        captured = {"called": False}

        def _fake_compress(cache, **kwargs):
            captured["called"] = True
            return cache

        monkeypatch.setattr(
            "espnet2.speechlm.model.speechlm.lm.parallel.compress_kv_cache",
            _fake_compress,
        )

        cache = _make_cache(num_layers=1, seq_len=30)
        config = {"max_step": 1, "min_step": 1, "temperature": 0, "topk": 1}
        mask = torch.zeros(1, 1, 1, len(model.vocab)).bool()
        prev_token = torch.zeros(1, 1, 1, dtype=torch.long)

        model.eos_token_id = 1
        model.eot_token_id = 2
        model.inference_segment(
            config=config, cache=cache, prev_token=prev_token, mask=mask
        )

        assert captured["called"] is False
