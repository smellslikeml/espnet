"""Tests for the language-identification (LID) token prepending integration.

Exercises the wiring added to ``CommonPreprocessor._text_process`` (the
existing call site) as well as the standalone helpers, covering the explicit
LID-token strategy from arXiv:2606.17820.
"""

import numpy as np
import pytest

from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.train.lang_id_prompt import prepend_lang_token, resolve_lang_symbol
from espnet2.train.preprocessor import CommonPreprocessor

# A small bilingual word vocabulary: blank/unk + two language tokens + words.
TOKEN_LIST = [
    "<blank>",
    "<unk>",
    "<mr>",
    "<kn>",
    "hello",
    "world",
    "foo",
]


def _build_preprocessor(**kwargs):
    return CommonPreprocessor(
        train=False,
        token_type="word",
        token_list=TOKEN_LIST,
        unk_symbol="<unk>",
        **kwargs,
    )


@pytest.mark.execution_timeout(30)
def test_text_process_prepends_per_utterance_lang_token():
    preprocessor = _build_preprocessor(use_lang_id_prompt=True)

    baseline = preprocessor("utt0", {"text": "hello world"})["text"]
    out = preprocessor("utt1", {"text": "hello world", "lang": "<mr>"})["text"]

    mr_id = TOKEN_LIST.index("<mr>")
    # The language token is prepended ahead of the original transcript ids.
    assert out[0] == mr_id
    assert list(out[1:]) == list(baseline)
    assert out.dtype == np.int64


@pytest.mark.execution_timeout(30)
def test_text_process_uses_fixed_lang_token():
    preprocessor = _build_preprocessor(use_lang_id_prompt=True, lang_token="<kn>")

    out = preprocessor("utt1", {"text": "foo"})["text"]

    assert out[0] == TOKEN_LIST.index("<kn>")
    assert out[1] == TOKEN_LIST.index("foo")


@pytest.mark.execution_timeout(30)
def test_per_utterance_overrides_fixed_lang_token():
    preprocessor = _build_preprocessor(use_lang_id_prompt=True, lang_token="<kn>")

    out = preprocessor("utt1", {"text": "foo", "lang": "<mr>"})["text"]

    assert out[0] == TOKEN_LIST.index("<mr>")


@pytest.mark.execution_timeout(30)
def test_disabled_by_default_leaves_text_untouched():
    plain = _build_preprocessor()
    out = plain("utt1", {"text": "hello world", "lang": "<mr>"})["text"]

    expected = plain.token_id_converter.tokens2ids(["hello", "world"])
    assert list(out) == expected


@pytest.mark.execution_timeout(30)
def test_unknown_lang_token_raises():
    converter = TokenIDConverter(token_list=TOKEN_LIST, unk_symbol="<unk>")
    with pytest.raises(KeyError):
        prepend_lang_token([4, 5], "<xx>", converter)


def test_resolve_lang_symbol_precedence():
    assert resolve_lang_symbol({"lang": "<mr>"}, "<kn>") == "<mr>"
    assert resolve_lang_symbol({}, "<kn>") == "<kn>"
    assert resolve_lang_symbol({}) is None
    # An array-valued column (already tokenized) is ignored.
    assert resolve_lang_symbol({"lang": np.array([1])}, "<kn>") == "<kn>"
