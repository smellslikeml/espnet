"""Prepend an explicit language-identification (LID) token to transcripts.

This implements the explicit LID-token strategy from "Improving low-resource
ASR using bilingual fine-tuning with language identification: a cross-linguistic
evaluation" (arXiv:2606.17820). The paper shows that, when bilingually
fine-tuning an ASR model on a low-resource language together with a related
higher-resource language, pre-pending a language-identification token to each
transcript lets the model condition on (or jointly predict) the language and
improves word error rate -- especially when language identification is hard.

ESPnet already supports a Whisper-specific prompt path (``use_lang_prompt`` in
``CommonPreprocessor``), but that path depends on the HuggingFace Whisper
tokenizer. These helpers bring the same LID-token contract to the standard
``TokenIDConverter`` path used by the toolkit's low-resource ASR recipes, so
the technique works for any ``token_type`` (bpe / char / word / phn).
"""

from typing import Dict, List, Optional, Union

import numpy as np

from espnet2.text.token_id_converter import TokenIDConverter

# Default name of the per-utterance data column carrying the language symbol.
DEFAULT_LANG_TOKEN_NAME = "lang"


def resolve_lang_symbol(
    data: Dict[str, Union[str, np.ndarray]],
    lang_token: Optional[str] = None,
    lang_token_name: str = DEFAULT_LANG_TOKEN_NAME,
) -> Optional[str]:
    """Return the LID symbol to pre-pend for this utterance, or ``None``.

    A per-utterance language column (``data[lang_token_name]``) takes
    precedence over the fixed ``lang_token`` configured on the preprocessor,
    so a bilingual mix can carry one language symbol per utterance while a
    monolingual shard can rely on a single fixed token.
    """
    symbol = data.get(lang_token_name)
    if isinstance(symbol, str) and symbol:
        return symbol
    if lang_token:
        return lang_token
    return None


def prepend_lang_token(
    text_ints: List[int],
    symbol: Optional[str],
    token_id_converter: TokenIDConverter,
) -> List[int]:
    """Pre-pend the integer id of ``symbol`` to a token-id sequence.

    ``symbol`` must already be present in the token list. An LID token that
    silently fell back to ``<unk>`` would make every language collapse to the
    same id and defeat the purpose of language identification, so an unknown
    symbol is reported as an error rather than mapped to ``<unk>``.
    """
    if symbol is None:
        return text_ints
    lang_id = token_id_converter.token2id.get(symbol)
    if lang_id is None:
        raise KeyError(
            f"Language-identification token '{symbol}' is not in the token "
            "list. Add it (e.g. as a non-linguistic symbol) so it maps to a "
            "dedicated id instead of <unk>."
        )
    return [lang_id] + list(text_ints)
