"""Shared fixtures for the data-engine tests (tiny config: vocab 512, 64 specials)."""

from __future__ import annotations

import pytest

from m63_omni.config import tiny
from m63_omni.data.tokenizer import ByteBPETokenizer, train_bpe

TRAIN_TEXTS = [
    "The quick brown fox jumps over the lazy dog. The dog sleeps in the sun.",
    "Data quality is the largest program risk because the architecture cannot compensate for poor alignment.",
    "Audio and video chunks are interleaved in timestamp order rather than concatenated.",
    "def pack(docs, seq_len):\n    return [d for d in docs if len(d) <= seq_len]\n",
    "Every batch can be reproduced from a manifest containing source versions, transformations, seeds, and ordering.",
    "数据质量是最大的项目风险。音频和视频按时间戳交错。",
    "データ品質は最大のリスクです。音声と映像はタイムスタンプ順に並べます。",
    "데이터 품질은 가장 큰 위험입니다.",
    "Качество данных — главный риск программы.",
    "La qualité des données est le plus grand risque du programme.",
] * 4


@pytest.fixture(scope="session")
def tiny_cfg():
    return tiny()


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_cfg) -> ByteBPETokenizer:
    return train_bpe(TRAIN_TEXTS, tiny_cfg.model.vocab_size, tiny_cfg.model.special_tokens)
