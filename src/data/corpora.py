"""中英混合训练语料：HuggingFace ``datasets`` streaming 加载并交错混合。

- 英文：默认 ``HuggingFaceFW/fineweb-edu``（高质量教育性网页文本）
- 中文：默认 ``opencsg/chinese-fineweb-edu``
- 也可在 yaml 中切换为本地小数据（如 ``wikitext``）做快速验证。

设计
----
返回一个 ``IterableDataset``，按设定 ``ratio`` 概率从两个流随机抽 sample，
每个 yield 出一条原始字符串文本。tokenize 由下游 ``ActivationStore`` 完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import json
import random


def _lazy_import_datasets():
    """延迟导入 datasets：本地纯 .txt 路径不需要它，避免环境依赖问题。"""
    from datasets import load_dataset, IterableDataset  # type: ignore
    return load_dataset, IterableDataset


@dataclass
class CorpusConfig:
    en_name: str = "HuggingFaceFW/fineweb-edu"
    en_subset: Optional[str] = "sample-10BT"
    en_split: str = "train"
    en_text_col: str = "text"

    zh_name: str = "opencsg/chinese-fineweb-edu"
    zh_subset: Optional[str] = None
    zh_split: str = "train"
    zh_text_col: str = "text"

    en_ratio: float = 0.5  # 抽英文的概率（剩下抽中文）
    seed: int = 0
    cache_dir: Optional[str] = "data/cache"
    streaming: bool = True
    streaming_batch_size: int = 1024


def _load(name: str, subset: Optional[str], split: str, text_col: str, cfg: CorpusConfig):
    load_dataset, _ = _lazy_import_datasets()
    return load_dataset(
        name,
        subset,
        split=split,
        streaming=cfg.streaming,
        cache_dir=cfg.cache_dir,
        columns=[text_col],
        batch_size=cfg.streaming_batch_size,
    )


class BilingualTextStream:
    """按 ratio 从中英两个 streaming dataset 交错采样文本字符串。"""

    def __init__(self, cfg: CorpusConfig):
        self.cfg = cfg
        self._en = _load(cfg.en_name, cfg.en_subset, cfg.en_split, cfg.en_text_col, cfg)
        self._zh = _load(cfg.zh_name, cfg.zh_subset, cfg.zh_split, cfg.zh_text_col, cfg)
        self._en_iter = iter(self._en)
        self._zh_iter = iter(self._zh)
        self._rng = random.Random(cfg.seed)

    def __iter__(self) -> Iterator[str]:
        return self

    def _next_text(self, it, col: str, name: str) -> str:
        try:
            row = next(it)
        except StopIteration:
            # 流尽了重启
            if name == "en":
                self._en = _load(self.cfg.en_name, self.cfg.en_subset, self.cfg.en_split, self.cfg.en_text_col, self.cfg)
                self._en_iter = iter(self._en)
                row = next(self._en_iter)
            else:
                self._zh = _load(self.cfg.zh_name, self.cfg.zh_subset, self.cfg.zh_split, self.cfg.zh_text_col, self.cfg)
                self._zh_iter = iter(self._zh)
                row = next(self._zh_iter)
        return row[col]

    def __next__(self) -> str:
        if self._rng.random() < self.cfg.en_ratio:
            return self._next_text(self._en_iter, self.cfg.en_text_col, "en")
        else:
            return self._next_text(self._zh_iter, self.cfg.zh_text_col, "zh")


# ---- 离线/本地小数据回退（无法访问 HF Hub 时用）----

class WikitextStream:
    """``wikitext-2-raw-v1`` 的轻量 wrapper，便于离线/快速验证。需 datasets 库。"""

    def __init__(self, cache_dir: Optional[str] = "data/cache"):
        load_dataset, _ = _lazy_import_datasets()
        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="train",
            cache_dir=cache_dir,
        )
        self._rows = [r["text"] for r in ds if r["text"].strip()]

    def __iter__(self):
        i = 0
        while True:
            yield self._rows[i % len(self._rows)]
            i += 1


class LocalTextFileStream:
    """从本地 .txt / .jsonl 一行一文本循环 yield，**不依赖 datasets 库**。

    用于 smoke test / 离线场景。
    """

    def __init__(self, path: str | Path = "data/smoke_text.txt", text_col: str = "text"):
        self.path = Path(path)
        self.text_col = text_col
        if not self.path.exists():
            raise FileNotFoundError(f"本地文本不存在: {self.path}")
        self._fh = None
        self._open()

    def _open(self) -> None:
        if self._fh is not None:
            self._fh.close()
        self._fh = self.path.open("r", encoding="utf-8")

    def _parse_line(self, line: str) -> str:
        line = line.strip()
        if not line:
            return ""
        if self.path.suffix.lower() == ".jsonl":
            row = json.loads(line)
            return str(row.get(self.text_col, ""))
        return line

    def __iter__(self):
        return self

    def __next__(self) -> str:
        # 最多扫两轮，避免空文件导致死循环。
        for _ in range(2):
            assert self._fh is not None
            for line in self._fh:
                text = self._parse_line(line)
                if text and text.strip():
                    return text
            self._open()
        raise RuntimeError(f"{self.path} 中无可用文本行。")


class LocalBilingualTextStream:
    """从本地中英文文件按 ratio 混合采样，文件可为 .txt 或 .jsonl。"""

    def __init__(
        self,
        en_path: str | Path,
        zh_path: str | Path,
        en_ratio: float = 0.5,
        seed: int = 0,
        text_col: str = "text",
    ):
        self._en = LocalTextFileStream(en_path, text_col=text_col)
        self._zh = LocalTextFileStream(zh_path, text_col=text_col)
        self._rng = random.Random(seed)
        self.en_ratio = en_ratio

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self._rng.random() < self.en_ratio:
            return next(self._en)
        return next(self._zh)


def build_text_stream(cfg: CorpusConfig, fallback_local: bool = True):
    """构建文本流。HF 失败时根据 ``fallback_local`` 决定是否回落到 wikitext。"""
    try:
        return BilingualTextStream(cfg)
    except Exception as e:
        if not fallback_local:
            raise
        print(f"[corpora] HF 流失败 ({type(e).__name__}: {e})，回退到 wikitext-2 本地数据。")
        return WikitextStream(cache_dir=cfg.cache_dir)
