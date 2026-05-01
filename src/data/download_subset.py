"""Download a fixed bilingual HF streaming subset to local JSONL files.

Run from the project root, for example:

    python src/data/download_subset.py --config configs/train_topk.yaml \
        --output-dir data/local_corpus/fineweb_edu_100k \
        --en-train-docs 100000 --zh-train-docs 100000 \
        --en-val-docs 5000 --zh-val-docs 5000

The output is split by language so training can keep the same en_ratio mixing
logic without loading the full local corpus into memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_yaml, dump_yaml


@dataclass
class DatasetSpec:
    name: str
    subset: Optional[str]
    split: str
    text_col: str
    lang: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a fixed HF corpus subset to local JSONL files.")
    p.add_argument("--config", default="configs/train_topk.yaml", help="训练配置，用于读取 corpus 数据集设置")
    p.add_argument("--output-dir", default="data/local_corpus/fineweb_edu_subset", help="本地子集输出目录")
    p.add_argument("--en-train-docs", type=int, default=100000, help="英文训练文档数")
    p.add_argument("--zh-train-docs", type=int, default=100000, help="中文训练文档数")
    p.add_argument("--en-val-docs", type=int, default=5000, help="英文验证文档数")
    p.add_argument("--zh-val-docs", type=int, default=5000, help="中文验证文档数")
    p.add_argument("--min-chars", type=int, default=50, help="少于该字符数的文本跳过")
    p.add_argument("--max-chars", type=int, default=20000, help="单条文本最多保留字符数；<=0 表示不截断")
    p.add_argument("--shuffle-buffer", type=int, default=10000, help="HF streaming 近似 shuffle buffer；0 表示不 shuffle")
    p.add_argument("--streaming-batch-size", type=int, default=None, help="覆盖 corpus.streaming_batch_size")
    p.add_argument("--seed", type=int, default=None, help="覆盖 trainer.seed")
    p.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出 jsonl/manifest")
    return p.parse_args()


def iter_hf_text(spec: DatasetSpec, cache_dir: Optional[str], batch_size: int, shuffle_buffer: int, seed: int) -> Iterable[str]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        spec.name,
        spec.subset,
        split=spec.split,
        streaming=True,
        cache_dir=cache_dir,
        columns=[spec.text_col],
        batch_size=batch_size,
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    for row in ds:
        text = row.get(spec.text_col, "")
        if text:
            yield str(text)


def ensure_output_file(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在: {path}。如需覆盖，添加 --overwrite")


def write_lang_subset(
    spec: DatasetSpec,
    out_dir: Path,
    train_docs: int,
    val_docs: int,
    cache_dir: Optional[str],
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
    min_chars: int,
    max_chars: int,
    overwrite: bool,
) -> dict[str, object]:
    train_path = out_dir / f"train_{spec.lang}.jsonl"
    val_path = out_dir / f"val_{spec.lang}.jsonl"
    ensure_output_file(train_path, overwrite)
    ensure_output_file(val_path, overwrite)

    train_written = 0
    val_written = 0
    skipped = 0
    target = train_docs + val_docs
    started = time.time()

    pbar = tqdm(total=target, desc=f"download {spec.lang}", unit="doc")
    text_iter = iter_hf_text(spec, cache_dir, batch_size, shuffle_buffer, seed)
    with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
        for text in text_iter:
            text = text.strip()
            if len(text) < min_chars:
                skipped += 1
                continue
            if max_chars > 0:
                text = text[:max_chars]

            row = json.dumps({"text": text, "lang": spec.lang}, ensure_ascii=False)
            if val_written < val_docs:
                val_f.write(row + "\n")
                val_written += 1
            elif train_written < train_docs:
                train_f.write(row + "\n")
                train_written += 1
            else:
                break
            pbar.update(1)
    pbar.close()

    if train_written < train_docs or val_written < val_docs:
        raise RuntimeError(
            f"{spec.lang} 子集不足：train {train_written}/{train_docs}, val {val_written}/{val_docs}"
        )

    return {
        "source": asdict(spec),
        "train_path": str(train_path).replace("\\", "/"),
        "val_path": str(val_path).replace("\\", "/"),
        "train_docs": train_written,
        "val_docs": val_written,
        "skipped_docs": skipped,
        "elapsed_s": round(time.time() - started, 1),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    corpus = cfg["corpus"]
    trainer = cfg.get("trainer", {})
    seed = args.seed if args.seed is not None else trainer.get("seed", 0)
    batch_size = args.streaming_batch_size or corpus.get("streaming_batch_size", 1024)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    en_spec = DatasetSpec(
        name=corpus["en_name"],
        subset=corpus.get("en_subset"),
        split=corpus.get("en_split", "train"),
        text_col=corpus.get("en_text_col", "text"),
        lang="en",
    )
    zh_spec = DatasetSpec(
        name=corpus["zh_name"],
        subset=corpus.get("zh_subset"),
        split=corpus.get("zh_split", "train"),
        text_col=corpus.get("zh_text_col", "text"),
        lang="zh",
    )

    manifest_path = out_dir / "manifest.json"
    corpus_yaml_path = out_dir / "corpus_local.yaml"
    train_config_path = out_dir / "train_config_local.yaml"
    ensure_output_file(manifest_path, args.overwrite)
    ensure_output_file(corpus_yaml_path, args.overwrite)
    ensure_output_file(train_config_path, args.overwrite)

    en_result = write_lang_subset(
        en_spec,
        out_dir,
        args.en_train_docs,
        args.en_val_docs,
        corpus.get("cache_dir"),
        batch_size,
        args.shuffle_buffer,
        seed,
        args.min_chars,
        args.max_chars,
        args.overwrite,
    )
    zh_result = write_lang_subset(
        zh_spec,
        out_dir,
        args.zh_train_docs,
        args.zh_val_docs,
        corpus.get("cache_dir"),
        batch_size,
        args.shuffle_buffer,
        seed + 1,
        args.min_chars,
        args.max_chars,
        args.overwrite,
    )

    local_corpus_cfg = {
        "local_en_train_path": en_result["train_path"],
        "local_zh_train_path": zh_result["train_path"],
        "local_en_val_path": en_result["val_path"],
        "local_zh_val_path": zh_result["val_path"],
        "local_text_col": "text",
        "en_ratio": corpus.get("en_ratio", 0.5),
        "cache_dir": corpus.get("cache_dir", "data/cache"),
    }
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(Path(args.config)).replace("\\", "/"),
        "output_dir": str(out_dir).replace("\\", "/"),
        "seed": seed,
        "shuffle_buffer": args.shuffle_buffer,
        "streaming_batch_size": batch_size,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "en": en_result,
        "zh": zh_result,
        "local_corpus_config": local_corpus_cfg,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_yaml({"corpus": local_corpus_cfg}, corpus_yaml_path)
    local_train_cfg = dict(cfg)
    local_train_cfg["corpus"] = local_corpus_cfg
    dump_yaml(local_train_cfg, train_config_path)

    print(f"[download_subset] 已写入: {out_dir}")
    print(f"[download_subset] 本地 corpus 配置片段: {corpus_yaml_path}")
    print(f"[download_subset] 可直接训练的配置: {train_config_path}")


if __name__ == "__main__":
    main()
