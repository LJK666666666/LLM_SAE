"""维护 ``{results_root}/overall_config_metrics.csv`` 与 .json，每次实验完成后追加一行。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_RESULTS_ROOT = Path("results")


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def append_record(record: dict[str, Any], results_root: str | Path = DEFAULT_RESULTS_ROOT) -> None:
    """``record`` 应当是一个扁平或嵌套 dict，自动扁平化后追加。"""
    root = Path(results_root)
    csv_path = root / "overall_config_metrics.csv"
    json_path = root / "overall_config_metrics.json"

    root.mkdir(parents=True, exist_ok=True)
    flat = _flatten(record)

    # JSON：list of dicts，整体读改写
    if json_path.exists():
        all_records = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        all_records = []
    # 去重：若同 exp_dir 已存在则替换
    exp_dir_key = flat.get("exp_dir")
    if exp_dir_key:
        all_records = [r for r in all_records if r.get("exp_dir") != exp_dir_key]
    all_records.append(flat)
    json_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV：用 pandas 重写以保持列对齐
    df = pd.DataFrame(all_records)
    df.to_csv(csv_path, index=False, encoding="utf-8")
