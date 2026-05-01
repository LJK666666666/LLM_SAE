"""加载 yaml 配置 + 命令行参数合并。

约定（CLAUDE.md 规则 17）：
- 高频调整参数 → 命令行
- 低频参数 → yaml
- 命令行参数若提供，覆盖 yaml 中同名键。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_cli_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """命令行非 None 值覆盖 yaml；嵌套点路径如 ``sae.k`` 也支持。"""
    out = dict(cfg)
    for key, val in overrides.items():
        if val is None:
            continue
        if "." in key:
            parts = key.split(".")
            cur = out
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = val
        else:
            out[key] = val
    return out


def dump_yaml(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
