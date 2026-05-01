"""实验目录管理：自动 ``{results_root}/{tag}_{n}/``，n 取已存在最大值+1。

支持两种模式：
- ``new``：扫描后 +1 创建新目录。
- ``resume``：定位 tag 对应**最大编号**目录，沿用（不增加）。
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_RESULTS_ROOT = Path("results")


def _scan(tag: str, results_root: str | Path = DEFAULT_RESULTS_ROOT) -> list[Path]:
    root = Path(results_root)
    if not root.exists():
        return []
    pat = re.compile(rf"^{re.escape(tag)}_(\d+)$")
    found = []
    for p in root.iterdir():
        if p.is_dir() and pat.match(p.name):
            found.append(p)
    found.sort(key=lambda p: int(pat.match(p.name).group(1)))
    return found


def make_or_resume_exp_dir(
    tag: str,
    resume: bool = False,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
) -> Path:
    """根据 tag 创建新实验目录或定位已有最大编号目录用于恢复。"""
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    existing = _scan(tag, root)
    if resume:
        if not existing:
            raise FileNotFoundError(f"--resume 失败：找不到任何 {root}/{tag}_*/ 目录。")
        return existing[-1]
    next_n = (int(re.match(rf"^{re.escape(tag)}_(\d+)$", existing[-1].name).group(1)) + 1
              if existing else 1)
    out = root / f"{tag}_{next_n}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "figures").mkdir(exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    return out
