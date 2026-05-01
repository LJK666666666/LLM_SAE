"""离线 SAE 烟雾测试：仅用合成激活数据，不依赖 Qwen 模型/HF 下载。

执行：
    python tests/test_sae_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.sae_topk import TopKSAE, TopKSAEConfig
from src.models.sae_jumprelu import JumpReLUSAE, JumpReLUSAEConfig


def test_topk_forward_backward():
    torch.manual_seed(0)
    cfg = TopKSAEConfig(d_in=64, d_sae=256, k=8, k_aux=16, dead_steps_threshold=2)
    sae = TopKSAE(cfg)
    x = torch.randn(32, 64)
    sae.train()
    out = sae(x)
    out.loss.backward()
    sae.remove_parallel_grad_component_()
    assert torch.isfinite(out.loss), "TopK loss 出现 NaN/Inf"
    assert out.x_hat.shape == x.shape
    assert out.z.shape == (32, 256)
    nz = (out.z != 0).sum(dim=-1)
    assert (nz <= cfg.k).all(), f"TopK 应保证每行非零数 ≤ k={cfg.k}，实际 max={nz.max().item()}"
    print(f"[topk] OK loss={out.loss.item():.4f} l0={out.l0.item():.1f} "
          f"dead={out.dead_frac.item():.2%} expl_var={out.explained_variance.item():.4f}")


def test_topk_dead_features_recover():
    torch.manual_seed(0)
    cfg = TopKSAEConfig(d_in=32, d_sae=128, k=4, k_aux=8, dead_steps_threshold=3, aux_loss_coef=1.0)
    sae = TopKSAE(cfg)
    sae.train()
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for _ in range(20):
        x = torch.randn(64, 32)
        out = sae(x)
        opt.zero_grad()
        out.loss.backward()
        sae.remove_parallel_grad_component_()
        opt.step()
        sae._normalize_decoder_()
    assert torch.isfinite(out.loss)
    print(f"[topk dead-feature loop] OK final loss={out.loss.item():.4f} "
          f"dead_frac={out.dead_frac.item():.2%}")


def test_jumprelu_forward_backward():
    torch.manual_seed(0)
    cfg = JumpReLUSAEConfig(d_in=64, d_sae=256, sparsity_coef=1e-3, bandwidth=1e-3)
    sae = JumpReLUSAE(cfg)
    x = torch.randn(32, 64)
    sae.train()
    out = sae(x)
    out.loss.backward()
    sae.remove_parallel_grad_component_()
    assert torch.isfinite(out.loss), "JumpReLU loss 出现 NaN/Inf"
    assert out.x_hat.shape == x.shape
    # 阈值参数应当有梯度
    assert sae.log_threshold.grad is not None and torch.isfinite(sae.log_threshold.grad).all()
    print(f"[jumprelu] OK loss={out.loss.item():.4f} l0={out.l0.item():.1f} "
          f"expl_var={out.explained_variance.item():.4f}")


def test_jumprelu_train_loop():
    torch.manual_seed(0)
    cfg = JumpReLUSAEConfig(d_in=32, d_sae=128, sparsity_coef=1e-2, bandwidth=1e-2)
    sae = JumpReLUSAE(cfg)
    sae.train()
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for _ in range(10):
        x = torch.randn(64, 32)
        out = sae(x)
        opt.zero_grad()
        out.loss.backward()
        sae.remove_parallel_grad_component_()
        opt.step()
        sae._normalize_decoder_()
    assert torch.isfinite(out.loss)
    print(f"[jumprelu train loop] OK final loss={out.loss.item():.4f}")


def test_exp_dir_and_metrics():
    """exp_dir + overall_metrics 简单 smoke。用 monkey-patch 而不 chdir，避免 Windows 临时目录清理问题。"""
    import tempfile, json
    import src.utils.exp_dir as ed
    import src.utils.overall_metrics as om

    with tempfile.TemporaryDirectory() as tmp:
        results_root = Path(tmp) / "results"
        ed_root_backup = ed.RESULTS_ROOT
        om_root_backup, om_csv_backup, om_json_backup = om.RESULTS_ROOT, om.CSV_PATH, om.JSON_PATH
        try:
            ed.RESULTS_ROOT = results_root
            om.RESULTS_ROOT = results_root
            om.CSV_PATH = results_root / "overall_config_metrics.csv"
            om.JSON_PATH = results_root / "overall_config_metrics.json"
            d1 = ed.make_or_resume_exp_dir("foo")
            d2 = ed.make_or_resume_exp_dir("foo")
            assert d1.name == "foo_1" and d2.name == "foo_2", (d1, d2)
            d3 = ed.make_or_resume_exp_dir("foo", resume=True)
            assert d3.name == "foo_2", d3
            om.append_record({"exp_dir": str(d2), "tag": "foo", "metric": 0.123})
            om.append_record({"exp_dir": str(d2), "tag": "foo", "metric": 0.999})  # 同 exp_dir 应替换
            recs = json.loads(om.JSON_PATH.read_text(encoding="utf-8"))
            assert len(recs) == 1 and recs[0]["metric"] == 0.999
            print(f"[exp_dir & overall_metrics] OK")
        finally:
            ed.RESULTS_ROOT = ed_root_backup
            om.RESULTS_ROOT, om.CSV_PATH, om.JSON_PATH = om_root_backup, om_csv_backup, om_json_backup


if __name__ == "__main__":
    test_topk_forward_backward()
    test_topk_dead_features_recover()
    test_jumprelu_forward_backward()
    test_jumprelu_train_loop()
    test_exp_dir_and_metrics()
    print("\n所有 SAE 离线烟雾测试通过。")
