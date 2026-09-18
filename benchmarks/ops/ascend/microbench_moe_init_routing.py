# -*- coding: utf-8 -*-
"""MoE Init Routing Custom 算子级 Microbenchmark（Ascend 910B3 / CANN 9.0.0）。

对应任务书第 3 节要求：Microbenchmark 平均时延 + P50/P90。

计时口径（与模型级 benchmark 的区别，答辩需明确）：

* 本脚本是 **算子级** 口径：直接调用 ``torch.ops._C_ascend.npu_moe_init_routing_custom``，
  每个样本 = 一次 host 下发 + device 执行 + ``torch.npu.synchronize()``，
  即单次算子调用的端到端时延；不含模型前向、调度、KV cache 等模型级开销。
* 固定 shape（与单测 9 用例一致）：x (8, 16) fp16 / bf16，expert_idx (8, 2) int32，
  top_k = 2，expert_num = 4，seed 固定；输入在计时前一次性生成，避免 randn 开销混入。
* 每个场景先 warmup 再采样 iters 次，报告 mean / P50 / P90 / min / max（ms），
  另附「连续调用」口径：连续呼叫 loop 次只同步一次，折算平均单次时延，
  用于剔除逐次 synchronize 的固定开销（该口径偏乐观，作为下界参考）。

用法（仓库根目录，无需手动 source，脚本自检环境）：

    python benchmarks/ops/ascend/microbench_moe_init_routing.py
    python benchmarks/ops/ascend/microbench_moe_init_routing_custom.py --iters 500 --warmup 100

    # 等价的手动环境准备：
    #   source /usr/local/Ascend/ascend-toolkit/set_env.sh
    #   source vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # benchmarks/ops/ascend/<this>.py
VENDOR_DIR = os.environ.get(
    "VLLM_FL_CUSTOM_OPP",
    str(REPO_ROOT / "vllm_fl" / "_cann_ops_custom" / "vendors" / "custom_transformer"),
)


def _bootstrap_env():
    """补齐运行所需环境，使脚本可以不经 source 直接运行。

    缺两样：① CANN 运行时库（libhccl.so 等，torch_npu 依赖）；② 自定义算子包 op_api/lib。
    LD_LIBRARY_PATH 只在进程启动时被动态链接器读取，故补齐后用 os.execv 重启自身（带 guard，只重启一次）。
    """
    parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    changed = False

    if not any("cann-" in p or "ascend-toolkit" in p for p in parts):
        for script in (
            "/usr/local/Ascend/ascend-toolkit/set_env.sh",
            "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh",
        ):
            if not os.path.exists(script):
                continue
            res = subprocess.run(
                ["bash", "-c", f'source "{script}" >/dev/null 2>&1; printf %s "$LD_LIBRARY_PATH"'],
                capture_output=True,
                text=True,
            )
            for p in res.stdout.strip().split(":"):
                if p and p not in parts:
                    parts.append(p)
                    changed = True
            if changed:
                break

    if os.path.isdir(VENDOR_DIR):
        os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", VENDOR_DIR)
        op_api = os.path.join(VENDOR_DIR, "op_api", "lib")
        if op_api not in parts:
            parts.insert(0, op_api)
            changed = True
    else:
        print(f"WARNING: 未找到算子包目录 {VENDOR_DIR}，请用 VLLM_FL_CUSTOM_OPP=<path> 指定", flush=True)

    if changed:
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        if not os.environ.get("_VLLM_FL_MOE_MICROBENCH_ENV_READY"):
            os.environ["_VLLM_FL_MOE_MICROBENCH_ENV_READY"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_bootstrap_env()

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401  # 注册 npu 后端

try:
    import vllm_fl._C_ascend  # noqa: E402,F401
except ImportError as exc:  # pragma: no cover
    print(f"FATAL: 无法 import vllm_fl._C_ascend: {exc}")
    print("       请先在仓库根目录执行 VLLM_VENDOR=ascend python setup.py build_ext --inplace")
    sys.exit(2)

# ------------------------------------------------------------------ constants
DEVICE = "npu:0"
EXPERT_NUM = 4
TOP_K = 2
SEED = 0

CUSTOM_OP = torch.ops._C_ascend.npu_moe_init_routing_custom


def _make_inputs(batch, hidden, top_k=TOP_K, expert_num=EXPERT_NUM, dtype=torch.float16):
    """固定 seed 生成输入（在计时窗口之外一次性完成）。"""
    torch.manual_seed(SEED)
    x = torch.randn(batch, hidden, device=DEVICE, dtype=dtype)
    expert_idx = torch.randint(0, expert_num, (batch, top_k), device=DEVICE, dtype=torch.int32)
    return x, expert_idx


def _call(x, expert_idx, *, drop_pad_mode=0, expert_capacity=-1, row_idx_type=1,
          tokens_type=1, active_num=-1, expert_num=EXPERT_NUM):
    return CUSTOM_OP(
        x,
        expert_idx,
        active_num=active_num,
        expert_capacity=expert_capacity,
        expert_num=expert_num,
        drop_pad_mode=drop_pad_mode,
        expert_tokens_num_type=tokens_type,
        expert_tokens_num_flag=True,
        quant_mode=-1,
        active_expert_range=[0, expert_num],
        row_idx_type=row_idx_type,
    )


def _percentile(sorted_vals, q):
    """线性插值分位数（与 numpy.percentile 默认行为一致，避免额外依赖）。"""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _bench(fn, warmup, iters, loop):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    # 连续调用口径：loop 次只同步一次，折算单次时延
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(loop):
        fn()
    torch.npu.synchronize()
    loop_avg = (time.perf_counter() - t0) * 1000.0 / loop

    s = sorted(samples)
    return {
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": _percentile(s, 0.50),
        "p90_ms": _percentile(s, 0.90),
        "min_ms": s[0],
        "max_ms": s[-1],
        "loop_avg_ms": loop_avg,
    }


def build_scenes():
    """返回 [(场景名, shape, dtype 名, 模式说明, 调用闭包)]，覆盖 dropless/scatter/drop-pad/多规模/bf16。"""
    scenes = []

    def add(name, batch, hidden, dtype, mode, drop_pad_mode, row_idx_type,
            expert_capacity=-1, tokens_type=1):
        x, expert_idx = _make_inputs(batch, hidden, dtype=dtype)
        n = batch * TOP_K
        fn = lambda: _call(  # noqa: E731
            x, expert_idx,
            drop_pad_mode=drop_pad_mode,
            expert_capacity=expert_capacity,
            row_idx_type=row_idx_type,
            tokens_type=tokens_type,
            active_num=n,
        )
        scenes.append((name, f"{batch}x{hidden}", str(dtype).replace("torch.", ""), mode, fn))

    add("dropless_gather", 8, 16, torch.float16, "dropless/gather/COUNT", 0, 1)
    add("dropless_scatter", 8, 16, torch.float16, "dropless/scatter/COUNT", 0, 0)
    add("dropless_cumsum", 8, 16, torch.float16, "dropless/gather/CUMSUM", 0, 1, tokens_type=0)
    add("drop_pad_cap2", 8, 16, torch.float16, "drop-pad/capacity=2", 1, 0, expert_capacity=2)
    add("dropless_gather_bf16", 8, 16, torch.bfloat16, "dropless/gather/COUNT", 0, 1)
    # 规模梯度：用于观察 device 时间占比（8x16 为任务书固定 shape 主口径）
    add("dropless_gather_1024x1024", 1024, 1024, torch.float16, "dropless/gather/COUNT", 0, 1)
    add("dropless_gather_4096x4096", 4096, 4096, torch.float16, "dropless/gather/COUNT", 0, 1)
    return scenes


def main():
    ap = argparse.ArgumentParser(description="MoE Init Routing Custom 算子级 microbenchmark")
    ap.add_argument("--warmup", type=int, default=50, help="预热次数（默认 50）")
    ap.add_argument("--iters", type=int, default=200, help="计时采样次数（默认 200）")
    ap.add_argument("--loop", type=int, default=1000, help="连续调用口径的循环次数（默认 1000）")
    ap.add_argument("--out", type=str, default="", help="输出 csv 路径，默认写入 benchmarks/ops/ascend/")
    ap.add_argument("--date-tag", type=str, default=time.strftime("%Y%m%d"), help="结果日期标记")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        REPO_ROOT / "benchmarks" / "ops" / "ascend" / f"microbench_moe_init_routing_{args.date_tag}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[microbench] device={DEVICE} warmup={args.warmup} iters={args.iters} loop={args.loop}")
    print(f"[microbench] out={out}")
    print(f"{'scene':<26}{'shape':>10}{'dtype':>10}  {'mean':>9}{'P50':>9}{'P90':>9}{'loop_avg':>10}")

    rows = []
    for name, shape, dtype, mode, fn in build_scenes():
        r = _bench(fn, args.warmup, args.iters, args.loop)
        rows.append({
            "scene": name,
            "shape": shape,
            "dtype": dtype,
            "mode": mode,
            "warmup": args.warmup,
            "iters": args.iters,
            "mean_ms": f"{r['mean_ms']:.4f}",
            "p50_ms": f"{r['p50_ms']:.4f}",
            "p90_ms": f"{r['p90_ms']:.4f}",
            "min_ms": f"{r['min_ms']:.4f}",
            "max_ms": f"{r['max_ms']:.4f}",
            "loop_avg_ms": f"{r['loop_avg_ms']:.4f}",
        })
        print(f"{name:<26}{shape:>10}{dtype:>10}  {r['mean_ms']:>9.4f}{r['p50_ms']:>9.4f}"
              f"{r['p90_ms']:>9.4f}{r['loop_avg_ms']:>10.4f}")

    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[microbench] saved -> {out}")
    print("[microbench] 口径: mean/P50/P90 为逐次调用(含 synchronize)时延; loop_avg 为连续"
          f"{args.loop}次调用折算单次时延(下界)")


if __name__ == "__main__":
    main()
