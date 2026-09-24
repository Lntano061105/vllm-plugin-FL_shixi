"""benchmarks/ops/ascend/bench_chunk_gated_delta_rule.py
Chunk Gated Delta Rule 算子（npu_chunk_gated_delta_rule）Microbenchmark

要求（任务书 §3-R6 / §4-R3）：
- 预热（warmup）：首次调用含 kernel 编译/初始化，不计入统计
- 同步（sync）：每轮计时前 torch.npu.synchronize()，确保异步执行完成
- 重复（repeat）：多次重复取平均，报告平均时延、P50、P90
- 覆盖固定 Shape：以 Qwen3.6-27B Prefill 实际规模为准（T=1024, Nk=4, Nv=8, Dk=128, Dv=128）

输出：mean / P50 / P90（毫秒），以及总吞吐（tok/s）
运行：python3 bench_chunk_gated_delta_rule.py [--n-warmup 20] [--n-repeat 100]
"""
import argparse
import statistics
import time

import torch

# NPU 依赖做成可选：与 tests/ops/ascend/test_chunk_gated_delta_rule.py 保持一致，
# 无卡环境下给出明确提示而非 Traceback。
_NPU_IMPORT_ERROR = ""
try:
    import torch_npu  # noqa: F401  引入 npu 设备接口

    import vllm_fl._C_ascend  # noqa: F401  确保算子注册
except Exception as _e:  # pragma: no cover - 环境相关
    _NPU_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"


def make_inputs(T, Nk, Nv, Dk, Dv, B, seed=42):
    torch.manual_seed(seed)
    dev = "npu"
    q = torch.randn(T, Nk, Dk, dtype=torch.bfloat16, device=dev)
    k = torch.randn(T, Nk, Dk, dtype=torch.bfloat16, device=dev)
    q = q / q.norm(dim=-1, keepdim=True)   # L2 归一化（与调用侧 l2norm_fwd 一致）
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(T, Nv, Dv, dtype=torch.bfloat16, device=dev) * 0.1
    beta = torch.rand(T, Nv, dtype=torch.bfloat16, device=dev).sigmoid()
    g = (-torch.nn.functional.softplus(torch.randn(T, Nv, device=dev), beta=1.0)).float()
    initial_state = torch.randn(B, Nv, Dv, Dk, dtype=torch.bfloat16, device=dev) * 0.01
    actual_seq_lengths = torch.tensor([T // B] * B, dtype=torch.int32, device=dev)
    scale = Dk ** -0.5
    return q, k, v, beta, g, initial_state, actual_seq_lengths, scale


def bench_once(q, k, v, beta, g, initial_state, actual_seq_lengths, scale):
    t0 = time.perf_counter()
    torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        q, k, v, beta, initial_state, actual_seq_lengths, g=g, scale_value=scale)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0   # ms


def main():
    # 环境守卫：无 NPU 时明确退出，不伪装成"跑出结果"
    if _NPU_IMPORT_ERROR:
        print(f"[SKIP] 无可用 NPU 算子环境（{_NPU_IMPORT_ERROR}）。")
        print("       本脚本需在昇腾环境运行（先 source set_env.bash 并确认算子已编译）。")
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-repeat", type=int, default=100)
    ap.add_argument("--T", type=int, default=1024, help="总 token 数（Qwen 27B prefill 规模）")
    ap.add_argument("--Nk", type=int, default=4, help="KV head 数（Qwen3.6-27B）")
    ap.add_argument("--Nv", type=int, default=8, help="Value head 数")
    ap.add_argument("--Dk", type=int, default=128)
    ap.add_argument("--Dv", type=int, default=128)
    ap.add_argument("--B", type=int, default=1, help="batch（序列数）")
    args = ap.parse_args()

    q, k, v, beta, g, initial_state, actual_seq_lengths, scale = make_inputs(
        args.T, args.Nk, args.Nv, args.Dk, args.Dv, args.B)

    # 预热（含 kernel 编译/首次初始化，不计入统计）
    print(f"预热 {args.n_warmup} 次 ...")
    for _ in range(args.n_warmup):
        bench_once(q, k, v, beta, g, initial_state, actual_seq_lengths, scale)

    # 同步 + 重复测试
    lat = [bench_once(q, k, v, beta, g, initial_state, actual_seq_lengths, scale)
           for _ in range(args.n_repeat)]
    lat.sort()
    mean_ms = statistics.mean(lat)
    p50_ms = lat[int(len(lat) * 0.50)]
    p90_ms = lat[int(len(lat) * 0.90)]
    throughput = args.T / (mean_ms / 1000.0)

    print("=" * 56)
    print("Chunk Gated Delta Rule Microbenchmark")
    print(f"Shape: T={args.T} Nk={args.Nk} Nv={args.Nv} Dk={args.Dk} Dv={args.Dv} B={args.B}")
    print(f"重复 {args.n_repeat} 次（预热 {args.n_warmup} 次后）")
    print("-" * 56)
    print(f"Mean : {mean_ms:8.3f} ms")
    print(f"P50  : {p50_ms:8.3f} ms")
    print(f"P90  : {p90_ms:8.3f} ms")
    print(f"吞吐 : {throughput:8.1f} tok/s")
    print("=" * 56)


if __name__ == "__main__":
    main()
