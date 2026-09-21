#!/usr/bin/env python3
"""
MoE Gating Top-K 算子性能测试
模拟真实 MoE 场景：不同 token 数、专家数、k 值
"""

import torch
import time
import numpy as np
import json
from datetime import datetime

# 加载算子
torch.ops.load_library('/workspace/vllm-plugin-FL/vllm_fl/_C_ascend.cpython-311-aarch64-linux-gnu.so')

print("=" * 70)
print("MoE Gating Top-K 性能测试")
print("=" * 70)

# 测试配置
test_configs = [
    # (num_tokens, num_experts, k, desc)
    (128, 64, 2, "小规模"),
    (256, 128, 2, "中规模"),
    (512, 256, 2, "大规模"),
    (1024, 256, 4, "大规模 + k=4"),
    (2048, 256, 2, "大批次"),
    (4096, 256, 2, "大批次 +"),
]

results = []

def benchmark(name, num_tokens, num_experts, k, num_warmup=10, num_iter=100):
    """运行性能测试"""
    print(f"\n[{name}] tokens={num_tokens}, experts={num_experts}, k={k}")
    
    # 准备输入
    x = torch.randn(num_tokens, num_experts, dtype=torch.float16).npu()
    
    # 预热
    for _ in range(num_warmup):
        _ = torch.ops._C_ascend.moe_gating_top_k(
            x, k, 1, 1, 0, 0, 0, True, 1.0, 1e-20
        )
    torch.npu.synchronize()
    
    # 正式测试
    latencies = []
    for _ in range(num_iter):
        start = time.perf_counter()
        result = torch.ops._C_ascend.moe_gating_top_k(
            x, k, 1, 1, 0, 0, 0, True, 1.0, 1e-20
        )
        torch.npu.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # 转为毫秒
    
    # 统计
    latencies = np.array(latencies)
    stats = {
        'name': name,
        'num_tokens': num_tokens,
        'num_experts': num_experts,
        'k': k,
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p90_ms': np.percentile(latencies, 90),
        'p99_ms': np.percentile(latencies, 99),
        'throughput_tokens_per_sec': num_tokens * 1000 / np.mean(latencies),
    }
    
    print(f"  平均时延: {stats['mean_ms']:.3f} ms")
    print(f"  P50: {stats['p50_ms']:.3f} ms, P90: {stats['p90_ms']:.3f} ms")
    print(f"  吞吐量: {stats['throughput_tokens_per_sec']:.0f} tokens/sec")
    
    return stats

print("\n开始性能测试...")
print("-" * 70)

for config in test_configs:
    num_tokens, num_experts, k, desc = config
    stats = benchmark(
        name=f"{desc} ({num_tokens}×{num_experts}→k={k})",
        num_tokens=num_tokens,
        num_experts=num_experts,
        k=k
    )
    results.append(stats)

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)

# 保存结果
output = {
    'timestamp': datetime.now().isoformat(),
    'results': results
}

with open('benchmark_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n📁 结果已保存到: benchmark_results.json")

# 输出汇总表格
print("\n📊 汇总:")
print("-" * 80)
print(f"{'配置':<35} {'平均时延':<12} {'P50':<10} {'P90':<10} {'吞吐量(tok/s)':<15}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<35} {r['mean_ms']:<12.2f} {r['p50_ms']:<10.2f} {r['p90_ms']:<10.2f} {r['throughput_tokens_per_sec']:<15.0f}")
print("-" * 80)
