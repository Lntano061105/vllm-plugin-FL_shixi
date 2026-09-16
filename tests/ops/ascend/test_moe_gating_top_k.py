#!/usr/bin/env python3
import torch
import numpy as np

# 1. 加载算子
torch.ops.load_library('/workspace/vllm-plugin-FL/vllm_fl/_C_ascend.cpython-311-aarch64-linux-gnu.so')

print("=" * 60)
print("MoE Gating Top-K 固定输入输出测试")
print("=" * 60)

# 2. 固定输入
print("\n[1] 准备固定输入:")
# 4个token，8个专家，固定的随机种子
torch.manual_seed(42)
x = torch.randn(4, 8).npu()
print(f"    输入 x 形状: {x.shape}")
print(f"    x[0][:5]: {x[0][:5]}")

# 3. 固定参数
k = 2
k_group = 1
group_count = 1
group_select_mode = 0
renorm = 0
norm_type = 0
out_flag = True
routed_scaling_factor = 1.0
eps = 1e-20

print(f"\n[2] 参数:")
print(f"    k: {k}")
print(f"    k_group: {k_group}")
print(f"    group_count: {group_count}")
print(f"    out_flag: {out_flag}")

# 4. 调用算子
print("\n[3] 调用 moe_gating_top_k:")
try:
    y, expert_idx, out = torch.ops._C_ascend.moe_gating_top_k(
        x, k, k_group, group_count, group_select_mode,
        renorm, norm_type, out_flag, routed_scaling_factor, eps
    )
    print("    ✅ 调用成功")
except Exception as e:
    print(f"    ❌ 调用失败: {e}")
    exit(1)

# 5. 验证输出
print("\n[4] 输出验证:")
print(f"    y 形状: {y.shape} (期望: [4, 2])")
print(f"    expert_idx 形状: {expert_idx.shape} (期望: [4, 2])")
print(f"    out 形状: {out.shape} (期望: [4, 8])")

# 6. 检查形状是否正确
assert y.shape == (4, 2), f"y 形状错误: {y.shape}"
assert expert_idx.shape == (4, 2), f"expert_idx 形状错误: {expert_idx.shape}"
assert out.shape == (4, 8), f"out 形状错误: {out.shape}"
print("    ✅ 形状验证通过")

# 7. 检查值范围
print("\n[5] 值范围检查:")
print(f"    y 范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
print(f"    expert_idx 范围: [{expert_idx.min().item()}, {expert_idx.max().item()}]")
print(f"    out 范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

# 8. 检查专家索引是否在合理范围（0-7）
assert expert_idx.min().item() >= 0, "expert_idx 最小值 < 0"
assert expert_idx.max().item() < 8, f"expert_idx 最大值 >= 8, 实际: {expert_idx.max().item()}"
print("    ✅ 专家索引范围验证通过")

# 9. 显示部分输出
print("\n[6] 输出示例 (前2个token):")
print(f"    y[0]: {y[0].tolist()}")
print(f"    expert_idx[0]: {expert_idx[0].tolist()}")
print(f"    out[0][:5]: {out[0][:5].tolist()}")

print("\n" + "=" * 60)
print("✅ 所有测试通过！")
print("=" * 60)

# 10. 保存固定结果（用于后续对比）
print("\n[7] 保存固定结果:")
result = {
    'x': x.cpu().tolist(),
    'y': y.cpu().tolist(),
    'expert_idx': expert_idx.cpu().tolist(),
    'out': out.cpu().tolist(),
}
torch.save(result, '/workspace/vllm-plugin-FL/fixed_result.pt')
print("    ✅ 结果已保存到 fixed_result.pt")
