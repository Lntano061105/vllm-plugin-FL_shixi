# -*- coding: utf-8 -*-
"""MoE Init Routing Custom 算子连接测试（CANN framework / aclnn 路径）。

算子源码：``csrc/ascend/moe/moe_init_routing_custom/``
torch 接口：``torch.ops._C_ascend.npu_moe_init_routing_custom``

运行方式（在仓库根目录，一条命令即可）：

    python tests/custom_ops_tests/test_moe_init_routing_custom.py

脚本会自检并补齐所需环境（CANN 运行时库 + 自定义算子包 ``op_api/lib``），
随后重启自身使 ``LD_LIBRARY_PATH`` 生效，因此不必先 source；等价的手动准备是：

    source /usr/local/Ascend/ascend-toolkit/set_env.sh    # libhccl.so 等 CANN 运行时
    source vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    python tests/custom_ops_tests/test_moe_init_routing_custom.py

算子包不在仓库内默认位置时，用 ``VLLM_FL_CUSTOM_OPP=<path>`` 指定。

接口签名（依据 ``op_host/moe_init_routing_custom_def.cpp``）：

    npu_moe_init_routing_custom(x, expert_idx, scale=None, offset=None,
                                active_num=-1, expert_capacity=-1, expert_num=-1,
                                drop_pad_mode=0, expert_tokens_num_type=1,
                                expert_tokens_num_flag=True, quant_mode=-1,
                                active_expert_range=[0, -1], row_idx_type=0)
    -> (expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale)

真机语义（Ascend 910B3 + CANN 9.0.0 实测确认）：

* Dropless（``drop_pad_mode=0``）
  - ``row_idx_type=1``：``expanded_row_idx`` 为 gather map，长度 ``num_rows*top_k``，
    满足 ``x.repeat_interleave(top_k, dim=0)[row_idx] == expanded_x``，
    且 gather 后的专家号序列非递减（即 expanded_x 已按专家排序）；
  - ``row_idx_type=0``：为上一者的逆映射（scatter：flat token 行 -> expanded 行号）。
* Drop-Pad（``drop_pad_mode=1``，此时 ``row_idx_type`` 必须为 0）
  - ``expanded_x`` 形状 ``(expert_num, expert_capacity, hidden)``；
  - ``expanded_row_idx`` 长度 ``num_rows*top_k``，值为该 token 在
    ``expanded_x.flatten(0, 1)`` 中的行号，超出 ``expert_capacity`` 被丢弃的为 ``-1``；
  - counts 记录未截断前的真实每专家 token 数；
  - ``expert_capacity`` 受 tiling 约束：``0 < capacity < 4``。
* ``expert_tokens_num_type``：0=CUMSUM（前缀和）、1=COUNT、2=KEY_VALUE（``[expert_num, 2]``
  的 ``[expert_id, count]``）。
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = os.environ.get(
    "VLLM_FL_CUSTOM_OPP",
    str(REPO_ROOT / "vllm_fl" / "_cann_ops_custom" / "vendors" / "custom_transformer"),
)


def _bootstrap_env():
    """补齐运行所需环境，使脚本可以不经 source 直接运行。

    缺两样东西：① CANN 运行时库（libhccl.so 等，torch_npu 依赖）；
    ② 自定义算子包 op_api/lib。LD_LIBRARY_PATH 只在进程启动时被动态链接器读取，
    因此补齐后用 ``os.execv`` 重启自身（带 guard 变量，只重启一次）。
    """
    parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    changed = False

    # ① CANN 运行时库
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

    # ② 自定义算子包
    if os.path.isdir(VENDOR_DIR):
        os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", VENDOR_DIR)
        op_api = os.path.join(VENDOR_DIR, "op_api", "lib")
        if op_api not in parts:
            parts.insert(0, op_api)
            changed = True
    else:
        print(
            f"WARNING: 未找到算子包目录 {VENDOR_DIR}，请用 VLLM_FL_CUSTOM_OPP=<path> 指定",
            flush=True,
        )

    if changed:
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        if not os.environ.get("_VLLM_FL_MOE_TEST_ENV_READY"):
            os.environ["_VLLM_FL_MOE_TEST_ENV_READY"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_bootstrap_env()

import torch
import torch_npu  # noqa: F401  # 注册 npu 后端

try:
    import vllm_fl._C_ascend  # noqa: F401
except ImportError as exc:  # pragma: no cover
    print(f"FATAL: 无法 import vllm_fl._C_ascend: {exc}")
    print("       请先在仓库根目录执行 VLLM_VENDOR=ascend python setup.py build_ext --inplace")
    sys.exit(2)

# ------------------------------------------------------------------ constants
DEVICE = "npu:0"
EXPERT_NUM = 4
BATCH = 8
HIDDEN = 16
TOP_K = 2

DROP_PAD_MODE_DROPLESS = 0
DROP_PAD_MODE_DROP_PAD = 1

TOKENS_NUM_CUMSUM = 0
TOKENS_NUM_COUNT = 1
TOKENS_NUM_KEY_VALUE = 2

QUANT_MODE_UNQUANT = -1

ROW_IDX_SCATTER = 0
ROW_IDX_GATHER = 1

CUSTOM_OP = torch.ops._C_ascend.npu_moe_init_routing_custom


def _rand_inputs(batch=BATCH, hidden=HIDDEN, top_k=TOP_K, expert_num=EXPERT_NUM,
                 dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch, hidden, device=DEVICE, dtype=dtype)
    expert_idx = torch.randint(0, expert_num, (batch, top_k), device=DEVICE,
                               dtype=torch.int32)
    return x, expert_idx


def _run(x, expert_idx, expert_num=EXPERT_NUM, **kwargs):
    params = dict(
        active_num=-1,
        expert_capacity=-1,
        expert_num=expert_num,
        drop_pad_mode=DROP_PAD_MODE_DROPLESS,
        expert_tokens_num_type=TOKENS_NUM_COUNT,
        expert_tokens_num_flag=True,
        quant_mode=QUANT_MODE_UNQUANT,
        active_expert_range=[0, expert_num],
        row_idx_type=ROW_IDX_GATHER,
    )
    params.update(kwargs)
    return CUSTOM_OP(x, expert_idx, **params)


def _bincount(expert_idx, expert_num=EXPERT_NUM):
    return torch.bincount(expert_idx.reshape(-1).long(), minlength=expert_num)


# --------------------------------------------------------------------- tests
def test_dropless_gather_map():
    """Dropless + row_idx_type=1：row_idx 是合法 gather map 且 expanded_x 按专家有序。"""
    x, expert_idx = _rand_inputs(seed=0)
    n = x.shape[0] * TOP_K
    expanded_x, row_idx, counts, _ = _run(x, expert_idx, active_num=n,
                                          row_idx_type=ROW_IDX_GATHER)
    torch.npu.synchronize()

    assert tuple(expanded_x.shape) == (n, x.shape[1]), f"expanded_x shape {tuple(expanded_x.shape)}"
    assert expanded_x.dtype == x.dtype, "expanded_x dtype 与输入不一致"

    row_idx = row_idx.long()
    assert row_idx.numel() == n, "row_idx 长度不等于 num_rows*top_k"
    assert torch.equal(row_idx.sort().values, torch.arange(n, device=DEVICE)), \
        "row_idx 不是 0..n-1 的排列"

    x_flat = x.repeat_interleave(TOP_K, dim=0)
    assert torch.equal(x_flat.index_select(0, row_idx), expanded_x), \
        "expanded_x != x.repeat_interleave(top_k)[row_idx]"

    flat_expert = expert_idx.reshape(-1).long().index_select(0, row_idx)
    assert torch.all(flat_expert[1:] >= flat_expert[:-1]), "expanded_x 未按专家号非递减排序"

    assert torch.equal(counts.long(), _bincount(expert_idx)), \
        f"counts {counts.tolist()} != {_bincount(expert_idx).tolist()}"


def test_dropless_scatter_map():
    """Dropless + row_idx_type=0：row_idx 为 gather map 的逆映射。"""
    x, expert_idx = _rand_inputs(seed=1)
    n = x.shape[0] * TOP_K
    expanded_x, scatter, _, _ = _run(x, expert_idx, active_num=n,
                                     row_idx_type=ROW_IDX_SCATTER)
    torch.npu.synchronize()

    scatter = scatter.long()
    assert torch.equal(scatter.sort().values, torch.arange(n, device=DEVICE)), \
        "scatter row_idx 不是排列"

    gather = torch.empty_like(scatter)
    gather.index_copy_(0, scatter, torch.arange(n, device=DEVICE))
    x_flat = x.repeat_interleave(TOP_K, dim=0)
    assert torch.equal(x_flat.index_select(0, gather), expanded_x), \
        "scatter 逆映射还原出的 expanded_x 不正确"


def test_expert_tokens_num_types():
    """CUMSUM / COUNT / KEY_VALUE 三种计数类型。"""
    x, expert_idx = _rand_inputs(seed=2)
    n = x.shape[0] * TOP_K
    expected = _bincount(expert_idx)

    _, _, count, _ = _run(x, expert_idx, active_num=n,
                          expert_tokens_num_type=TOKENS_NUM_COUNT)
    torch.npu.synchronize()
    assert torch.equal(count.long(), expected), f"COUNT {count.tolist()} != {expected.tolist()}"

    _, _, cumsum, _ = _run(x, expert_idx, active_num=n,
                           expert_tokens_num_type=TOKENS_NUM_CUMSUM)
    torch.npu.synchronize()
    assert torch.equal(cumsum.long(), torch.cumsum(expected, 0)), \
        f"CUMSUM {cumsum.tolist()} != {torch.cumsum(expected, 0).tolist()}"

    _, _, kv, _ = _run(x, expert_idx, active_num=n,
                       expert_tokens_num_type=TOKENS_NUM_KEY_VALUE)
    torch.npu.synchronize()
    assert tuple(kv.shape) == (EXPERT_NUM, 2), f"KEY_VALUE shape {tuple(kv.shape)}"
    assert torch.equal(kv[:, 0].long(), torch.arange(EXPERT_NUM, device=DEVICE)), \
        "KEY_VALUE 第一列应为专家号"
    assert torch.equal(kv[:, 1].long(), expected), "KEY_VALUE 第二列应等于每专家 token 数"


def test_tokens_num_flag_disabled():
    """expert_tokens_num_flag=False：counts 不保证被填充，仅校验输出结构合法。

    真机实测：flag=False 时 counts buffer 内容不确定（未清零），调用方不应依赖其数值；
    本用例只校验该开关不破坏其余输出的结构契约。
    """
    x, expert_idx = _rand_inputs(seed=3)
    n = x.shape[0] * TOP_K
    expanded_x, row_idx, counts, _ = _run(x, expert_idx, active_num=n,
                                          expert_tokens_num_flag=False)
    torch.npu.synchronize()
    assert tuple(expanded_x.shape) == (n, x.shape[1]), "flag=False 影响 expanded_x 形状"
    assert row_idx.numel() == n, "flag=False 影响 row_idx 长度"
    assert tuple(counts.shape) == (EXPERT_NUM,), f"counts shape {tuple(counts.shape)}"
    assert counts.dtype == torch.int64, f"counts dtype {counts.dtype}"


def test_drop_pad_capacity():
    """Drop-Pad：expert_capacity 截断，超出部分 row_idx=-1，counts 仍为真实计数。"""
    x, expert_idx = _rand_inputs(seed=4)
    n = x.shape[0] * TOP_K
    capacity = 2
    expanded_x, row_idx, counts, _ = _run(
        x, expert_idx, active_num=-1, expert_capacity=capacity,
        drop_pad_mode=DROP_PAD_MODE_DROP_PAD, row_idx_type=ROW_IDX_SCATTER)
    torch.npu.synchronize()

    assert tuple(expanded_x.shape) == (EXPERT_NUM, capacity, x.shape[1]), \
        f"Drop-Pad expanded_x shape {tuple(expanded_x.shape)}"

    row_idx = row_idx.long()
    x_flat = x.repeat_interleave(TOP_K, dim=0)
    kept = row_idx >= 0
    expected_kept = int(torch.clamp(_bincount(expert_idx), max=capacity).sum())
    assert row_idx.numel() == n, "Drop-Pad row_idx 长度应等于 num_rows*top_k"
    assert int(kept.sum()) == expected_kept, \
        f"保留 token 数 {int(kept.sum())} != sum(min(count_e, capacity)) {expected_kept}"
    assert int((~kept).sum()) == n - expected_kept, "丢弃 token 数与容量不符"
    assert int(row_idx[kept].max()) < EXPERT_NUM * capacity, "row_idx 超出 expanded 行数"
    assert torch.equal(expanded_x.reshape(-1, x.shape[1]).index_select(0, row_idx[kept]),
                       x_flat[kept]), "Drop-Pad 保留 token 的展开位置不正确"
    assert torch.equal(counts.long(), _bincount(expert_idx)), \
        "Drop-Pad counts 应为未截断的真实计数"


def test_drop_pad_capacity_boundary():
    """Drop-Pad 在不同 capacity（1/2/3）下的丢弃数量与保留映射正确。"""
    x, expert_idx = _rand_inputs(seed=5)
    n = x.shape[0] * TOP_K
    x_flat = x.repeat_interleave(TOP_K, dim=0)
    expert_counts = _bincount(expert_idx)
    for capacity in (1, 2, 3):
        expanded_x, row_idx, _, _ = _run(
            x, expert_idx, active_num=-1, expert_capacity=capacity,
            drop_pad_mode=DROP_PAD_MODE_DROP_PAD, row_idx_type=ROW_IDX_SCATTER)
        torch.npu.synchronize()
        assert tuple(expanded_x.shape) == (EXPERT_NUM, capacity, x.shape[1])
        row_idx = row_idx.long()
        kept = row_idx >= 0
        expected_kept = int(torch.clamp(expert_counts, max=capacity).sum())
        assert int(kept.sum()) == expected_kept, \
            f"capacity={capacity} 保留数 {int(kept.sum())} != {expected_kept}"
        assert int(row_idx[kept].max()) < EXPERT_NUM * capacity, \
            f"capacity={capacity} row_idx 超出 expanded 行数"
        assert torch.equal(expanded_x.reshape(-1, x.shape[1]).index_select(0, row_idx[kept]),
                           x_flat[kept]), f"capacity={capacity} 保留映射不正确"


def test_boundary_single_expert():
    """边界：所有 token 路由到专家 0，其余专家为空。"""
    torch.manual_seed(6)
    x = torch.randn(BATCH, HIDDEN, device=DEVICE, dtype=torch.float16)
    expert_idx = torch.zeros(BATCH, TOP_K, device=DEVICE, dtype=torch.int32)
    n = BATCH * TOP_K

    expanded_x, row_idx, counts, _ = _run(x, expert_idx, active_num=n,
                                          row_idx_type=ROW_IDX_GATHER)
    torch.npu.synchronize()
    counts = counts.long()
    assert int(counts[0]) == n and int(counts[1:].sum()) == 0, f"counts 异常 {counts.tolist()}"
    x_flat = x.repeat_interleave(TOP_K, dim=0)
    assert torch.equal(expanded_x, x_flat), "单专家场景 expanded_x 应等于按 top_k 展开的输入"


def test_boundary_partially_empty_experts():
    """边界：仅使用部分专家（0 与 3），其余专家计数为 0。"""
    torch.manual_seed(7)
    x = torch.randn(BATCH, HIDDEN, device=DEVICE, dtype=torch.float16)
    expert_idx = torch.randint(0, 2, (BATCH, TOP_K), device=DEVICE, dtype=torch.int32) * 3
    n = BATCH * TOP_K

    expanded_x, row_idx, counts, _ = _run(x, expert_idx, active_num=n,
                                          row_idx_type=ROW_IDX_GATHER)
    torch.npu.synchronize()
    counts = counts.long()
    assert int(counts[1]) == 0 and int(counts[2]) == 0, f"空专家计数非 0：{counts.tolist()}"
    assert int(counts[0] + counts[3]) == n, "计数总和应等于 num_rows*top_k"

    x_flat = x.repeat_interleave(TOP_K, dim=0)
    assert torch.equal(x_flat.index_select(0, row_idx.long()), expanded_x), \
        "部分空专家场景展开结果不正确"


def test_bf16_input():
    """bf16 输入全链路（展开 + 计数）正确。"""
    x, expert_idx = _rand_inputs(dtype=torch.bfloat16, seed=8)
    n = x.shape[0] * TOP_K
    expanded_x, row_idx, counts, _ = _run(x, expert_idx, active_num=n,
                                          row_idx_type=ROW_IDX_GATHER)
    torch.npu.synchronize()

    assert expanded_x.dtype == torch.bfloat16, "输出未保持 bf16"
    x_flat = x.repeat_interleave(TOP_K, dim=0)
    assert torch.equal(x_flat.index_select(0, row_idx.long()), expanded_x), "bf16 展开结果不正确"
    assert torch.equal(counts.long(), _bincount(expert_idx)), "bf16 计数不正确"


# ----------------------------------------------------------------- execution
TESTS = [
    test_dropless_gather_map,
    test_dropless_scatter_map,
    test_expert_tokens_num_types,
    test_tokens_num_flag_disabled,
    test_drop_pad_capacity,
    test_drop_pad_capacity_boundary,
    test_boundary_single_expert,
    test_boundary_partially_empty_experts,
    test_bf16_input,
]

if __name__ == "__main__":
    passed, failed = 0, 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS: {fn.__name__}", flush=True)
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
            failed += 1
    print(f"\nmoe_init_routing_custom test: {passed}/{len(TESTS)} passed")
    sys.exit(0 if failed == 0 else 1)
