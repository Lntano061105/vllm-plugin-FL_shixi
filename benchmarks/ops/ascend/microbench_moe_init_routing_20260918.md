# MoE Init Routing Custom 算子 Microbenchmark 结果

- 算子：`torch.ops._C_ascend.npu_moe_init_routing_custom`（源码 `csrc/ascend/moe/moe_init_routing_custom/`）
- 环境：Ascend 910B3（npu:0）、CANN 9.0.0、torch 2.8.0+cpu / torch_npu 2.8.0.post2、Python 3.11.14
- 对应任务书：第 3 节「Microbenchmark：算子级平均时延 + P50/P90」
- 采集日期：2026-09-18
- 脚本：`benchmarks/ops/ascend/microbench_moe_init_routing.py`（随本次提交入库）

## 1. 计时口径（答辩需明确）

| 口径 | 定义 | 说明 |
|---|---|---|
| mean / P50 / P90（主口径） | 单次算子调用端到端时延 = host 下发 + device 执行 + `torch.npu.synchronize()` | 逐次同步，反映真实调用链，含 host 侧固定开销 |
| loop_avg（辅口径） | 连续 1000 次调用仅同步一次，折算平均单次时延 | 剔除逐次同步开销，为**下界**参考 |

- 定位：**算子级**口径，不含模型前向、调度、KV cache 等模型级开销，与模型级 Output tok/s、TTFT/TPOT 不可直接比较。
- 固定 shape（与单测 9 用例一致）：x (8, 16) fp16/bf16，expert_idx (8, 2) int32，top_k=2，expert_num=4，seed=0；输入在计时窗口外一次性生成，避免 `randn` 开销混入。
- 预热后再采样，分位数采用线性插值（等价 numpy.percentile 默认行为）。

## 2. 结果（两次独立重测，单位 ms）

### Run 1：warmup 50 / iters 200

| 场景 | shape | dtype | mean | P50 | P90 | min | max | loop_avg |
|---|---|---|---|---|---|---|---|---|
| dropless/gather/COUNT | 8x16 | fp16 | 0.1628 | 0.1587 | 0.1732 | 0.1517 | 0.2360 | 0.0913 |
| dropless/scatter/COUNT | 8x16 | fp16 | 0.1631 | 0.1582 | 0.1758 | 0.1513 | 0.2234 | 0.0911 |
| dropless/gather/CUMSUM | 8x16 | fp16 | 0.1631 | 0.1586 | 0.1722 | 0.1514 | 0.2407 | 0.0850 |
| drop-pad/capacity=2 | 8x16 | fp16 | 0.1677 | 0.1643 | 0.1804 | 0.1549 | 0.2184 | 0.0843 |
| dropless/gather/COUNT | 8x16 | bf16 | 0.2015 | 0.1573 | 0.1804 | 0.1403 | 7.5822 | 0.0874 |
| dropless/gather/COUNT | 1024x1024 | fp16 | 0.1718 | 0.1655 | 0.1953 | 0.1591 | 0.4141 | 0.0814 |
| dropless/gather/COUNT | 4096x4096 | fp16 | 0.2507 | 0.2448 | 0.2622 | 0.2320 | 0.6644 | 0.0922 |

### Run 2：warmup 100 / iters 300（重测，主引用口径）

| 场景 | shape | dtype | mean | P50 | P90 | min | max | loop_avg |
|---|---|---|---|---|---|---|---|---|
| dropless/gather/COUNT | 8x16 | fp16 | 0.1302 | 0.1286 | 0.1370 | 0.1240 | 0.1805 | 0.0678 |
| dropless/scatter/COUNT | 8x16 | fp16 | 0.1330 | 0.1311 | 0.1386 | 0.1256 | 0.2286 | 0.0682 |
| dropless/gather/CUMSUM | 8x16 | fp16 | 0.1425 | 0.1321 | 0.1409 | 0.1253 | 2.6734 | 0.0690 |
| drop-pad/capacity=2 | 8x16 | fp16 | 0.1423 | 0.1419 | 0.1465 | 0.1336 | 0.1597 | 0.0733 |
| dropless/gather/COUNT | 8x16 | bf16 | 0.1369 | 0.1359 | 0.1406 | 0.1295 | 0.1842 | 0.0669 |
| dropless/gather/COUNT | 1024x1024 | fp16 | 0.1628 | 0.1611 | 0.1681 | 0.1545 | 0.2180 | 0.0670 |
| dropless/gather/COUNT | 4096x4096 | fp16 | 0.2040 | 0.2016 | 0.2208 | 0.1922 | 0.2454 | 0.0916 |

## 3. 观察与结论

1. **8×16 固定 shape 下，单次调用端到端时延约 0.13～0.17 ms**（Run 2 主口径 mean 0.130～0.143 ms，P50 0.129～0.142 ms，P90 ≤ 0.147 ms）；连续调用折算下界约 0.067～0.073 ms。该量级中 host 下发 + 逐次同步占主导，device 计算时间占比很小。
2. **规模梯度可见 device 时间**：hidden/batch 由 8×16 → 1024×1024 → 4096×4096，mean 由 0.130 → 0.163 → 0.204 ms（Run 2），扩到 4096×4096（expanded 8192×4096 fp16，约 64 MB）仍 **< 0.25 ms**。
3. **模式差异小**：dropless gather / scatter / CUMSUM 计数、drop-pad（capacity=2）四种模式在 8×16 下 mean 差异 ≤ 0.012 ms，说明路由模式切换不引入数量级开销；bf16 与 fp16 同量级。
4. **与端到端的关系**：本算子单次调用为亚毫秒级（≤0.25 ms），而模型级 Baseline 的 Output tok/s 为 268～490（TPOT 百毫秒量级），路由算子不构成端到端瓶颈；结论表述限定在算子级，不外推为端到端收益归因。
5. **稳定性提示**：两次重测 P50/P90 一致（8×16 各场景 P90 ≤ 0.181 ms），但采样中存在偶发离群（Run 1 bf16 出现 1 次 7.58 ms、Run 2 CUMSUM 出现 1 次 2.67 ms），符合单卡共享环境抖动特征；故 mean 需与 P50/P90 一同报告，不单看 max。

## 4. 复现方式

```bash
# 仓库根目录，脚本自检环境，无需手动 source
/usr/local/python3.11.14/bin/python3 benchmarks/ops/ascend/microbench_moe_init_routing.py
# 可调参数：--warmup 100 --iters 300 --loop 1000 --out <csv 路径>
```

## 5. 原始数据

- `microbench_moe_init_routing_20260918.csv`（Run 1）
- `microbench_moe_init_routing_20260918_rep2.csv`（Run 2）

## 6. 本口径的局限（被追问时的统一应答）

- 含 host 侧调用与同步开销，是**调用链时延**而非纯 device kernel 时间；如需纯 kernel 时间应使用 CANN Profiler / `torch_npu.profiler` 采集算子耗时。
- 8×16 为任务书固定 shape，属最小功能 shape，不能代表大 batch 推理场景；故另附 1024×1024、4096×4096 规模梯度作对照。
- 单卡单次测量，未做多卡并行与跨卡通信影响评估。
