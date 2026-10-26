# QuantLightningIndexerV2 DCP（解码上下文并行）特性设计 — arch22 / 910B

| 项 | 内容 |
|---|---|
| 算子 | `VllmAscendQuantLightningIndexerV2`（`csrc/attention/quant_lightning_indexer_v2`） |
| 目标芯片 | arch22：`DAV_2201`（Ascend 910B / 910_93 / Atlas A3） |
| 需求来源 | `https://gitcode.com/jj_tang/main/issues/7`《kim3.1适配》 |
| 状态 | 设计锁定（2026-10-26），进入实现 |
| 关键词 | DCP（Decode Context Parallel，解码上下文并行）、CP（Context Parallel，上下文并行）、PA（Page Attention，分页注意力）、TND（Token-Num-Dim，变长序列 0 轴拼接布局）、TopK（前 K 个候选） |

---

## 1. 需求分析

### 1.1 问题

vLLM 的 DCP 把 KV cache（Key/Value 缓存）按 token 轮转切分到 CP 组的各张卡上。当前 CANN 侧的 FIA（Flash Infer Attention，融合注意力）类算子在 CP 场景下靠框架构造二维 mask（掩码矩阵）再传给算子，mask 的构造与搬运成为性能瓶颈。

Lightning Indexer 需要为每个 Q（query）行在**本地可见 KV** 中选出 TopK，输出候选 index 与可选 score。CP 场景下有两件事与非 CP 不同：

1. Q 的全局位置不能由本地 KV 长度反推（本地只有全局序列的 1/P）；
2. 相邻 Q 行的本地可见 KV 数不一定每次 +1（rank 轮转导致可见长度呈阶梯增长）。

因此现有的两级 mask 边界公式在 CP 下都不成立：

- 块级遍历上界：`validS2Len = L - Q_len + s1Offset + s1BaseSize`
- 行级候选上界：`visible_len = L - Q_len + rowIdx + 1`

（`L` 为本地有效 KV 长度，`Q_len` 为该请求本次调用的 Q 长度。）

### 1.2 方案

把全局因果上界 `C` 与分片参数 `P/r/S` 直接作为算子输入/属性，kernel 内用闭式公式算本地可见长度，**不构造、不传递 mask**：

```
global_q = C - Q_len + i
global_k = ((local_k // S) * P + r) * S + local_k % S
visible  = (local_k < L) and (global_k <= global_q)
```

其中：
- `C`：每请求本次 Q 片段的全局 exclusive 上界（最后一个 Q 行的全局位置 + 1），Device Tensor，避免每轮同步回 CPU；
- `P`：CP 组大小（`cp_world`）；
- `r`：当前 rank（`cp_rank`）；
- `S`：分片粒度（`cp_kv_cache_interleave_size`），每连续 `S` 个 token 轮换 rank；逐 token 为 1，整 block 分片时传物理 `block_size`。

### 1.3 竞品参照

| 框架 | 做法 | 关键点 |
|---|---|---|
| vLLM（CUDA） | `[Attention][DCP] Use FlashInfer native CP for MLA decode` 在算子内实现 CP causal | 新增 `enable_dcp/cp_world/cp_rank/cp_kv_cache_interleave_size/causal_seqlens_kv_global` |
| FlashInfer | `feat(mla): support packed low-head and variable-Q decode` | 逐 token 轮转分片，逻辑等价于上面的 `global_k` 公式 |
| vLLM DSA indexer | 本地算 TopK → 跨 rank gather `(score, global_id)` → 稳定全局 TopK → 全局 id 回映本地 | 与本设计的输出协议一致 |

### 1.4 现有算子流水线（arch22）

对外形态：`out = TopK_{visible}( Σ_g w[q,g] · ReLU(qk[q,g,k]) · k_scale[k] )`。

6 个阶段：
1. Cube：QK^T（INT8，`k=128`），fixpipe `DEQF16`（×1/1024）+ ReLU；
2. Vector：`w · q_scale` 广播；
3. Cube：WS（`W` 与 `S` 的 g 维归约）；
4. Vector：乘 `k_scale`、可选 candidate mask；
5. Vector：Sort32 + MergeSort 维护每行 TopK 累加器；
6. Vector：`Extract` 输出 index（本轮新增 value）。

mask 只影响第 1 步“遍历到哪个 S2 块”和第 4 步“每行取多少个 KV”，正是本设计要替换的两级边界。

### 1.5 验收标准

1. `enable_dcp=false` 时，所有既有用例（单例、参数集、candidate、LD 相关 UT）输出与基线逐位一致，性能不劣化；
2. `enable_dcp=true` 时，单 rank 输出与按公式构造的 golden 一致（index 集合 + score）；
3. 多 rank 输出经框架侧合并后，与同一全局序列按 `P=1` 跑的结果集合一致；
4. 覆盖 `C` 落在 `P*S` 边界、末 rank 截断、`L_r=0`、`Q_len>1` 等边界；
5. 不新增 mask、不新增 host 同步。

---

## 2. 接口设计

### 2.1 算子原型新增项

**输入（追加在最后，索引 14）**

| 名称 | 类型 | shape | 说明 |
|---|---|---|---|
| `causal_seqlens_kv_global` | INT32，Device Tensor | `[B]` | 每请求本次 Q 片段的全局 exclusive 上界 `C`。独立请求维度，不是 batch 前缀和，也不是物理 cache 容量 |

**属性（追加在 `key_dequant_scale_stride0` 之后，索引 13–16）**

| 名称 | 类型 | 默认 | 约束 |
|---|---|---|---|
| `enable_dcp` | Bool | `false` | 开启 CP 位置语义；关闭时保持原行为 |
| `cp_world` | Int | `1` | `P ≥ 1` |
| `cp_rank` | Int | `0` | `0 ≤ r < P` |
| `cp_kv_cache_interleave_size` | Int | `1` | `S ≥ 1`，每 `S` 个 token 轮换 rank |

### 2.2 已有参数语义（DCP 下不变）

| 参数 | 语义 |
|---|---|
| `actual_seq_lengths_query` / `cu_seqlens_q` | 沿用既有布局协议；TND 为前缀和，算子自行解析每请求 `Q_len` |
| `actual_seq_lengths_key` / `seqused_k` | **本地**有效 KV 长度 `L`，不能用全局 `C` 替代 |
| `block_table` | 本地逻辑 KV 块 → 物理块；本地逻辑 KV 按全局 token 顺序排列 |
| 物理 `block_size` | 仍取自 `key.shape[1]`；分片粒度 `S` 单独传，避免把“物理块容量”和“rank 轮转粒度”隐式绑定 |
| Q tile 内偏移 | kernel 已有 `s1Offset`/`cuS1BeginIdxPerAiv`，不新增接口参数 |

### 2.3 校验规则

`enable_dcp=true`：

| 项 | 要求 |
|---|---|
| 芯片 | `DAV_2201`（本期） |
| `layout_q` / `layout_k` | `TND` / `PA_BBND` |
| `mask_mode` | `0` 或 `3` |
| `cmp_ratio` | `1`（与上游 vLLM 一致：CP 不与 KV 压缩叠加） |
| `candidate_mode` | `3`（DCP 与两级 candidate 互斥） |
| `return_value` | `1`（跨 rank 全局 TopK 必需 score） |
| `causal_seqlens_kv_global` | 非空、INT32、`numel == B` |
| `cp_world` / `cp_rank` / `S` | `P ≥ 1`、`0 ≤ r < P`、`S ≥ 1` |
| `output_idx_offset` | 必须为空（避免与框架的 local→global 映射重复偏移） |

`enable_dcp=false` 做反向严格校验：`cp_world==1`、`cp_rank==0`、`S==1`、`causal_seqlens_kv_global` 为空，保证既有调用零变化。

### 2.4 绑定面

| 层 | 文件 | 改动 |
|---|---|---|
| 算子原型 | `op_host/vllm_ascend_quant_lightning_indexer_v2_def.cpp` | 追加 1 输入 + 4 属性 |
| Tiling | `op_host/quant_lightning_indexer_v2_tiling.{h,cpp}` | 常量、TilingData、解析与校验 |
| Infershape | `op_host/quant_lightning_indexer_v2_infershape.cpp` | 输入枚举追加，输出 shape 不变 |
| torch 扩展 | `torch_extension/quant_lightning_indexer.py` | 主/candidate schema 追加 kw-only 参数 |
| torch 扩展 C++ | `torch_extension/csrc/quant_lightning_indexer.cpp` | aclnn 调用参数对齐 |
| vllm 适配头 | `quant_lightning_indexer_v2_torch_adpt.h` | `QuantLightningIndexerCandidate` 增加 DCP 参数与 `returnValue` |
| vllm 绑定 | `csrc/torch_binding.cpp` | `npu_quant_lightning_indexer_v2`（compat）与 `_v3`（candidate）追加参数；compat 取消 `return_value==0` 硬校验 |
| 文档 | `docs/aclnnQuantLightningIndexerV2.md`、`README.md`、`examples/` | 参数表、约束、示例 |

新增参数全部**追加在末尾**，旧调用方不传即保持原行为；aclnn 签名变化后旧二进制不可混用，必须整体重编。

---

## 3. 数学语义

### 3.1 单行可见长度（闭式）

对请求内第 `i` 行（0 基，含 Q tile 偏移）：

```
u = max(C - Q_len + i + 1, 0)
visible_len = min(L, (u // (P*S)) * S + clamp(u % (P*S) - r*S, 0, S))
clamp(x, 0, S) = min(max(x, 0), S)
```

推导：全局序列按 `S` 一段轮转分配给 `P` 个 rank。前 `u` 个全局 token 中，完整轮数为 `u // (P*S)`，每轮本 rank 获得 `S` 个；剩余 `u % (P*S)` 个 token 中，本 rank 只在 `[r*S, (r+1)*S)` 区间内持有，数量为 `clamp(u % (P*S) - r*S, 0, S)`。再与本地实际长度 `L` 取小。

### 3.2 算例（issue #7）

`C=257, Q_len=4, P=2, S=128`：

| rank | 本地 KV | 正确可见长度 |
|---|---|---|
| 0 | `k0..k127, k256`，`L=129` | `[128,128,128,129]` |
| 1 | `k128..k255`，`L=128` | `[126,127,128,128]` |

| Q | `u` | 完整轮数 | 余量 | rank 0 | rank 1 |
|---|---|---|---|---|---|
| q253 | 254 | 0 | 254 | 128 | 126 |
| q254 | 255 | 0 | 255 | 128 | 127 |
| q255 | 256 | 1 | 0 | 128 | 128 |
| q256 | 257 | 1 | 1 | 129 | 128 |

### 3.3 两级边界映射到 kernel

| 级别 | 现有公式 | DCP 公式 |
|---|---|---|
| 块级（一个 gS1 行组共用一个上界） | `L - Q_len + s1Offset + s1BaseSize` | `u = C - Q_len + min(s1Offset + s1BaseSize, Q_len)`，再代入 `visible_len` 闭式 |
| 行级（每行候选上界） | `L - Q_len + rowIdx + 1` | `u = C - Q_len + rowIdx + 1`，再代入闭式 |

块级取行组内最后一行的上界（可见长度随 `i` 单调不减），只影响“遍历到哪个 S2 块”，不参与正确性判定；行级决定每行实际读取的 KV 数量。

### 3.4 输出协议与全局 TopK 精确性

- 每个 rank 输出**本地逻辑 KV 序号**的 TopK（最多 `topk` 个），以及对应 score（BF16，无效槽 `-inf`）；
- 框架侧把 `(score, local_idx)` gather 到 CP 组，按 `global_idx = (local_idx // S * P + r) * S + local_idx % S` 转全局，再按 `(score 降序, global_idx 升序)` 做稳定 TopK；
- 精确性：设全局 TopK 集合为 `S*`，则 `S*_r = S* ∩ rank_r` 满足 `|S*_r| ≤ topk`，且 `S*_r` 必是 rank_r 本地 TopK 的子集（否则该 rank 上存在 `≥ topk` 个更高分且全局可见的 token，矛盾）。因此每 rank 取 `topk` 个候选足够；
- 边界并列：TopK 边界分数并列时，本地排序的并列顺序不受约束，跨 rank 合并可能选出不同的并列成员。这是既有 TopK 的已知性质，验收沿用集合比对 + 边界容忍（见 §7.2）。

---

## 4. Host Tiling 设计

### 4.1 TilingData（尾部追加 4×4B）

```
uint32 dcpEnable;
uint32 cpWorld;
uint32 cpRank;
uint32 cpInterleaveSize;
```

`TilingKey` 不变：DCP 是运行期分支，不新增模板实例。

### 4.2 校验实现位置

- `GetOptionalInputParaInfo()`：读取 `causal_seqlens_kv_global`；
- `GetAttrParaInfo()`：读取 4 个新属性；
- `CheckAttrParaInfo()`：DCP 组合校验 + `return_value` 放开（`DAV_2201` 允许 0/1，DCP 强制 1）；
- `CheckShapeDim()` / shape size 校验：`causal_seqlens_kv_global` 维度为 1、元素数等于 `B`；
- `CheckSparseValuesShape()`：`DAV_2201 && return_value==1` 时校验 `sparse_values` 的 N2/topk 维（原逻辑仅 950）；
- TND 输出校验（`sparse_values` dim0 == `qTsize`）同步从 `DAV_3510` 扩展到 `DAV_2201`。

### 4.3 多核切分不变性

arch22 的 metadata 算子（`quant_lightning_indexer_v2_metadata`）只按 `bN2/gS1` 给核分配任务，`s2End` 恒为 0（`supportFd_` 仅 950 为真），kernel 侧 `CalcS2LoopParams()` 在 `s2End==0` 时用 `GetS2BaseBlockNumOnMask()` 自算块数。因此：

- **DCP 不需要修改 metadata 算子**；
- 核间负载均衡使用的是非 DCP 的 cost 估计，DCP 下可能不最优（各 rank 实际可见长度不同），但不影响正确性；作为后续性能项（§10）。

arch35（950）的 FD/LD 会把 metadata 的 `s2End` 当权威上界，DCP 必须先让 metadata AICPU 感知 `C/P/r/S`，否则按非 DCP 上界切块会漏算——本期不做。

### 4.4 Workspace

无增量。arch22 的 workspace 仍为 `2*s1*s2*4B + w workspace + V1_DECODE` 区域。

---

## 5. Kernel 设计（arch22）

### 5.1 新增状态

| 位置 | 字段 |
|---|---|
| `QLIV2Common::ConstInfo` | `bool dcpEnable`、`uint32_t cpWorld`、`cpRank`、`cpInterleaveSize` |
| `QLIV2Common::RunInfo` | `uint32_t causalGlobalLen`（当前 bIdx 的 `C`） |
| `QLIV2Preload` | `GlobalTensor<int32_t> causalSeqLensKvGlobalGm`（允许 nullptr） |

`InitTilingData()` 从 TilingData 填充；`Init()` 新增一个 `__gm__ uint8_t *causalSeqlensKvGlobal` 入参（可为 nullptr）。

### 5.2 可见长度计算

kernel 内共用辅助（AIC/AIV 均可用，纯标量、无 UB 依赖）：

```cpp
__aicore__ inline uint32_t GetDcpVisibleLen(uint32_t actS1Size, uint32_t rowIdx,
                                            uint32_t localS2Len, uint32_t causalGlobalLen)
{
    int64_t u = static_cast<int64_t>(causalGlobalLen) - static_cast<int64_t>(actS1Size) +
                static_cast<int64_t>(rowIdx) + 1;
    if (u <= 0) {
        return 0;
    }
    uint64_t period = static_cast<uint64_t>(constInfo.cpWorld) * constInfo.cpInterleaveSize;
    uint64_t full = (static_cast<uint64_t>(u) / period) * constInfo.cpInterleaveSize;
    int64_t rem = static_cast<int64_t>(static_cast<uint64_t>(u) % period) -
                  static_cast<int64_t>(constInfo.cpRank) * constInfo.cpInterleaveSize;
    uint64_t tail = rem <= 0 ? 0 : static_cast<uint64_t>(Min(rem, static_cast<int64_t>(constInfo.cpInterleaveSize)));
    return static_cast<uint32_t>(Min(static_cast<uint64_t>(localS2Len), full + tail));
}
```

全程 int64/uint64 中间量，避免 `P*S` 与 `C` 相加溢出。

### 5.3 块级边界：`GetS2BaseBlockNumOnMask()`

```cpp
if (constInfo.dcpEnable) {
    // 行组最后一行（含尾部 padding 行时按实际 Q_len 截断）
    uint32_t lastRow = Min(s1Offset + constInfo.s1BaseSize, actS1Size);
    validS2Len = GetDcpVisibleLen(actS1Size, lastRow, actS2Size, causalGlobalLen);
    validS2Len = Max(validS2Len, 1);   // 保持既有“至少 1 块”的不变量
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}
// 原公式（cmpRatio 归一化）保持不变
```

`GetTotalBaseBlockNum()` 需要按 `bIdx` 取 `C`：在循环内先 `GetS1S2ActualSeqLen(bIdx, ...)`，再读取 `causalSeqLensKvGlobalGm.GetValue(bIdx)` 传入。

### 5.4 行级边界：`ProcessVec1()`

- 非 DCP：保持现有累加逻辑（`cuRealAcSeqIni + count` 后除 `cmpRatio`）；
- DCP：每行现算
  ```cpp
  int32_t cuRealAcSeq = GetDcpVisibleLen(info.actS1Size, cuS1BeginIdxPerAiv + innerS1Idx,
                                         info.actS2Size, info.causalGlobalLen);
  ```

### 5.5 零可见行（关键边界）

DCP 下 `L > 0` 但某行 `visible_len == 0` 是常态（Q 片段尚未轮转到本 rank 的 token）。`ProcessVec1()` 已有 `cuRealAcSeq <= 0` 分支，会调用 `CleanInvalidOutput()` 把该行 index 全写 `-1`；本设计只需：

- 在 `CleanInvalidOutput()` 内补写 `sparse_values = -inf`；
- 新增的每行 `cuRealAcSeq` 计算保证该分支在 DCP 下能被正确触发（非 DCP 路径不变）。

arch22 上 `isNeedLD` 恒为 false（metadata `s2Start/s2End` 为 0），因此不需要为零可见行补 LD workspace 条目；代码中加注释固化该前提。

### 5.6 arch22 `return_value=1`

现有限制：`DAV_2201` 直接拒绝 `return_value=1`，`sparse_values` 恒为空。DCP 跨 rank 全局 TopK 必需 score，因此本期打开：

| 位置 | 改动 |
|---|---|
| `ProcessVec1()` 非 LD 路径 | `ExtractIndex` → `Extract(valueUb, idxUb, pairBuf, BASE_TOPK/32)` 一次取回 value/index；value 由 FP32 `Cast` 到 BF16 后按与 index 相同的 GM 偏移写出 |
| `ProcessLD()` | 已有 `Extract(outValueUb, outIdxUb, ...)`，补 BF16 转换与 copy-out |
| `CleanInvalidOutput()` / `WriteInvalidRow()` / `ProcessInvalid()` | 同步填 `sparse_values = -inf` |
| `DealActSeqLenIsZero()` | 走 `CleanInvalidOutput()`，自动覆盖 |
| UB | `outQueue_` 8KB → 16KB（idx 8KB + value 8KB）。非 candidate 总占用 144→152KB；candidate 与 DCP 互斥，不会叠加 |

无效槽天然是 `-inf`：排序缓冲的 padding 值就是 `NEG_INF = 0xFF800000`（FP32 负无穷），`Extract` 后直接转 BF16 仍为 `-inf`。

风险：`Cast(fp32 → bfloat16_t)` 在 arch22 的可用性需实测；若不可用，退化为 `Cast(half)` → `Cast(bfloat16_t)` 两步。

### 5.7 明确不改的部分

- `cmp_residual_k` / 压缩路径：DCP 下 `cmp_ratio==1`，不走；
- `output_idx_offset`：DCP 下禁止传入；
- candidate 相关 buffer 与分支：DCP 下 `candidate_mode==3`，不分配；
- metadata 算子、workspace、TilingKey。

### 5.8 性能预期

- 每行新增约 10 条标量指令（除法/取模可被常量折叠优化），相对单行排序/归约可忽略；
- 值输出每行新增一次 `Extract` + 一次 `Cast` + 一次 copy-out，估计矢量耗时 +1% 量级；
- rank `r>0` 的可见长度通常远小于 `L`，块级遍历块数下降，整体是净收益；
- `enable_dcp=false` 分支不改变任何指令路径。

---

## 6. 框架侧契约（本期不改，供对接）

1. **KV cache 布局**：rank `r` 只持有 `owner(t) = (t // S) % P == r` 的 token，本地逻辑序号按时间序排列；物理块通过本地 `block_table` 寻址；
2. **长度**：`seqused_k[b] = L_r`（本地）；`causal_seqlens_kv_global[b] = C`（全局 exclusive，Device Tensor）；
3. **调用**：`enable_dcp=true`、`cp_world=P`、`cp_rank=r`、`cp_kv_cache_interleave_size=S`、`return_value=1`；
4. **合并**：CP 组内 all-gather `(score, local_idx)`；`global_idx = (local_idx // S * P + r) * S + local_idx % S`；按 `(score 降序, global_idx 升序)` 稳定 TopK；
5. **回映**：稀疏 attention 算子需要本地序号时，把全局 TopK 中属于本 rank 的项压缩到前缀，其余填 `-1`（与上游 `triton_filter_and_convert_dcp_index` 等价）；
6. **精度**：合并使用 BF16 score，与现有 `return_value` 契约一致；若需严格 FP32 排序，需后续增加 FP32 score 输出。

---

## 7. 验证计划

### 7.1 Golden

扩展 `tests/pytest/quant_lightning_indexer_v2_golden.py`：

- 新增 `cp_world / cp_rank / cp_kv_cache_interleave_size / causal_seqlens_kv_global`；
- 输入构造：先生成全局 K（长度 `C`），按 `owner(t)` 取本 rank 的本地 K，保持时间序；本地 `L=len(local K)`；本地 `block_table` 随机置换；
- 可见长度用 §3.1 闭式；
- 输出比对：每 rank 的 index（本地序号）+ value（BF16）；
- 端到端：各 rank 输出按 §6 合并，与同一全局序列按 `P=1` 的结果做集合比对。

### 7.2 比对策略

沿用 `result_compare_method.py` 的集合比对 + 边界容忍：TopK 结果按行比较集合，边界分数并列时允许成员差异；`sparse_values` 用 `check_result_return_value` 按 index gather 后比对。

### 7.3 测试矩阵

| 维度 | 取值 |
|---|---|
| `P` | 2, 4 |
| `S` | 1, 16, 128（含 `S == block_size`） |
| `Q_len` | 1, 2, 4 |
| 全局长度 `C` | 127/128/129、255/256/257、511/512、1024 |
| 不均衡 `L` | 末 rank 截断；`L_r == 0`（`C < r*S`） |
| `topk` | 512, 2048 |
| `mask_mode` | 0, 3 |
| 布局 | TND + PA_BBND |

### 7.4 入口

- `test_quant_lightning_indexer_v2_paramset.py` 新增 `quant_li_dcp` 参数集，并补 `Ascend910B` 分发分支；
- 新增 `tests/pytest/test_quant_lightning_indexer_v2_dcp.py`：逐 rank 子进程隔离执行 + 离线合并比对；
- Host UT：DCP 合法/非法组合、TilingData 字段、`causal_seqlens_kv_global` shape 校验；
- 非 DCP 全量回归：单例、参数集、candidate 套件零变化。

### 7.5 构建与验证顺序（AGENTS 全局规则第 2 条）

1. 设计文档；
2. op_host + UT 编译；
3. kernel 编译，核对 `.run` mtime 与 “package generated”；
4. 安装后在部署路径 grep `enable_dcp` 确认新代码生效；
5. 910B3 跑 DCP 用例 + 非 DCP 回归；
6. （可选）性能抽查：`enable_dcp=false` 不劣化；DCP 各 rank 耗时与可见长度成比例。

---

## 8. 风险与开放项

| # | 风险/开放项 | 影响 | 处理 |
|---|---|---|---|
| R1 | BF16 score 合并精度 | 分差极小时全局 TopK 顺序可能变化 | 沿用既有 BF16 契约；集合 + 边界容忍比对；后续可加 FP32 输出 |
| R2 | 零可见行漏写 | GM 脏数据 | 新增 `WriteInvalidRow`，严格 `dcpEnable` 分支 |
| R3 | `Cast` 到 BF16 的 arch22 支持 | 编译/精度 | 先最小验证，失败走 half 中转 |
| R4 | aclnn 签名变化 | 旧包/新包不可混用 | 全部调用点同步重编，安装后 grep 校验 |
| R5 | 核间负载均衡仍按非 DCP 估计 | DCP 下略有偏斜 | 不影响正确性；后续可让 metadata 感知 DCP |
| R6 | arch35 不支持 | 950 无法使用 | 需先扩展 metadata AICPU（`C` 输入 + DCP 版 s2 切分） |
| R7 | DCP + candidate 不支持 | 两级候选场景不可用 | 首版校验拒绝；块级候选需新增块分数协议 |
| R8 | 并列分数跨 rank 合并 | 并列成员可能不同 | 已知性质，验收用集合比对；如需确定性，需在本地 TopK 引入全局 id 作为次级键 |

---

## 9. 文件级实施拆解

| 序 | 文件 | 内容 |
|---|---|---|
| 1 | `docs/qli_v2_dcp_design.md` | 本文档 |
| 2 | `op_host/vllm_ascend_quant_lightning_indexer_v2_def.cpp` | 1 输入 + 4 属性 |
| 3 | `op_host/quant_lightning_indexer_v2_tiling.h` | 索引常量、TilingData、ParaInfo、TilingInfo |
| 4 | `op_host/quant_lightning_indexer_v2_tiling.cpp` | 解析、校验、return_value 放开、赋值 |
| 5 | `op_host/quant_lightning_indexer_v2_infershape.cpp` | 输入枚举（输出不变） |
| 6 | `torch_extension/quant_lightning_indexer.py` | schema/Fake/Impl/适配层 |
| 7 | `torch_extension/csrc/quant_lightning_indexer.cpp` | aclnn 调用参数 |
| 8 | `quant_lightning_indexer_v2_torch_adpt.h` | candidate 入口参数 + returnValue |
| 9 | `csrc/torch_binding.cpp` | v2/v3 绑定与 schema |
| 10 | `op_kernel/quant_lightning_indexer_v2.cpp` | kernel 入口参数与 INVOKE 宏 |
| 11 | `op_kernel/arch22/quant_lightning_indexer_v2_common_arch22.h` | ConstInfo/RunInfo |
| 12 | `op_kernel/arch22/quant_lightning_indexer_v2_kernel_arch22.h` | GM、Init、两级边界 |
| 13 | `op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h` | 行级边界、零可见行、score 输出、UB |
| 14 | `tests/pytest/quant_lightning_indexer_v2_golden.py` | DCP golden |
| 15 | `tests/pytest/test_quant_lightning_indexer_v2_paramset.py` | DCP 参数集与 910B 分发 |
| 16 | `tests/pytest/test_quant_lightning_indexer_v2_dcp.py` | 新增多 rank 测试 |
| 17 | `tests/ut/op_host/*` | DCP tiling UT |
| 18 | `docs/aclnnQuantLightningIndexerV2.md`、`README.md` | 接口文档 |

---

## 10. 当前验证状态（2026-10-26）

### 10.1 编译

| 项 | 结果 |
|---|---|
| 目标算子构建（`ASCEND_OP_NAME=quant_lightning_indexer_v2;quant_lightning_indexer_v2_metadata`，`ascend910b`，CANN 9.2.0） | 退出码 0，日志 0 个 `error:` |
| 产物 | `binary/ascend910b/bin/vllm_ascend_quant_lightning_indexer_v2` 与 `...metadata`，`.o` mtime 为本次构建时间 |
| vllm 扩展（`torch_binding.cpp` + `torch_extension` 适配头） | 全量编译通过（0 error；仅有既有 narrowing 警告） |

### 10.2 910B3 实机

环境：Ascend 910B3，CANN 9.2.0，torch 2.10.0 / torch_npu 2.10.0.post4，算子包安装到临时 vendor，经 `ASCEND_CUSTOM_OPP_PATH` 加载。

| 用例 | 结果 |
|---|---|
| DCP 6 组：`P=2/S=1`、`P=2/S=128/Q_len=4`、`P=4/S=16/Q_len=2`、`P=2/S=128/B=2`、`P=4/S=128/L_r=0`、`P=2/S=64/Q_len=4/topk=2048` | 通过 |
| DCP + `mask_mode=0` | 通过 |
| 非 DCP 回归（3 组）：`enable_dcp=false` 与 `P=1` DCP 逐位一致，`return_value=0` 时 values 为空 | 通过 |
| 非 DCP `return_value=1`：index 集合与 BF16 score 对齐 golden | 通过 |
| 合计 | **11 passed** |

测试命令（最小扩展方式）：

```bash
QLI_DCP_EXT=/path/to/qli_min_ext.so \
ASCEND_CUSTOM_OPP_PATH=/path/to/vendors/custom_transformer \
LD_LIBRARY_PATH=/path/to/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH \
python -m pytest -q csrc/attention/quant_lightning_indexer_v2/tests/pytest/test_quant_lightning_indexer_v2_dcp.py
```

### 10.3 未覆盖项

| 项 | 说明 |
|---|---|
| host UT | 用例已加入 `tests/ut/op_host/arch22/...`，但本 clone 缺 `tests/ut/framework_normal`，无法在本仓构建；需在完整 ops-transformer 仓执行 |
| candidate 模式 | 本次未改 candidate 内核路径（`dcpEnable=false`、`valueOutBuf_` 不分配），最小验证扩展固定 `candidate_mode=3`；完整回归需在 ops-transformer 的 candidate 套件执行 |
| 框架侧 pip 接入 | 不在本期（见 §6 契约） |
| arch35（950） | 不支持，需先扩展 metadata AICPU（§4.3） |
