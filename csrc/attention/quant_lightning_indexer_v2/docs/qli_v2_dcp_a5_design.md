# QuantLightningIndexerV2 DCP（解码上下文并行）特性设计 — arch35 / Ascend950（A5）

| 项 | 内容 |
|---|---|
| 算子 | `VllmAscendQuantLightningIndexerV2`（`csrc/attention/quant_lightning_indexer_v2`） |
| 关联算子 | `VllmAscendQuantLightningIndexerV2Metadata`（`csrc/attention/quant_lightning_indexer_v2_metadata`） |
| 目标芯片 | arch35：`DAV_3510` / `ascend950` / Ascend950PR（A5） |
| 基线实现 | arch22 DCP（提交 `a16aae5c8`；FD/LD 门禁与 LD 补写修复见 `233c229fd`/`f54a82372`，`docs/qli_v2_dcp_design.md`） |
| 范围 | 仅算子层（csrc）；框架侧 `dsa_cp.py` 接线不在本期 |
| 状态 | 已实现，A5 实机 DCP（`return_value=1`）测试 **33 passed**，含大 shape、P=8/16、topk=8192、连续 block_table 与 A2 既有 LD 用例（2026-10-28，基线 `f54a82372`） |
| 关键词 | DCP（Decode Context Parallel）、CP、PA、TND、TopK、FD（跨核归约）、LD |

---

## 1. 需求分析

### 1.1 问题

A2/A3（arch22）与 A5（arch35）的 DCP 实现是两套不同的分工：

- **A2/A3**：metadata 仍按本地完整 L 切分（FD 时 `s2End` 是块游标，不做 DCP 换算），kernel 在块级/行级
  用闭式公式自行裁剪可见长度；`enable_dcp=1` 仅用于在 A2/A3 上打开 FD/LD（要求 `cmp_ratio=1`）。
  FD 打开后 metadata 的切分范围可能包含整段不可见块，kernel 对「首个分片 ≤ 可见块数」和
  「末分片 `s2End==0` 表示本地完整 L」分别处理，避免漏写归约 partial（上游 `f54a82372`）。
- **A5（arch35）**：`supportFd_` 在 `maxVisibleS2 >= topk` 且 `maxVisibleS2 > 5*s2BaseSize_` 时为真
  （`maxVisibleS2` 是 DCP 可见长度的 batch 最大值，见 §6.4；非 DCP 时退化为本地 L），
  metadata 会把行组按块切分给多个核；arch35 kernel 的 `CalcS2LoopParams()` 在核尾行组直接使用
  `splitCoreInfo.s2End` 作为循环上界，把 metadata 的 `s2End` 当权威值。非 DCP 的组上界是
  `L - Q_len + row`；DCP 可见长度在长 Q chunk 的早期行组可远大于该值（例如
  `C=1024, P=2, S=1, Q_len=512` 时行 100 的 DCP 可见长度约 307，非 DCP 只有 101）。
  因此 A5 上只移植 kernel 的闭式公式会漏算，必须让 metadata AICPU 用同一套公式计算每行组的块区间，
  并让 FD/LD 的切分直接消费该区间。

### 1.2 输入输出定义

| 项目 | 内容 |
|---|---|
| 新增输入 | `causal_seqlens_kv_global`：INT32、Device Tensor、`[B]`，每请求本次 Q 片段的全局 exclusive 上界 `C` |
| 新增属性 | `enable_dcp=false`、`cp_world=1`、`cp_rank=0`、`cp_kv_cache_interleave_size=1` |
| 已有输入 | `query`、`key`、`weights`、`query_dequant_scale`、`key_dequant_scale`、`cu_seqlens_q`、`seqused_k`、`block_table`、`metadata` |
| 输出 | `sparse_indices` INT32；`return_value=1` 时 `sparse_values` BF16（无效槽 `-inf`、index `-1`） |
| 布局 | 本期 DCP 只开放 `layout_q=TND`、`layout_k=PA_BBND` |
| dtype | A5 支持 `quant_mode 1..5`；DCP 逻辑与 dtype 无关 |
| shape 约束 | `seqused_k[b]=L` 是本地有效 KV 长度；`C` 是全局值，不能用 `L` 代替 |

### 1.3 数学语义

对请求内第 `i` 个 Q 行（0 基，含 Q tile 偏移）：

```
global_q = C - Q_len + i
u = max(global_q + 1, 0)
visible_len = min(L, (u // (P*S)) * S + clamp(u % (P*S) - r*S, 0, S))
```

块级取行组最后一行（可见长度随行号单调不减）；行级逐行现算。零可见行必须输出 `index=-1`、`score=-inf`。

### 1.4 校验规则

`enable_dcp=true`：

| 项 | 要求 |
|---|---|
| 芯片 | `DAV_2201` 或 `DAV_3510` |
| `layout_q` / `layout_k` | `TND` / `PA_BBND` |
| `mask_mode` | `0` 或 `3` |
| `cmp_ratio` | `1` |
| `candidate_mode` | `3`（off） |
| `return_value` | `1` |
| `causal_seqlens_kv_global` | 非空、INT32、`numel == B` |
| `cp_world` / `cp_rank` / `S` | `P>=1`、`0<=r<P`、`S>=1` |
| `output_idx_offset` | 必须为空 |

`enable_dcp=false` 做反向严格校验：`P==1`、`r==0`、`S==1`、`C` 为空。

### 1.5 验收标准

1. `enable_dcp=false`：A5 既有用例输出与基线一致，性能不劣化；
2. `enable_dcp=true`：单 rank 的 index 集合与 BF16 score 对齐 golden；
3. 多 rank 按 `global_idx=(local//S*P+r)*S+local%S` 合并后，与 `P=1` 全局结果集合一致；
4. 覆盖 `C` 落在 `P*S` 边界、末 rank 截断、`L_r=0`、`Q_len>1`、`mask_mode=0/3`、`topk=8192`；
5. metadata 与 kernel 对同一行组的 DCP 块数完全一致。

---

## 2. 芯片执行模型与 C/V 数据通路

| 模块 | 本设计的关注点 |
|---|---|
| Cube/AIC | QK^T 的 S2 遍历范围由块级上界决定；MatMul 内部不变 |
| Vector/AIV | `ProcessTopK` 逐行可见长度、零可见行清理、BF16 score 输出 |
| Scalar | metadata AICPU 的块区间/cost 计算；kernel 的闭式标量计算 |
| MTE | 不新增搬运；kScale 预取的 `validS2Len` 随块级上界变化 |
| GM | `C` 为 Device Tensor，kernel 逐请求标量读取；metadata AICPU 直接读 host 可见的 Device 数据 |
| UB/L1/L0 | 无增量 |

数据通路不变：

```
GM 输入 -> Vector 预处理 -> Cube QK -> Vector scale/mask/candidate -> Vector TopK -> GM 输出
```

DCP 只改变“Cube 遍历到哪个 S2 块”和“Vector 每行取多少 KV”。

---

## 3. 算法阶段拆解

| 阶段 | 是否受影响 | 设计 |
|---|---|---|
| 输入解析 | 是 | kernel 读 `C/P/r/S`；metadata AICPU 读 `C/P/r/S`；`seqused_k` 仍是本地 `L` |
| 预处理 | 否 | scale/layout 解析不变 |
| 第一段 MatMul（QK^T） | 否 | 只改 S2 块数 |
| score 后处理 | 是 | 块级上界改 DCP 闭式；kScale 预取的 `validS2Len` 同步 |
| 归约/softmax | 否 | 无 softmax；WS 归约不变 |
| 选择/稀疏化 | 是 | arch35 在 `ProcessTopK` 逐行算 DCP 可见长度 |
| 第二段 MatMul | 否 | 无 |
| 输出后处理 | 复用 | A5 已支持 `return_value`；`CleanInvalidOutput` 已写 `-1/-inf` |
| metadata AICPU | 是 | DCP-aware 的 `s2CmpLength/s2End`、cost、FD/LD 切分 |

中间结果验证：可选解码 metadata 的 `s2End`，与按公式计算的每行组块数比对；开发期可用非 DCP 路径做逐位回归。

---

## 4. Host Tiling 设计

### 4.1 维度建模

B/N2/G/S1/S2/D/block 维度不变。`C` 是请求级标量，不参与维度切分。

### 4.2 多核切分

A5 仍由 metadata AICPU 按 `bN2/gS1` 分配任务，FD 时按 S2 块切分。DCP 只改变每行组的 S2 上界与 cost，不改变分配框架。

### 4.3 TilingData 与 TilingKey

`QLIV2TilingData` 已包含：

```
uint32 dcpEnable;
uint32 cpWorld;
uint32 cpRank;
uint32 cpInterleaveSize;
```

不新增字段；`TilingKey` 不变（DCP 是运行期分支，禁止新增模板实例）。

### 4.4 Host 校验

唯一改动：`op_host/quant_lightning_indexer_v2_tiling.cpp:501-503` 放开 `DAV_3510`。其余 DCP 校验保持；`C` 的 dtype/shape 校验和 `sparse_values` 形状校验已覆盖 `DAV_3510`。

### 4.5 Buffer / Workspace

- UB/L1/L0 无增量；
- Workspace 无增量；
- 无新增同步点。

---

## 5. Kernel 设计（arch35）

### 5.1 新增状态

| 位置 | 字段 |
|---|---|
| `QLIV2Common::ConstInfo` | `bool dcpEnable`、`uint32_t cpWorld/cpRank/cpInterleaveSize` |
| `QLIV2Common::RunInfo` | `uint32_t causalGlobalLen` |
| `TempLoopInfo` | `uint32_t causalGlobalLen` |
| `QLIV2Preload` | `GlobalTensor<int32_t> causalSeqLensKvGlobalGm`、`bool hasCausalGlobalLen` |

`InitTilingData()` 从 TilingData 填充；`Init()` 新增 `__gm__ uint8_t *causalSeqlensKvGlobal`（可为 nullptr）。

### 5.2 共享公式

新增纯 C++ 头 `op_kernel/quant_lightning_indexer_v2_dcp_common.h`，供 arch35 与 metadata AICPU 使用：

```cpp
namespace QLIV2Dcp {
QLI_DCP_INLINE uint32_t CalcDcpVisibleLen(uint32_t u, uint32_t localS2Len,
                                          uint32_t cpWorld, uint32_t cpRank,
                                          uint32_t cpInterleaveSize);
}
```

arch22 的 `QLIV2Common::CalcDcpVisibleLen` 保持原样，不迁移；A2/A5 共用 golden 防漂移。

### 5.3 块级边界

`GetS2BaseBlockNumOnMask()`：

```cpp
uint32_t lastRowCount = Min(s1Offset + constInfo.s1BaseSize, actS1Size);
int64_t u = (int64_t)causalGlobalLen - actS1Size + lastRowCount;
uint32_t visible = QLIV2Dcp::CalcDcpVisibleLen(u<=0?0:u, actS2Size,
                                               cpWorld, cpRank, cpInterleave);
visible = Max(visible, 1U);           // 保持“至少 1 块”的不变量
tempLoopInfo.validS2Len = visible;
return (visible + s2BaseSize - 1) / s2BaseSize;
```

`SplitCoreByAICPU()` 的 `s2End==0` 重算路径和 `CalcS2LoopParams()` 都传 `GetGlobalCausalLen(bIdx)`。

### 5.4 行级边界

`ProcessTopK`：

```cpp
if (dcpEnable && attenMaskFlag) {
    int64_t u = causalGlobalLen - actS1Size + curAivS1Idx + i + 1;
    validAllS2Len = CalcDcpVisibleLen(u<=0?0:u, actS2Size, P, r, S) / cmpRatio;
} else if (attenMaskFlag) {
    validAllS2Len = (i + cuRealAcSeq) / cmpRatio;
}
```

`mask_mode=0` 不走 DCP 行级分支（本地全可见）。零可见行走现有 `validS2Len <= 0` 分支，写 `-1/-inf`。

### 5.5 输出

A5 已原生支持 `return_value`：
- `CleanInvalidOutput()` 写 index=-1、value=-inf；
- `ProcessTopK` 主路径和 LD 路径都输出 BF16 score；
- `ProcessInvalid/DealActSeqLenIsZero` 复用。

### 5.6 MTE2 搬运优化（PA / FP8 主路径，与 DCP 正交）

- `KeyNd2NzForPA()`：当 `keyStride0 == kCacheBlockSize * D`（key 0 轴紧凑）且相邻逻辑块映射到连续物理块时，
  把相邻块的 Nd2Nz 合并成一次搬运，减少小粒度 MTE2 指令；
- `ComputeMm1()`：首次 S2 块（`isFirstS2InnerLoop && s2GmOffset == 0`）的 `MTE2_MTE1` 握手由同迭代的
  query 分支统一发出，删除重复的一对 set/wait；
- 两项均不改变计算语义；同包 msprof A/B（DP4TP8DCP8EP32 bs128 decode）kernel 77.89us vs 79.92us（约 -2.5%），
  MTE2 pipe 61.46us vs 62.64us；连续 block_table 用例覆盖合并路径。

---

## 6. Metadata AICPU 设计

### 6.1 接口

新增可选输入 `causal_seqlens_kv_global` 和 4 个属性。所有参数追加在末尾并给默认值，旧调用零变化。

### 6.2 DCP-aware 块区间

`CalcS2TokenRange()` 在 `attentionMode_==1 && enable_dcp_` 时：

```cpp
int64_t u = (int64_t)C - (int64_t)s1Size + s1LastToken + 1;
uint32_t visible = QLIV2Dcp::CalcDcpVisibleLen(u<=0?0:u, revertS2Size, P, r, S);
// 与 kernel 一致：本地 L>0 时至少保留 1 个块
uint32_t visibleForBlock = (visible == 0 && revertS2Size > 0) ? 1 : visible;
if (visibleForBlock == 0) return {0, -1};
return {0, (int64_t)visibleForBlock - 1};
```

`CalcS1GCache()`：
- `s2TokenRange.second < 0` 时 `s2CmpLength=0`、`s2Start=s2End=0`；
- 否则沿用现有 clip 和 `s2CmpLength=(s2Last-s2First+1)/cmpRatio`；
- cost、尾块、FD 记录沿用，自动 DCP-aware。

`mask_mode=0` 保持 `attentionMode_==0` 的全本地可见路径。`CalcSplitInfo()` 的 `s2BaseNum` 仍按本地 L 计算（只用于空 batch 判定）；FD 阈值改用 DCP 可见长度，见 §6.4。

### 6.3 FD/LD 一致性

- FD 开：metadata 输出的每核 `[s2Start,s2End]` 已是 DCP-aware，kernel 截断到同一值；
- FD 关：metadata `s2End=0`，kernel 用同一闭式重算；
- `RecordFDInfo/SplitFD/AssignBlockToCore` 不改。

### 6.4 FD 阈值 DCP 化

`CalcSplitInfo()` 的 `supportFd_` 判据改为使用每个 batch 的 DCP 最大可见长度：

- 可见长度对 `u` 单调，batch 内最大值在最后一行（`u = C`），因此
  `maxVisible = max_b CalcDcpVisibleLen(C_b, L_b, P, r, S)`，不需要遍历行组；
- 仅当 `enable_dcp=1 && attentionMode_ != 0 && SoC=950` 时使用该值；非 DCP、`mask_mode=0`、
  A2/A3 仍用本地 L（A2/A3 的 metadata 不做 DCP 换算，见 §1）；
- 判据本身不变：`maxVisible >= topk && maxVisible > 5*s2BaseSize_`；
- 目的：`seqused_k` padding 很大、实际可见很小时不再白开 FD 和 S2 切分。

状态：代码已改、编译通过；A5 实机的阈值边界（`maxVisible = 5*s2BaseSize_ ± 1`、`< topk`、`=0`）
需要在 950 上补测，本机只有 910B3。

---

## 7. 接口与绑定

| 层 | 文件 | 改动 |
|---|---|---|
| 主算子原型 | 已有 | 不改（A2 提交已完成） |
| metadata proto | `op_graph/quant_lightning_indexer_v2_metadata_proto.h` | 1 输入 + 4 属性 |
| metadata aclnn | `op_host/op_api/aclnn_quant_lightning_indexer_v2_metadata.{h,cpp}` | 签名、DFX、l0op |
| metadata l0op | `op_host/op_api/quant_lightning_indexer_v2_metadata.{h,cpp}` | 签名、attr 注册 |
| metadata check | `op_host/quant_lightning_indexer_v2_metadata_check.h` | DCP 校验 |
| metadata AICPU | `op_kernel_aicpu/*` | 输入/属性、DCP 块区间 |
| vllm binding | `csrc/torch_binding.cpp` | metadata schema/实现 |
| vllm meta | `csrc/torch_binding_meta.cpp` | Meta schema |
| vllm 适配头 | `quant_lightning_indexer_v2_torch_adpt.h` | metadata 参数 |
| standalone 扩展 | `torch_extension/quant_lightning_indexer.py`、`csrc/quant_lightning_indexer.cpp` | schema/Fake/aclnn |
| graph converter | `graph_convert_quant_lightning_indexer.py` | 占位签名同步（如引用） |

---

## 8. 验证计划

### 8.1 Golden

复用 `tests/pytest/quant_lightning_indexer_v2_golden.py` 的 DCP 公式（A2/A5 共用），无需为 A5 新写 golden。

### 8.2 pytest

`tests/pytest/test_quant_lightning_indexer_v2_dcp.py`：
- `_build_metadata` 传入 `C/P/r/S`（`enable_dcp=False` 时 C 为空、cp 保持默认）；
- 复用 10 组基础 DCP 用例（含 P=8/16 大 C）；
- A5 专属：`topk=8192`、`LARGE_CASES`（大 C/S、多 batch、长 Q_len）、`HIGH_P_CASES`（P=8/16，覆盖 S=1/64/128 与 FD 开/关）；
- `test_qliv2_contiguous_pa_block_table`：连续物理块，覆盖 `KeyNd2NzForPA` 相邻块合并；
- `mask_mode=0`、多 rank 合并 == `P=1`、非 DCP 回归；
- A2 既有 LD 用例的 A5 适配：`ld_invisible_segments` 在 A5 上不强制 `qld==1`（metadata 已裁掉不可见段，单块无需跨核归约），`ld_gate_non_dcp_vs_dcp` 在 A5 上期望非 DCP 长 S2 仍开 FD/LD（与上游“950 维持原有行为”一致）。

### 8.3 Host UT

`tests/ut/op_host/arch35/test_quant_lightning_indexer_v2_tiling.cpp` 新增 DCP 合法/非法组合，对照 arch22 `tests/ut/op_host/arch22/test_quant_lightning_indexer_v2_tiling.cpp:194-298`。

### 8.4 实机

- 设备：Ascend950PR，`ASCEND_RT_VISIBLE_DEVICES=3`；
- 算子包：`bash csrc/build.sh --pkg --ops="quant_lightning_indexer_v2;quant_lightning_indexer_v2_metadata" --soc=ascend950`；
- 扩展：仓外最小扩展（`QLI_DCP_EXT`），不依赖 vllm；
- 记录：编译 0 error、pytest 通过列表、设备/CANN/torch_npu 版本、metadata `s2End` 检查结果。

### 8.5 性能

- `enable_dcp=false` 不劣化；
- DCP 各 rank 的块遍历数应与可见长度成比例，rank `r>0` 通常净收益。

---

## 9. 风险与开放项

| # | 风险 | 处理 |
|---|---|---|
| R1 | metadata 与 kernel 块数不一致 | 共用 `QLIV2Dcp::CalcDcpVisibleLen`；至少 1 块不变量；metadata `s2End` 断言 |
| R2 | 零可见行导致 `s2End=0` 歧义 | `visible==0 && L>0` 时按 1 块；行级输出 -1/-inf |
| R3 | FD 负载偏斜 | cost 自动 DCP-aware；不改 FD 框架 |
| R4 | `cmp_ratio=1` 与框架当前 4 冲突 | 本期仅算子层，文档明确 |
| R5 | A5 环境依赖 | 最小扩展，不装 vllm；torch_npu 用华为镜像 |
| R6 | arch22 与共享头两份实现漂移 | arch22 不动；A2/A5 共用 golden；合并前 A2 补跑 |
| R7 | BF16 并列边界 | 沿用集合比对 + 边界容忍 |
| R8 | `s2End==0` 重算路径 | 传 `C`，复用同一函数；FD 开/关覆盖 |

---

## 10. 实机验证结果（2026-10-28，基线 `f54a82372`）

环境：Ascend950PR_9579（soc_version=260）、CANN 9.2.0-beta.2、torch 2.10.0+cpu、torch_npu 2.10.0。

运行方式（仓外最小扩展，不安装 vllm）：

```bash
QLI_DCP_EXT=<qli_min_ext.so> ASCEND_RT_VISIBLE_DEVICES=4 \
python -m pytest -v -s csrc/attention/quant_lightning_indexer_v2/tests/pytest/test_quant_lightning_indexer_v2_dcp.py \
  -k "not non_dcp_regression"
# 33 passed, 3 deselected
```

| 项 | 结果 |
|---|---|
| ascend950 算子包编译 | exit 0，`quant_lightning_indexer_v2` 与 `quant_lightning_indexer_v2_metadata` 均生成 |
| DCP pytest（仅 `return_value=1`） | **33 passed**（`-k "not non_dcp_regression"`，排除 `return_value=0` 的 3 个非 DCP 回归 case） |
| 基础 DCP | P=2/4/8/16、S=1/16/64/128/256、Q_len=1/2/4、C=127/257/513/2049、B=1/2/4、L_r=0、mask_mode=0/3，全部通过 |
| A5 大 shape | `C=32769/topk=8192`（本地 L=16385）、`S=512`、`B=4`、`Q_len=16`，全部通过 |
| A5 高 P | P=8/16 共 4 组：`P=8/S=1`、`P=8/S=128`、`P=16/S=64`、`P=16/S=128`，FD 开/关均覆盖，全部通过 |
| topk=8192 | 单 rank golden 校验通过 |
| 连续物理块 | `test_qliv2_contiguous_pa_block_table` 通过，覆盖 `KeyNd2NzForPA` 相邻块合并 |
| A2 既有 LD 用例 | `ld_invisible_segments` 4 组通过（A5 上 metadata 裁掉不可见段，单块场景不强制跨核 LD）；`ld_gate_non_dcp_vs_dcp` 通过（A5 非 DCP 长 S2 仍开 FD/LD） |
| 多 rank 合并 | 与同一全局序列 `P=1` 结果集合一致 |
| 非 DCP、`return_value=1` | `enable_dcp=false` 与 `P=1` 结果一致，values 正常输出 |
| `return_value=0` | 不属于 A5 DCP 验收范围，本轮未跑；测试文件中的非 DCP 回归保留供其他芯片使用 |
| 边界并列 | BF16 量化后同值或分差 ≤ 1 ULP 视为并列；测试判据已按此比较 |
| 验证方式 | 仓外最小扩展（`QLI_DCP_EXT`），未安装 vllm；vllm binding 本轮未编译 |
| Host UT | 本 clone 缺 `tests/ut/framework_normal`，未执行；需在完整 ops-transformer 仓补跑 |
| 构建注意 | 增量构建时 `libtransformer_aicpu_kernels.so` 的 custom command 不会随 `.o` 变更自动重链，需删除该 `.so` 后重建；干净构建不受影响 |

A2（910B）回归未在本机执行（无 910B 设备）；arch22 代码未改，合并前建议在 910B 补跑一次 A2 DCP 测试。

## 11. 设计 Checklist

- [x] 输入输出定义已确认
- [x] dtype/layout/shape 范围已确认
- [x] 数学语义已确认
- [x] 精度和性能验收标准已确认
- [x] 算法阶段已拆解
- [x] 每阶段输入输出已记录
- [x] host tiling 公式已设计
- [x] workspace 大小和 offset 已设计（无增量）
- [x] UB/L1/L0 buffer 已核算（无增量）
- [x] API 约束已检查
- [x] 同步点已标注（不新增）
