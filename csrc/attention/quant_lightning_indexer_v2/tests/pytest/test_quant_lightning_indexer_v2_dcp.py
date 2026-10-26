#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""QuantLightningIndexerV2 DCP (Decode Context Parallel, 解码上下文并行) 专项测试.

覆盖:
1. 单 rank 输出与 CPU golden 一致 (本地 index 集合 + BF16 score);
2. 多 rank 输出按 issue #7 的协议合并后, 与同一全局序列按 P=1 的结果一致;
3. 边界: C 落在 P*S 边界、末 rank 截断、L_r=0、Q_len>1、mask_mode=0/3。

用法:
    pytest -s test_quant_lightning_indexer_v2_dcp.py
    pytest -s test_quant_lightning_indexer_v2_dcp.py -k "case_p2_s1"
"""

import itertools
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import quant_lightning_indexer_v2_golden as golden_lib  # noqa: E402

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover - CPU-only dev box
    torch_npu = None

# 可选: 直接加载独立编译的最小扩展 (.so), 用于没有安装 vllm_ascend/cann_ops_transformer 的环境
_QLI_DCP_EXT = os.environ.get("QLI_DCP_EXT", "").strip()
if _QLI_DCP_EXT and torch_npu is not None:
    torch.ops.load_library(_QLI_DCP_EXT)

HEAD_DIM = 128
N1 = 64
N2 = 1
BLOCK_SIZE = 16


def _require_npu():
    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")
    if _qli_ops() is None:
        pytest.skip("neither vllm_ascend._C nor cann_ops_transformer is installed")


_QLI_BACKEND = None


def _qli_backend():
    """优先使用 vllm_ascend 的 _C_ascend 绑定, 回退到独立的 cann_ops_transformer 扩展."""
    global _QLI_BACKEND
    if _QLI_BACKEND is None:
        if hasattr(torch.ops, "_C_ascend") and hasattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2"):
            _QLI_BACKEND = ("vllm", torch.ops._C_ascend)
        elif hasattr(torch.ops, "cann_ops_transformer") and hasattr(
            torch.ops.cann_ops_transformer, "quant_lightning_indexer"
        ):
            _QLI_BACKEND = ("cann", torch.ops.cann_ops_transformer)
    return _QLI_BACKEND


def _qli_ops():
    backend = _qli_backend()
    return None if backend is None else backend[1]


def _meta_op_name():
    return "npu_quant_lightning_indexer_v2_metadata" if _qli_backend()[0] == "vllm" else "quant_lightning_indexer_metadata"


def _indexer_op_name():
    return "npu_quant_lightning_indexer_v2" if _qli_backend()[0] == "vllm" else "quant_lightning_indexer"


def dcp_local_indices(global_len, cp_world, cp_rank, cp_interleave):
    """全局 token 序号 -> 本 rank 本地逻辑序号 (本地按时间序)."""
    return [t for t in range(global_len) if (t // cp_interleave) % cp_world == cp_rank]


def dcp_global_index(local_idx, cp_world, cp_rank, cp_interleave):
    return (local_idx // cp_interleave * cp_world + cp_rank) * cp_interleave + local_idx % cp_interleave


def dcp_visible_len(global_end, q_len, row_idx, local_len, cp_world, cp_rank, cp_interleave):
    u = max(global_end - q_len + row_idx + 1, 0)
    if u <= 0:
        return 0
    period = cp_world * cp_interleave
    full_rounds = (u // period) * cp_interleave
    rem = u % period - cp_rank * cp_interleave
    tail = min(max(rem, 0), cp_interleave)
    return min(local_len, full_rounds + tail)


def _build_metadata(case, topk, mask_mode=3):
    ops = _qli_ops()
    return getattr(ops, _meta_op_name())(
        num_heads_q=N1,
        num_heads_k=N2,
        head_dim=HEAD_DIM,
        topk=topk,
        quant_mode=2,
        cu_seqlens_q=case["cu_seqlens_q"].npu(),
        seqused_k=case["seqused_k"].npu(),
        batch_size=case["batch_size"],
        max_seqlen_q=case["q_len"],
        max_seqlen_k=case["max_l"],
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=mask_mode,
        cmp_ratio=1,
    )


def _make_case(cp_world, cp_rank, cp_interleave, batch_size, q_len, causal_lens, topk, seed=0):
    """构造同一全局序列在指定 rank 上的本地输入 (PA_BBND 物理池 + 本地逻辑 K)."""
    rng = np.random.default_rng(seed)
    q_t = batch_size * q_len
    causal_lens = list(causal_lens)
    max_c = max(causal_lens)

    # 全局 K / 全局 qk 分数基准 (同一 seed 保证各 rank 看到相同的全局 token)
    global_k = rng.integers(-128, 128, size=(max_c, HEAD_DIM), dtype=np.int16).astype(np.int8)
    global_k_scale = rng.uniform(0.5, 1.5, size=(max_c,)).astype(np.float16)
    query = rng.integers(-128, 128, size=(q_t, N1, HEAD_DIM), dtype=np.int16).astype(np.int8)
    weights = rng.uniform(0.1, 1.0, size=(q_t, N1)).astype(np.float16)
    q_scale = rng.uniform(0.5, 1.5, size=(q_t, N1)).astype(np.float16)

    local_lens = []
    local_ks = []
    local_k_scales = []
    for b in range(batch_size):
        idx = dcp_local_indices(causal_lens[b], cp_world, cp_rank, cp_interleave)
        local_lens.append(len(idx))
        if idx:
            local_ks.append(global_k[idx])
            local_k_scales.append(global_k_scale[idx])
        else:
            local_ks.append(np.zeros((0, HEAD_DIM), dtype=np.int8))
            local_k_scales.append(np.zeros((0,), dtype=np.float16))
    max_l = max(local_lens)
    blocks_per_batch = max(1, (max_l + BLOCK_SIZE - 1) // BLOCK_SIZE)
    block_num = batch_size * blocks_per_batch

    # 物理池: 本地逻辑块随机置换到物理块
    key_pool = np.zeros((block_num, BLOCK_SIZE, N2, HEAD_DIM), dtype=np.int8)
    k_scale_pool = np.ones((block_num, BLOCK_SIZE, N2), dtype=np.float16)
    block_table = np.zeros((batch_size, blocks_per_batch), dtype=np.int32)
    local_k_padded = np.zeros((batch_size, N2, max_l, HEAD_DIM), dtype=np.int8)
    k_scale_logical = np.ones((batch_size, N2, max_l), dtype=np.float16)
    # 物理池: 每个 batch 分到互不重叠的物理块, 块内按本地逻辑块随机置换
    perm_all = rng.permutation(block_num)
    for b in range(batch_size):
        perm = perm_all[b * blocks_per_batch : (b + 1) * blocks_per_batch]
        block_table[b] = perm
        local_k_padded[b, 0, : local_lens[b]] = local_ks[b]
        k_scale_logical[b, 0, : local_lens[b]] = local_k_scales[b]
        for blk in range(blocks_per_batch):
            start = blk * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, local_lens[b])
            if end > start:
                key_pool[perm[blk], : end - start, 0] = local_ks[b][start:end]
                k_scale_pool[perm[blk], : end - start, 0] = k_scale_logical[b, 0, start:end]

    cu_seqlens_q = np.arange(0, q_t + 1, q_len, dtype=np.int32)
    return {
        "batch_size": batch_size,
        "q_len": q_len,
        "q_t": q_t,
        "topk": topk,
        "query": torch.from_numpy(query),
        "key_pool": torch.from_numpy(key_pool.reshape(block_num, BLOCK_SIZE, N2, HEAD_DIM)),
        "weights": torch.from_numpy(weights),
        "q_scale": torch.from_numpy(q_scale),
        "k_scale_pool": torch.from_numpy(k_scale_pool),
        "block_table": torch.from_numpy(block_table),
        "cu_seqlens_q": torch.from_numpy(cu_seqlens_q),
        "seqused_k": torch.tensor(local_lens, dtype=torch.int32),
        "causal_lens": torch.tensor(causal_lens, dtype=torch.int32),
        "local_k": torch.from_numpy(local_k_padded),
        "k_scale_logical": torch.from_numpy(k_scale_logical),
        "local_lens": local_lens,
        "max_l": max_l,
        "max_c": max_c,
    }


def _run_op(case, cp_world=1, cp_rank=0, cp_interleave=1, mask_mode=3, enable_dcp=True, return_value=1):
    ops = _qli_ops()
    metadata = _build_metadata(case, case["topk"], mask_mode).npu()
    return getattr(ops, _indexer_op_name())(
        case["query"].npu(),
        case["key_pool"].npu(),
        case["weights"].npu(),
        case["q_scale"].npu(),
        case["k_scale_pool"].npu(),
        case["topk"],
        2,
        cu_seqlens_q=case["cu_seqlens_q"].npu(),
        seqused_k=case["seqused_k"].npu(),
        block_table=case["block_table"].npu(),
        metadata=metadata,
        max_seqlen_q=case["q_len"],
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=mask_mode,
        cmp_ratio=1,
        return_value=return_value,
        causal_seqlens_kv_global=case["causal_lens"].npu() if enable_dcp else None,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        cp_kv_cache_interleave_size=cp_interleave,
    )


def _run_golden(case, cp_world=1, cp_rank=0, cp_interleave=1, sparse_mode=3, dcp_enable=True):
    g = golden_lib.GeneralizedQLIV2(
        batch_size=case["batch_size"],
        q_seq=case["q_len"],
        k_seq=case["max_l"],
        q_t_size=case["q_t"],
        k_t_size=0,
        q_head_num=N1,
        k_head_num=N2,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        block_num=BLOCK_SIZE,
        qk_dtype=torch.int8,
        weight_dtype=torch.float16,
        dequant_dtype=torch.float16,
        actual_seq_dtype=torch.int32,
        cu_seqlens_q=case["cu_seqlens_q"].clone(),
        cu_seqlens_k=torch.zeros(case["batch_size"] + 1, dtype=torch.int32),
        act_seq_q=None,
        act_seq_k=case["seqused_k"].clone(),
        cmp_residual_k=torch.zeros(case["batch_size"], dtype=torch.int32),
        max_seqlen_q=case["q_len"],
        quant_mode=2,
        layout_query="TND",
        layout_key="PA_BBND",
        sparse_count=case["topk"],
        sparse_mode=sparse_mode,
        query_datarange=[-128, 127],
        key_datarange=[-128, 127],
        weights_datarange=[0.1, 1.0],
        q_scale_datarange=[0.5, 1.5],
        k_scale_datarange=[0.5, 1.5],
        cmp_ratio=1,
        return_value=1,
        dcp_enable=dcp_enable,
        cp_world=cp_world,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
        causal_seqlens_kv_global=case["causal_lens"].clone(),
    )
    y, _, values = g.forward(
        case["query"],
        case["local_k"],
        case["weights"],
        case["q_scale"],
        case["k_scale_logical"],
        case["cu_seqlens_q"],
        torch.zeros(case["batch_size"] + 1, dtype=torch.int32),
        None,
        case["seqused_k"],
        None,
        None,
    )
    return y.numpy(), values.numpy()


def _row_scores(case, cp_world, cp_rank, cp_interleave):
    """按算子数学定义逐行重建 golden 分数 (用于并列/边界仲裁), 返回 [T1, max_l]."""
    q = case["query"].to(torch.int32).numpy()
    k = case["local_k"][:, 0].to(torch.int32).numpy()
    w = case["weights"].to(torch.float32).numpy()
    qs = case["q_scale"].to(torch.float32).numpy()
    ks = case["k_scale_logical"][:, 0].to(torch.float32).numpy()
    cu = case["cu_seqlens_q"].numpy()
    scores = np.full((case["q_t"], case["max_l"]), -np.inf, dtype=np.float32)
    for b in range(case["batch_size"]):
        q_len = case["q_len"]
        local_len = case["local_lens"][b]
        global_end = int(case["causal_lens"][b])
        for i in range(q_len):
            row = int(cu[b]) + i
            qk = q[row] @ k[b].T  # [N1, L]
            qk = np.maximum(qk / 1024.0, 0.0)
            wrow = w[row] * qs[row]
            score = (wrow[:, None] * qk).sum(axis=0) * ks[b]
            vis = dcp_visible_len(global_end, q_len, i, local_len, cp_world, cp_rank, cp_interleave)
            scores[row, :vis] = score[:vis]
    return scores


def _compare_rank(case, cp_world, cp_rank, cp_interleave):
    out, values = _run_op(case, cp_world, cp_rank, cp_interleave)
    out = out.cpu().numpy()[:, 0, :]
    values = values.cpu().float().numpy()[:, 0, :]
    gold, gold_values = _run_golden(case, cp_world, cp_rank, cp_interleave)
    gold = gold[:, 0, :]
    gold_values = gold_values[:, 0, :].astype(np.float32)
    scores = _row_scores(case, cp_world, cp_rank, cp_interleave)

    for row in range(out.shape[0]):
        pred = out[row][out[row] >= 0]
        ref = gold[row][gold[row] >= 0]
        assert len(pred) == len(ref), f"row {row}: valid count {len(pred)} != {len(ref)}"
        if len(ref) == 0:
            continue
        pred_set, ref_set = set(pred.tolist()), set(ref.tolist())
        if pred_set != ref_set:
            # 并列边界: 只允许"被替换成员与参考边界同分"
            only_pred = sorted(pred_set - ref_set)
            only_ref = sorted(ref_set - pred_set)
            boundary = min(scores[row, list(ref_set)]) if only_ref else None
            for i in only_pred:
                for j in only_ref:
                    assert abs(scores[row, i] - scores[row, j]) <= 1e-3 * max(1.0, abs(scores[row, j])), (
                        f"row {row}: index set mismatch pred-only={i} ref-only={j} "
                        f"score_pred={scores[row, i]} score_ref={scores[row, j]} boundary={boundary}"
                    )
        # score 校验: 取算子返回的 index 在重建分数里的值, 与返回 score 对齐
        valid = out[row] >= 0
        if valid.any():
            expect = scores[row][out[row][valid]]
            got = values[row][valid]
            np.testing.assert_allclose(got, expect, rtol=2e-2, atol=2e-3)
    return out, values


def _merge_global_per_row(rank_outputs, cp_world, cp_interleave, topk):
    """按行合并各 rank 的 (score, local_idx), 返回每行的全局 TopK 集合."""
    num_rows = rank_outputs[0][0].shape[0]
    merged_rows = []
    for row in range(num_rows):
        cand = []
        for rank, (idx, val) in enumerate(rank_outputs):
            for local_idx, score in zip(idx[row], val[row]):
                if local_idx < 0:
                    continue
                cand.append((float(score), int(dcp_global_index(int(local_idx), cp_world, rank, cp_interleave))))
        cand.sort(key=lambda x: (-x[0], x[1]))  # 稳定排序: score 降序, 并列取更小全局序号
        merged_rows.append({g for _, g in cand[:topk]})
    return merged_rows


CASES = [
    # (P, S, B, Q_len, global lens per batch, topk)
    (2, 1, 1, 1, [257], 512),
    (2, 128, 1, 4, [257], 512),
    (4, 16, 1, 2, [255], 512),
    (2, 128, 2, 1, [129, 257], 512),
    (4, 128, 1, 1, [127], 512),   # 末 rank L=0
    (2, 64, 1, 4, [513], 2048),
]


@pytest.mark.ci
@pytest.mark.parametrize("cp_world,cp_interleave,batch_size,q_len,causal_lens,topk", CASES)
def test_qliv2_dcp(cp_world, cp_interleave, batch_size, q_len, causal_lens, topk):
    _require_npu()
    rank_outputs = []
    for rank in range(cp_world):
        case = _make_case(
            cp_world, rank, cp_interleave, batch_size, q_len, causal_lens, topk, seed=0
        )
        idx, val = _compare_rank(case, cp_world, rank, cp_interleave)
        rank_outputs.append((idx, val))

    # 多 rank 合并 == 同一全局序列按 P=1 的结果 (全局 topk 精确等价)
    merged_rows = _merge_global_per_row(rank_outputs, cp_world, cp_interleave, topk)
    ref_case = _make_case(1, 0, 1, batch_size, q_len, causal_lens, topk, seed=0)
    ref_out, _ = _run_op(ref_case, 1, 0, 1)
    ref_out = ref_out.cpu().numpy()[:, 0, :]
    for row, merged in enumerate(merged_rows):
        ref = set(ref_out[row][ref_out[row] >= 0].tolist())
        assert merged == ref, f"row {row}: merged global topk != P=1 topk, diffs={sorted(merged ^ ref)}"


@pytest.mark.ci
def test_qliv2_dcp_mask_mode0():
    """mask_mode=0 (无 causal) 下 DCP 只影响框架合并, 算子行为与本地全可见一致."""
    _require_npu()
    case = _make_case(2, 0, 128, 1, 1, [257], 512, seed=7)
    out, _ = _run_op(case, 2, 0, 128, mask_mode=0)
    out = out.cpu().numpy()[:, 0, :]
    # 全可见: 行内有效 index 数 = min(topk, 本地 L)
    valid = (out >= 0).sum(axis=1)
    assert valid[0] == min(512, case["local_lens"][0])


NON_DCP_CASES = [
    (1, 1, [257]),
    (1, 4, [513]),
    (2, 2, [129, 257]),
]


@pytest.mark.ci
@pytest.mark.parametrize("batch_size,q_len,causal_lens", NON_DCP_CASES)
def test_qliv2_non_dcp_regression(batch_size, q_len, causal_lens):
    """非 DCP 路径回归: P=1 时 index 应与 DCP P=1 完全一致, return_value=0 时 values 为空."""
    _require_npu()
    case = _make_case(1, 0, 1, batch_size, q_len, causal_lens, 512, seed=0)
    ref_idx, _ = _run_op(case, 1, 0, 1)
    idx, val = _run_op(case, 1, 0, 1, enable_dcp=False, return_value=0)
    np.testing.assert_array_equal(idx.cpu().numpy(), ref_idx.cpu().numpy())
    assert val.numel() == 0, "return_value=0 时 sparse_values 必须为空 tensor"


@pytest.mark.ci
def test_qliv2_non_dcp_return_value():
    """arch22 新打开的 return_value=1: 与 CPU golden (非 DCP) 对齐 index 与 BF16 score."""
    _require_npu()
    case = _make_case(1, 0, 1, 1, 4, [513], 512, seed=0)
    idx, val = _run_op(case, 1, 0, 1, enable_dcp=False, return_value=1)
    idx = idx.cpu().numpy()[:, 0, :]
    val = val.cpu().float().numpy()[:, 0, :]
    gold, gold_val = _run_golden(case, 1, 0, 1, dcp_enable=False)
    gold = gold[:, 0, :]
    gold_val = gold_val[:, 0, :].astype(np.float32)
    for row in range(idx.shape[0]):
        pred = idx[row][idx[row] >= 0]
        ref = gold[row][gold[row] >= 0]
        assert set(pred.tolist()) == set(ref.tolist()), f"row {row}: index set mismatch"
        if len(ref) == 0:
            continue
        # 同序比较 score (并列顺序由排序实现决定, 先按 index 对齐再比较)
        ref_map = {int(i): float(v) for i, v in zip(ref, gold_val[row][gold[row] >= 0])}
        for i, v in zip(idx[row][idx[row] >= 0], val[row][idx[row] >= 0]):
            np.testing.assert_allclose(float(v), ref_map[int(i)], rtol=2e-2, atol=2e-3)

