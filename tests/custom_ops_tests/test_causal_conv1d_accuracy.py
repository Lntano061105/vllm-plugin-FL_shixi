# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Causal Conv1D accuracy unit tests (fixed seed + CPU reference).
# Covers the internship acceptance points:
#   1. Prefill state writeback
#   2. Decode state rolling
#   3. Varlen / tile crossing
#   4. cache_indices / pad_slot_id skipping
#   5. MTP num_accepted_tokens spec shift
# Plus a numerical comparison against the framework Triton baseline
# (vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d.causal_conv1d_ref).
#
# Run:  VLLM_PLUGINS=ascend <python3.11> test_causal_conv1d_accuracy.py

import itertools

import torch
import torch_npu

import vllm_fl._C_ascend  # noqa: F401

DEVICE = "npu:0"
OP = torch.ops._C_ascend.npu_causal_conv1d_custom
DTYPE = torch.bfloat16
TOL = 0.05


def test_prefill_state_writeback():
    torch.manual_seed(0)
    D, W, T, SL = 16, 3, 8, 2
    x    = torch.randn(T, D, dtype=DTYPE, device=DEVICE)
    wt   = torch.randn(W, D, dtype=DTYPE, device=DEVICE) * 0.3
    cs   = torch.zeros(1, SL, D, dtype=DTYPE, device=DEVICE)
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.1
    qsl  = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)
    out  = torch.empty(T, D, dtype=DTYPE, device=DEVICE)

    y = OP(out, x, wt, cs, bias, qsl, None, None, None, 0, -1, 0)

    xc, wc, bc = x.float().cpu(), wt.float().cpu(), bias.float().cpu()
    hist = torch.cat([torch.zeros(SL, D), xc])
    ref = torch.zeros(T, D)
    for t in range(T):
        for k in range(W):
            ref[t] += wc[k] * hist[t + k]
        ref[t] += bc
    assert (y.float().cpu() - ref).abs().max().item() < TOL

    # state writeback: expect last W-1 rows of x
    cs_after = cs.float().cpu()[0]
    assert (cs_after - xc[-SL:]).abs().max().item() < TOL


def test_decode_state_rolling():
    torch.manual_seed(0)
    D, W, B, SL = 16, 3, 2, 2
    x    = torch.randn(B, D, dtype=DTYPE, device=DEVICE)
    wt   = torch.randn(W, D, dtype=DTYPE, device=DEVICE) * 0.3
    cs   = torch.randn(B, SL, D, dtype=DTYPE, device=DEVICE) * 0.2
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.1
    out  = torch.empty(B, D, dtype=DTYPE, device=DEVICE)

    cc = cs.float().cpu()  # copy BEFORE kernel in-place writeback
    y = OP(out, x, wt, cs, bias, None, None, None, None, 0, -1, 1)

    xc, wc, bc = x.float().cpu(), wt.float().cpu(), bias.float().cpu()
    seq = torch.cat([cc, xc.unsqueeze(1)], dim=1)  # (B, W, D)
    ref = torch.einsum('kd,bkd->bd', wc, seq) + bc
    assert (y.float().cpu() - ref).abs().max().item() < TOL

    # state rolling: expect [s1, x[b]]
    assert (cs.float().cpu() - seq[:, 1:]).abs().max().item() < TOL


def test_prefill_matches_framework_baseline():
    from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import \
        causal_conv1d_ref as ref

    torch.manual_seed(0)
    D, W, T = 16, 3, 8
    x   = torch.randn(T, D, dtype=DTYPE, device=DEVICE)
    wt  = torch.randn(W, D, dtype=DTYPE, device=DEVICE) * 0.3
    cs  = torch.zeros(1, 2, D, dtype=DTYPE, device=DEVICE)
    b   = torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.1
    qsl = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)
    o   = torch.empty(T, D, dtype=DTYPE, device=DEVICE)

    y = OP(o, x, wt, cs, b, qsl, None, None, None, 0, -1, 0)

    xb = x.float().cpu().t().unsqueeze(0)  # (1, D, T)
    wb = wt.float().cpu().t()              # (D, W)
    bb = b.float().cpu()
    ref_out, fs = ref(xb, wb, bb, activation=None, return_final_states=True)
    assert (y.float().cpu().t() - ref_out.squeeze(0)).abs().max().item() < TOL
    assert (cs.float().cpu().squeeze(0).t() - fs.squeeze(0)).abs().max().item() < TOL


def test_varlen_tile_crossing():
    torch.manual_seed(0)
    D, W = 16, 3
    SL = W - 1
    lens = [3, 8, 5, 12, 6]  # 5 unequal sequences, host tile boundaries cross seqs
    B, T = len(lens), sum(lens)
    x    = torch.randn(T, D, dtype=DTYPE, device=DEVICE)
    wt   = torch.randn(W, D, dtype=DTYPE, device=DEVICE) * 0.3
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.1
    cs   = torch.randn(B, SL, D, dtype=DTYPE, device=DEVICE) * 0.2
    qsl  = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32,
                        device=DEVICE)
    out  = torch.empty(T, D, dtype=DTYPE, device=DEVICE)
    ism  = torch.ones(B, dtype=torch.int32, device=DEVICE)  # enable initial state on FN

    cc = cs.float().cpu()
    y = OP(out, x, wt, cs, bias, qsl, None, ism, None, 0, -1, 0)

    xc, wc, bc = x.float().cpu(), wt.float().cpu(), bias.float().cpu()
    ref = torch.zeros(T, D)
    ref_state = torch.zeros_like(cc)
    for s in range(B):
        st = sum(lens[:s])
        Tn = lens[s]
        seq = xc[st:st + Tn]
        hist = torch.cat([cc[s], seq])
        for t in range(Tn):
            for k in range(W):
                ref[st + t] += wc[k] * hist[t + k]
            ref[st + t] += bc
        ref_state[s] = seq[-SL:]
    assert (y.float().cpu() - ref).abs().max().item() < TOL
    assert (cs.float().cpu() - ref_state).abs().max().item() < TOL


def test_cache_indices_padding_skip():
    torch.manual_seed(0)
    D, W = 16, 4
    SL = W - 1
    B, NCL = 6, 6
    PAD = -1
    ci = [3, 0, PAD, 2, 5, 99]  # seq2=padding(-1), seq5 out-of-range(99) skipped
    x    = torch.randn(B, D, dtype=DTYPE, device=DEVICE)
    wt   = torch.randn(W, D, dtype=DTYPE, device=DEVICE) * 0.3
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.1
    cs   = torch.randn(NCL, SL, D, dtype=DTYPE, device=DEVICE) * 0.2
    out  = torch.full((B, D), float('nan'), dtype=DTYPE, device=DEVICE)
    cit  = torch.tensor(ci, dtype=torch.int32, device=DEVICE)

    cc = cs.float().cpu()
    y = OP(out, x, wt, cs, bias, None, cit, None, None, 0, PAD, 1)

    xc, wc, bc = x.float().cpu(), wt.float().cpu(), bias.float().cpu()
    ref = torch.full((B, D), float('nan'))
    ref_state = cs.float().cpu().clone()
    for s in range(B):
        c = ci[s]
        if c == PAD or not (0 <= c < NCL):
            continue
        hist = torch.cat([cc[c], xc[s].unsqueeze(0)])  # (W, D)
        yv = torch.zeros(D)
        for k in range(W):
            yv += wc[k] * hist[k]
        yv += bc
        ref[s] = yv
        ref_state[c] = hist[1:]
    mask = ~torch.isnan(ref)
    assert ((y.float().cpu() - ref).abs()[mask] < TOL).all()
    # skipped slots keep NaN
    assert torch.isnan(y.float().cpu()[2]).all()
    assert torch.isnan(y.float().cpu()[5]).all()
    assert (cs.float().cpu() - ref_state).abs().max().item() < TOL


def test_mtp_spec_shift():
    torch.manual_seed(0)
    D, W = 16, 4
    SL = 6  # stateLen >= (W-1) + (L-1)
    B, L = 3, 3
    accepted = [1, 2, 3]
    x    = torch.randn(B, L, D, dtype=DTYPE, device=DEVICE)
    wt   = torch.randn(W, D, dtype=DTYPE, device=DEVICE) * 0.3
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.1
    cs   = torch.randn(B, SL, D, dtype=DTYPE, device=DEVICE) * 0.2
    out  = torch.empty(B, L, D, dtype=DTYPE, device=DEVICE)
    nat  = torch.tensor(accepted, dtype=torch.int32, device=DEVICE)

    cc = cs.float().cpu()
    y = OP(out, x, wt, cs, bias, None, None, None, nat, 0, -1, 1)

    xc, wc, bc = x.float().cpu(), wt.float().cpu(), bias.float().cpu()
    ref = torch.zeros(B, L, D)
    ref_state = cs.float().cpu().clone()
    for s in range(B):
        # stateTokenOffset, same clamp as kernel
        off = max(0, min(accepted[s] - 1, SL - (W - 1)))
        hist = torch.cat([cc[s, off:off + W - 1], xc[s]])  # (W-1+L, D)
        for t in range(L):
            for k in range(W):
                ref[s, t] += wc[k] * hist[t + k]
            ref[s, t] += bc
        # WriteBackStateSpec shifted writeback
        new = ref_state[s].clone()
        new[0] = cc[s, off + 1]
        new[1] = cc[s, off + 2]
        new[2:2 + L] = xc[s]
        ref_state[s] = new
    assert (y.float().cpu() - ref).abs().max().item() < TOL
    assert (cs.float().cpu() - ref_state).abs().max().item() < TOL


if __name__ == "__main__":
    test_prefill_state_writeback()
    test_decode_state_rolling()
    test_prefill_matches_framework_baseline()
    test_varlen_tile_crossing()
    test_cache_indices_padding_skip()
    test_mtp_spec_shift()
    print("all causal_conv1d accuracy tests passed")
