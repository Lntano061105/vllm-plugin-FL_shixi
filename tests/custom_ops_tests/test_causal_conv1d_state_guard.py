# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Meta-check for the state-writeback guards in test_causal_conv1d_accuracy.py.
# Proves the "untouched slot / unwritten tail" assertions are not blind:
# a clean OP run passes them, and deliberately corrupting those positions
# after the kernel runs must make them fail.
#
# Run:  VLLM_PLUGINS=ascend <python3.11> test_causal_conv1d_state_guard.py

import torch
import torch_npu

import vllm_fl._C_ascend  # noqa: F401

DEVICE = "npu:0"
OP = torch.ops._C_ascend.npu_causal_conv1d_custom
DTYPE = torch.bfloat16


def _untouched_slots_ok(cs, cc, touched, ncl):
    """Same guard as test_cache_indices_padding_skip: unselected lines bit-identical."""
    cs_cpu = cs.float().cpu()
    for c in range(ncl):
        if c in touched:
            continue
        if (cs_cpu[c] - cc[c]).abs().max().item() != 0.0:
            return False
    return True


def _tail_ok(cs, cc, keep):
    """Same guard as test_mtp_spec_shift: rows past 2+L bit-identical."""
    return (cs.float().cpu()[:, keep:] - cc[:, keep:]).abs().max().item() == 0.0


def cache_guard(tamper):
    torch.manual_seed(0)
    D, W = 16, 4
    SL = W - 1
    B, NCL = 6, 6
    PAD = -1
    ci = [3, 0, PAD, 2, 5, 99]  # selected: 3, 0, 2, 5 -> unselected: 1, 4
    x    = torch.randn(B, D, dtype=DTYPE, device=DEVICE)
    wt   = torch.randn(W, D, dtype=DTYPE, device=DEVICE) * 0.3
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.1
    cs   = torch.randn(NCL, SL, D, dtype=DTYPE, device=DEVICE) * 0.2
    out  = torch.full((B, D), float('nan'), dtype=DTYPE, device=DEVICE)
    cit  = torch.tensor(ci, dtype=torch.int32, device=DEVICE)

    cc = cs.float().cpu()
    OP(out, x, wt, cs, bias, None, cit, None, None, 0, PAD, 1)

    if tamper:
        cs[1] += 1.0  # simulate an erroneous write to an unselected cache line

    touched = {c for c in ci if c != PAD and 0 <= c < NCL}
    return _untouched_slots_ok(cs, cc, touched, NCL)


def mtp_guard(tamper):
    torch.manual_seed(0)
    D, W = 16, 4
    SL = 6  # stateLen >= (W-1) + (L-1); rows 0..4 written, row 5 untouched
    B, L = 3, 3
    accepted = [1, 2, 3]
    x    = torch.randn(B, L, D, dtype=DTYPE, device=DEVICE)
    wt   = torch.randn(W, D, dtype=DTYPE, device=DEVICE) * 0.3
    bias = torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.1
    cs   = torch.randn(B, SL, D, dtype=DTYPE, device=DEVICE) * 0.2
    out  = torch.empty(B, L, D, dtype=DTYPE, device=DEVICE)
    nat  = torch.tensor(accepted, dtype=torch.int32, device=DEVICE)

    cc = cs.float().cpu()
    OP(out, x, wt, cs, bias, None, None, None, nat, 0, -1, 1)

    if tamper:
        cs[:, 2 + L] += 1.0  # simulate an erroneous write to the unwritten tail

    return _tail_ok(cs, cc, 2 + L)


if __name__ == "__main__":
    assert cache_guard(tamper=False) is True, "clean run must pass the cache-line guard"
    assert cache_guard(tamper=True) is False, "tampered slot must fail the cache-line guard"
    assert mtp_guard(tamper=False) is True, "clean run must pass the MTP tail guard"
    assert mtp_guard(tamper=True) is False, "tampered tail must fail the MTP tail guard"
    print("state guard meta-check passed: guards fire on tampered positions")
