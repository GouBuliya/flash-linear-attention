# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.triadic_kda.chunk_fwd import (
    CHUNK_SIZE,
    MAX_ANCHOR_SPAN,
    chunk_position,
    fwd_h,
    fwd_intra,
    state_dot,
)
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_offsets


@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_bwd_kernel_h(
    q2,
    k2,
    beta,
    gk,
    g2,
    Ai,
    Aqk,
    pq,
    pkp,
    pr,
    do,
    dx,
    dh,
    dht,
    dh0,
    cu,
    offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    VARLEN: tl.constexpr,
    INITIAL: tl.constexpr,
    FINAL: tl.constexpr,
):
    iv, nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    n, h = nh // H, nh % H
    if VARLEN:
        start, end = tl.load(cu + n).to(tl.int64), tl.load(cu + n + 1).to(tl.int64)
        nc = tl.cdiv(end - start, BT)
        first = tl.load(offsets + n).to(tl.int64)
    else:
        start, end, nc = n * T, (n + 1) * T, tl.cdiv(T, BT)
        first = n * nc
    t = tl.arange(0, BT).to(tl.int64)
    d = tl.arange(0, BK).to(tl.int64)
    vv = iv * BV + tl.arange(0, BV).to(tl.int64)
    ph = d[None, :] * V + vv[:, None]
    mh = (vv < V)[:, None] & (d < K)[None, :]

    hs = ()
    for e_init in tl.static_range(E):
        if FINAL:
            value = tl.load(dht + (nh * E + e_init) * K * V + ph, mh, other=0.0)
        else:
            value = tl.zeros((BV, BK), tl.float32)
        hs += (value,)

    for c in tl.range(nc - 1, -1, -1):
        tt = start + c * BT + t
        mt = tt < end
        pk = (tt[:, None] * H + h) * K + d[None, :]
        mk = mt[:, None] & (d < K)[None, :]
        pv = (tt[None, :] * H + h) * V + vv[:, None]
        mv = (vv < V)[:, None] & mt[None, :]
        pe = (tt * H + h) * E
        last = tl.minimum(start + (c + 1) * BT, end) - 1

        kg_t = tl.trans(tl.load(pkp + pk, mk, other=0.0))
        dout = tl.load(do + pv, mv, other=0.0)
        pa = (tt[:, None] * H + h) * BT + t[None, :]

        # read pass: subtract each slice's contribution and cache its state in bf16
        aqk = tl.load(Aqk + pa, mt[:, None], other=0.0)
        du = state_dot(dout, aqk) * scale
        for e_read in tl.range(E):
            hp = hs[0].to(tl.bfloat16)
            tl.store(dh + (((first + c) * H + h) * E + e_read) * K * V + ph, hp, mh)
            ge = tl.load(g2 + pe + e_read, mt, other=0.0)
            gel = tl.load(g2 + (last * H + h) * E + e_read)
            ke = tl.load(k2 + pe + e_read, mt, other=0.0) * tl.exp(tl.minimum(gel - ge, 0.0))
            du += tl.dot(hp, kg_t, out_dtype=tl.float32) * ke[None, :]
            if E == 1:
                hs = (hs[0],)
            elif E == 2:
                hs = (hs[1], hs[0])
            elif E == 4:
                hs = (hs[1], hs[2], hs[3], hs[0])
            elif E == 8:
                hs = (hs[1], hs[2], hs[3], hs[4], hs[5], hs[6], hs[7], hs[0])
            else:
                hs = (
                    hs[1], hs[2], hs[3], hs[4], hs[5], hs[6], hs[7], hs[8],
                    hs[9], hs[10], hs[11], hs[12], hs[13], hs[14], hs[15], hs[0],
                )

        # intra-chunk inverse, then the adjoint walk uses the same row block
        ai = tl.load(Ai + pa, mt[:, None], other=0.0)
        df = state_dot(du, ai)
        tl.store(dx + pv, df, mv)

        # write pass: decay the cached state and add the two outer products
        rg = tl.load(pr + pk, mk, other=0.0)
        qg = tl.load(pq + pk, mk, other=0.0)
        gl = tl.load(gk + (last * H + h) * K + d, d < K, other=0.0)
        b_beta = tl.load(beta + tt * H + h, mt, other=0.0).to(tl.float32)

        for e_write in tl.range(E):
            ge = tl.load(g2 + pe + e_write, mt, other=0.0)
            gel = tl.load(g2 + (last * H + h) * E + e_write)
            exp_ge = tl.exp(ge)
            qe = tl.load(q2 + pe + e_write, mt, other=0.0) * exp_ge
            re = tl.load(k2 + pe + e_write, mt, other=0.0) * b_beta * exp_ge
            value = hs[0] * tl.exp(gl + gel)[None, :]
            value += tl.dot((dout.to(tl.float32) * (qe * scale)[None, :]).to(tl.bfloat16), qg, out_dtype=tl.float32)
            value -= tl.dot((df * re[None, :]).to(tl.bfloat16), rg, out_dtype=tl.float32)
            if E == 1:
                hs = (value,)
            elif E == 2:
                hs = (hs[1], value)
            elif E == 4:
                hs = (hs[1], hs[2], hs[3], value)
            elif E == 8:
                hs = (hs[1], hs[2], hs[3], hs[4], hs[5], hs[6], hs[7], value)
            else:
                hs = (
                    hs[1], hs[2], hs[3], hs[4], hs[5], hs[6], hs[7], hs[8],
                    hs[9], hs[10], hs[11], hs[12], hs[13], hs[14], hs[15], value,
                )

    if INITIAL:
        for e_final in tl.static_range(E):
            tl.store(dh0 + (nh * E + e_final) * K * V + ph, hs[e_final], mh)


@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_bwd_kernel_inter(
    q,
    k,
    q2,
    k2,
    beta,
    gk,
    g2,
    states,
    dh,
    do,
    dx,
    u,
    dq,
    dk,
    dr,
    dgk,
    dq2,
    dk2,
    dr2,
    dg2,
    cu,
    indices,
    scale,
    T,
    TOTAL_T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    VARLEN: tl.constexpr,
):
    ik, c, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    start, end, gc = chunk_position(c, bh // H, T, cu, indices, BT, VARLEN)

    h = bh % H
    t = tl.arange(0, BT).to(tl.int64)
    d = ik * BK + tl.arange(0, BK).to(tl.int64)
    tt = start + t
    mt, md = tt < end, d < K
    pk = (tt[:, None] * H + h) * K + d[None, :]
    mask = mt[:, None] & md[None, :]
    pe = (tt * H + h) * E
    last = tl.minimum(start + BT, end) - 1

    # fuse q and r contractions against the shared bf16 forward state cache
    g = tl.load(gk + pk, mask, other=0.0)
    decay_qr = tl.exp(g)
    acc_q = tl.zeros((BT, BK), tl.float32)
    acc_r = tl.zeros((BT, BK), tl.float32)
    b_beta = tl.load(beta + tt * H + h, mt, other=0.0).to(tl.float32)

    for e in tl.range(E):
        ge = tl.load(g2 + pe + e, mt, other=0.0)
        exp_ge = tl.exp(ge)
        contraction_q = tl.zeros((BT, BK), tl.float32)
        contraction_r = tl.zeros((BT, BK), tl.float32)
        for iv in range(tl.cdiv(V, BV)):
            vv = iv * BV + tl.arange(0, BV).to(tl.int64)
            pv = (tt[:, None] * H + h) * V + vv[None, :]
            mv = mt[:, None] & (vv < V)[None, :]
            ph = ((gc * H + h) * E + e) * K * V + d[:, None] * V + vv[None, :]
            mh = md[:, None] & (vv < V)[None, :]

            state = tl.load(states + ph, mh, other=0.0)
            state_t = tl.trans(state)
            bv_do = tl.load(do + pv, mv, other=0.0).to(tl.bfloat16)
            bv_dx = tl.load(dx + pv, mv, other=0.0).to(tl.bfloat16)
            contraction_q += tl.dot(bv_do, state_t, out_dtype=tl.float32)
            contraction_r += tl.dot(bv_dx, state_t, out_dtype=tl.float32)

        contraction_q = contraction_q * (exp_ge * scale)[:, None]
        contraction_r = -contraction_r * exp_ge[:, None]
        qe = tl.load(q2 + pe + e, mt, other=0.0)
        re = tl.load(k2 + pe + e, mt, other=0.0) * b_beta
        acc_q += contraction_q * decay_qr * qe[:, None]
        acc_r += contraction_r * decay_qr * re[:, None]
        qv = tl.load(q + pk, mask, other=0.0).to(tl.float32)
        second_q = tl.sum(qv * decay_qr * contraction_q, 1)
        kv_s1 = tl.load(k + pk, mask, other=0.0).to(tl.float32)
        second_r = tl.sum(kv_s1 * decay_qr * contraction_r, 1)
        pp = ((ik * TOTAL_T + tt) * H + h) * E + e
        tl.store(dq2 + pp, second_q, mt)
        tl.store(dr2 + pp, second_r, mt)

    tl.store(dq + pk, acc_q, mask)
    tl.store(dr + pk, acc_r, mask)
    tl.debug_barrier()

    # contract k against dh and accumulate the state-decay gradient
    g = tl.load(gk + pk, mask, other=0.0)
    gl = tl.load(gk + (last * H + h) * K + d, md, other=0.0)
    decay_k = tl.exp(tl.minimum(gl[None, :] - g, 0.0))
    kv = tl.load(k + pk, mask, other=0.0).to(tl.float32)
    acc_k = tl.zeros((BT, BK), tl.float32)
    last_grad = tl.zeros((BK,), tl.float32)

    for e in tl.range(E):
        ge = tl.load(g2 + pe + e, mt, other=0.0)
        gel = tl.load(g2 + (last * H + h) * E + e)
        exp_gek = tl.exp(tl.minimum(gel - ge, 0.0))
        exp_end = tl.exp(gl + gel)
        contraction_k = tl.zeros((BT, BK), tl.float32)
        ss = tl.zeros((BK,), tl.float32)
        for iv in range(tl.cdiv(V, BV)):
            vv = iv * BV + tl.arange(0, BV).to(tl.int64)
            pv = (tt[:, None] * H + h) * V + vv[None, :]
            mv = mt[:, None] & (vv < V)[None, :]
            ph = ((gc * H + h) * E + e) * K * V + d[:, None] * V + vv[None, :]
            mh = md[:, None] & (vv < V)[None, :]

            dh_state = tl.load(dh + ph, mh, other=0.0)
            bv_u = tl.load(u + pv, mv, other=0.0).to(tl.bfloat16)
            contraction_k += tl.dot(bv_u, tl.trans(dh_state), out_dtype=tl.float32)
            previous = tl.load(states + ph, mh, other=0.0).to(tl.float32)
            ss += tl.sum(previous * dh_state.to(tl.float32), 1)

        contraction_k = contraction_k * exp_gek[:, None]
        ss = ss * exp_end
        ke = tl.load(k2 + pe + e, mt, other=0.0)
        acc_k += contraction_k * decay_k * ke[:, None]
        second_k = tl.sum(kv * decay_k * contraction_k, 1)
        pp = ((ik * TOTAL_T + tt) * H + h) * E + e
        tl.store(dk2 + pp, second_k, mt)
        qe = tl.load(q2 + pe + e, mt, other=0.0)
        re = ke * b_beta
        vg2 = qe * tl.load(dq2 + pp, mt, other=0.0) + re * tl.load(dr2 + pp, mt, other=0.0) - ke * second_k
        last_grad += ss
        last_e = tl.sum(ss, 0) + tl.sum(ke * second_k, 0)
        tl.store(dg2 + pp, vg2 + tl.where(tt == last, last_e, 0.0), mt)

    tl.store(dk + pk, acc_k, mask)

    last_grad += tl.sum(kv * acc_k, 0)
    ag = -kv * acc_k
    qv = tl.load(q + pk, mask, other=0.0).to(tl.float32)
    ag += qv * tl.load(dq + pk, mask, other=0.0)
    ag += kv * tl.load(dr + pk, mask, other=0.0)
    ag += tl.where((tt == last)[:, None], last_grad[None, :], 0.0)
    tl.store(dgk + pk, ag, mask)


@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_bwd_kernel_matrices(
    do,
    dx,
    u,
    v,
    beta,
    dv,
    dbeta_v,
    dA,
    dM,
    cu,
    indices,
    scale,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    VARLEN: tl.constexpr,
):
    c, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    start, end, _ = chunk_position(c, bh // H, T, cu, indices, BT, VARLEN)
    t = tl.arange(0, BT).to(tl.int64)
    tt = start + t
    mt = tt < end

    da, dm = tl.zeros((BT, BT), tl.float32), tl.zeros((BT, BT), tl.float32)
    b_beta = tl.load(beta + tt * H + bh % H, mt, other=0.0).to(tl.float32)
    dbeta_acc = tl.zeros((BT,), tl.float32)
    for iv in range(tl.cdiv(V, BV)):
        vv = iv * BV + tl.arange(0, BV).to(tl.int64)
        p = (tt[:, None] * H + bh % H) * V + vv[None, :]
        mask = mt[:, None] & (vv < V)[None, :]
        uu = tl.load(u + p, mask, other=0.0)
        dz = tl.load(dx + p, mask, other=0.0)
        vv_f = tl.load(v + p, mask, other=0.0).to(tl.float32)
        da -= tl.dot(dz.to(tl.float32), tl.trans(uu).to(tl.float32), input_precision='ieee')
        dm += tl.dot(tl.load(do + p, mask, other=0.0).to(tl.float32),
                     tl.trans(uu).to(tl.float32), input_precision='ieee') * scale
        tl.store(dv + p, dz * b_beta[:, None], mask)
        dbeta_acc += tl.sum(dz * vv_f, 1)

    tl.store(dbeta_v + tt * H + bh % H, dbeta_acc, mt)
    p = (tt[:, None] * H + bh % H) * BT + t[None, :]
    da = tl.where(t[:, None] > t[None, :], da, 0.0)
    dm = tl.where(t[:, None] >= t[None, :], dm, 0.0)
    tl.store(dA + p, da, mt[:, None])
    tl.store(dM + p, dm, mt[:, None])


@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_bwd_kernel_scores_e(
    q2,
    k2,
    beta,
    g2,
    Lk,
    Mk,
    dA,
    dM,
    dq,
    dk,
    dr,
    dg,
    outq,
    outk,
    dbeta_v,
    outg,
    cu,
    indices,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    VARLEN: tl.constexpr,
    NK: tl.constexpr,
    TOTAL_T: tl.constexpr,
):
    c, bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    start, end, _ = chunk_position(c, bh // H, T, cu, indices, BT, VARLEN)
    h = bh % H
    t = tl.arange(0, BT).to(tl.int64)
    tt = start + t
    mask = tt < end
    pa = (tt[:, None] * H + h) * BT + t[None, :]
    base_l = tl.load(dA + pa, mask[:, None], other=0.0) * tl.load(Lk + pa, mask[:, None], other=0.0)
    base_m = tl.load(dM + pa, mask[:, None], other=0.0) * tl.load(Mk + pa, mask[:, None], other=0.0)
    b_beta = tl.load(beta + tt * H + h, mask, other=0.0).to(tl.float32)
    dbeta_acc = tl.load(dbeta_v + tt * H + h, mask, other=0.0).to(tl.float32)

    for d in range(D):
        p = (tt * H + h) * D + d
        qi = tl.load(q2 + p, mask, other=0.0)
        ki = tl.load(k2 + p, mask, other=0.0)
        gi = tl.load(g2 + p, mask, other=0.0)
        ri = ki * b_beta

        first_g = tl.max(tl.where(t == 0, gi, float('-inf')), 0)
        last_g = tl.min(tl.where(mask, gi, float('inf')), 0)
        if first_g - last_g <= MAX_ANCHOR_SPAN:
            anchor = 0.5 * (first_g + last_g)
            left = tl.exp(tl.where(mask, gi - anchor, 0.0))
            right = tl.exp(tl.where(mask, anchor - gi, 0.0))
            decay = left[:, None] * right[None, :]
        else:
            decay = tl.exp(tl.minimum(gi[:, None] - gi[None, :], 0.0))

        dl, dm = base_l * decay, base_m * decay
        vq, vr = tl.sum(dm * ki[None, :], 1), tl.sum(dl * ki[None, :], 1)
        vk = tl.sum(dl * ri[:, None] + dm * qi[:, None], 0)
        diagonal = tl.sum(tl.where(t[:, None] == t[None, :], dm, 0.0), 1)

        inter_q, inter_k = tl.zeros((BT,), tl.float32), tl.zeros((BT,), tl.float32)
        inter_r, inter_g = tl.zeros((BT,), tl.float32), tl.zeros((BT,), tl.float32)
        for ik in tl.static_range(NK):
            offset = ik * TOTAL_T * H * D + p
            inter_q += tl.load(dq + offset, mask, other=0.0)
            inter_k += tl.load(dk + offset, mask, other=0.0)
            inter_r += tl.load(dr + offset, mask, other=0.0)
            inter_g += tl.load(dg + offset, mask, other=0.0)

        vg = qi * (vq - diagonal * ki) + ri * vr - ki * (vk - diagonal * qi) + inter_g
        erase = vr + inter_r

        tl.store(outq + p, vq + inter_q, mask)
        tl.store(outk + p, vk + inter_k + b_beta * erase, mask)
        dbeta_acc += ki * erase
        tl.store(outg + p, tl.cumsum(vg, reverse=True), mask)

    tl.store(dbeta_v + tt * H + h, dbeta_acc, mask)


@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_bwd_kernel_scores_k(
    q,
    k,
    g,
    q2,
    k2,
    beta,
    g2,
    dA,
    dM,
    dq,
    dk,
    dr,
    dg,
    outq,
    outk,
    outg,
    cu,
    indices,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    VARLEN: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64)
    c, block = block // (BT // BC), block % (BT // BC)
    bh, kd = tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    start, end, _ = chunk_position(c, bh // H, T, cu, indices, BT, VARLEN)
    h = bh % H
    i = block * BC + tl.arange(0, BC).to(tl.int64)
    d = kd * BD + tl.arange(0, BD).to(tl.int64)
    ti = start + i
    mi = (ti < end)[:, None] & (d < D)[None, :]
    pi = (ti[:, None] * H + h) * D + d[None, :]
    qi = tl.load(q + pi, mi, other=0.0)
    ki = tl.load(k + pi, mi, other=0.0)
    ri = ki.to(tl.float32)
    gi = tl.load(g + pi, mi, other=0.0)
    vq, vr, vk = tl.zeros((BC, BD), tl.float32), tl.zeros((BC, BD), tl.float32), tl.zeros((BC, BD), tl.float32)

    # load second-axis [BC, 16] slice for row block i (tl.dot requires K_dim >= 16)
    e_idx = tl.arange(0, 16).to(tl.int64)
    me_i = (ti < end)[:, None] & (e_idx < E)[None, :]
    pe_i = (ti[:, None] * H + h) * E + e_idx[None, :]
    q2_i = tl.load(q2 + pe_i, me_i, other=0.0)
    k2_i = tl.load(k2 + pe_i, me_i, other=0.0)
    b_beta_i = tl.load(beta + ti * H + h, ti < end, other=0.0).to(tl.float32)
    r2_i = k2_i * b_beta_i[:, None]
    g2_i = tl.load(g2 + pe_i, me_i, other=0.0)

    # off-diagonal sub-blocks before this block: dq and dr
    if start + block * BC < end and block > 0:
        gn = tl.load(g + ((start + block * BC) * H + h) * D + d, d < D, other=0.0)
        factor_q = tl.exp(tl.minimum(gi - gn[None, :], 0.0))
        g2_n = tl.load(g2 + ((start + block * BC) * H + h) * E + e_idx, e_idx < E, other=0.0)
        fg2_q = tl.where(me_i, tl.exp(tl.minimum(g2_i - g2_n[None, :], 0.0)), 0.0)
        for jb in range(0, block):
            oj = jb * BC + tl.arange(0, BC).to(tl.int64)
            tjb = start + oj
            mjb = (tjb < end)[:, None] & (d < D)[None, :]
            pjb = (tjb[:, None] * H + h) * D + d[None, :]
            gjb = tl.load(g + pjb, mjb, other=0.0)
            kjb = tl.load(k + pjb, mjb, other=0.0) * tl.exp(tl.minimum(gn[None, :] - gjb, 0.0))

            me_j = (tjb < end)[:, None] & (e_idx < E)[None, :]
            pe_j = (tjb[:, None] * H + h) * E + e_idx[None, :]
            k2_j = tl.load(k2 + pe_j, me_j, other=0.0)
            g2_j = tl.load(g2 + pe_j, me_j, other=0.0)
            k2_gt = tl.trans(k2_j * tl.where(me_j, tl.exp(tl.minimum(g2_n[None, :] - g2_j, 0.0)), 0.0))
            l2_sub = tl.dot(r2_i * fg2_q, k2_gt, input_precision='ieee')
            m2_sub = tl.dot(q2_i * fg2_q, k2_gt, input_precision='ieee')

            pa = (ti[:, None] * H + h) * BT + oj[None, :]
            ma = (ti < end)[:, None] & (tjb < end)[None, :]
            dl = tl.load(dA + pa, ma, other=0.0) * l2_sub
            dm = tl.load(dM + pa, ma, other=0.0) * m2_sub
            vr += state_dot(dl, kjb) * factor_q
            vq += state_dot(dm, kjb) * factor_q

    # off-diagonal sub-blocks after this block: dk only
    if start + (block + 1) * BC < end and block < BT // BC - 1:
        gn = tl.load(g + ((start + (block + 1) * BC - 1) * H + h) * D + d, d < D, other=0.0)
        factor_k = tl.exp(tl.minimum(gn[None, :] - gi, 0.0))
        g2_n = tl.load(g2 + ((start + (block + 1) * BC - 1) * H + h) * E + e_idx, e_idx < E, other=0.0)
        k2_i_scaled = k2_i * tl.where(me_i, tl.exp(tl.minimum(g2_n[None, :] - g2_i, 0.0)), 0.0)
        for jb in range(block + 1, BT // BC):
            oj = jb * BC + tl.arange(0, BC).to(tl.int64)
            tjb = start + oj
            mjb = (tjb < end)[:, None] & (d < D)[None, :]
            pjb = (tjb[:, None] * H + h) * D + d[None, :]
            gjb = tl.load(g + pjb, mjb, other=0.0)
            qjb = tl.load(q + pjb, mjb, other=0.0)
            rjb = tl.load(k + pjb, mjb, other=0.0).to(tl.float32)
            fg = tl.where((tjb < end)[:, None], tl.exp(tl.minimum(gjb - gn[None, :], 0.0)), 0.0)

            me_j = (tjb < end)[:, None] & (e_idx < E)[None, :]
            pe_j = (tjb[:, None] * H + h) * E + e_idx[None, :]
            q2_j = tl.load(q2 + pe_j, me_j, other=0.0)
            b_beta_j = tl.load(beta + tjb * H + h, tjb < end, other=0.0).to(tl.float32)
            r2_j = tl.load(k2 + pe_j, me_j, other=0.0) * b_beta_j[:, None]
            g2_j = tl.load(g2 + pe_j, me_j, other=0.0)
            fg2 = tl.where(me_j, tl.exp(tl.minimum(g2_j - g2_n[None, :], 0.0)), 0.0)
            l2_sub_t = tl.dot(k2_i_scaled, tl.trans(r2_j * fg2), input_precision='ieee')
            m2_sub_t = tl.dot(k2_i_scaled, tl.trans(q2_j * fg2), input_precision='ieee')

            pa = (tjb[None, :] * H + h) * BT + i[:, None]
            ma = (tjb < end)[None, :] & (ti < end)[:, None]
            dl = tl.load(dA + pa, ma, other=0.0) * l2_sub_t
            dm = tl.load(dM + pa, ma, other=0.0) * m2_sub_t
            vk += (state_dot(dl, rjb * fg) + state_dot(dm, qjb * fg)) * factor_k

    # diagonal [BC, BC] sub-block
    oi = tl.arange(0, BC).to(tl.int64)
    first_g2 = tl.max(tl.where(oi[:, None] == 0, g2_i, float('-inf')), 0)
    last_g2 = tl.min(tl.where(me_i, g2_i, float('inf')), 0)
    if tl.max(first_g2 - last_g2, 0) <= MAX_ANCHOR_SPAN:
        anchor2 = 0.5 * (first_g2 + last_g2)
        gq2 = tl.where(me_i, tl.exp(g2_i - anchor2[None, :]), 0.0)
        gk2_inv = tl.where(me_i, tl.exp(anchor2[None, :] - g2_i), 0.0)
        k2_gt_d = tl.trans(k2_i * gk2_inv)
        l2_d = tl.dot(r2_i * gq2, k2_gt_d, input_precision='ieee')
        m2_d = tl.dot(q2_i * gq2, k2_gt_d, input_precision='ieee')
    else:
        l2_d = tl.zeros((BC, BC), tl.float32)
        m2_d = tl.zeros((BC, BC), tl.float32)
        for col2 in range(BC):
            m_col2 = oi == col2
            g2_col = tl.sum(tl.where(m_col2[:, None], g2_i, 0.0), 0)
            k2_col = tl.sum(tl.where(m_col2[:, None], k2_i, 0.0), 0)
            k2_g = k2_col[None, :] * tl.exp(tl.minimum(g2_i - g2_col[None, :], 0.0))
            l2_d += tl.where(m_col2[None, :], tl.sum(r2_i * k2_g, 1)[:, None], 0.0)
            m2_d += tl.where(m_col2[None, :], tl.sum(q2_i * k2_g, 1)[:, None], 0.0)

    pa = (ti[:, None] * H + h) * BT + (block * BC + oi)[None, :]
    ma = (ti < end)[:, None] & (ti < end)[None, :] & (oi[:, None] >= oi[None, :])
    dl_d = tl.load(dA + pa, ma, other=0.0) * tl.where(ma, l2_d, 0.0)
    dm_d = tl.load(dM + pa, ma, other=0.0) * tl.where(ma, m2_d, 0.0)

    first_g = tl.max(tl.where(oi[:, None] == 0, gi, float('-inf')), 0)
    last_g = tl.min(tl.where(mi, gi, float('inf')), 0)
    if tl.max(first_g - last_g, 0) <= MAX_ANCHOR_SPAN:
        anchor = 0.5 * (first_g + last_g)
        gq = tl.where(mi, tl.exp(gi - anchor[None, :]), 0.0)
        gk_inv = tl.where(mi, tl.exp(anchor[None, :] - gi), 0.0)
        k_scaled = ki.to(tl.float32) * gk_inv
        vq += tl.dot(dm_d, k_scaled, input_precision='ieee') * gq
        vr += tl.dot(dl_d, k_scaled, input_precision='ieee') * gq
        vk += (
            tl.dot(tl.trans(dl_d), ri * gq, input_precision='ieee')
            + tl.dot(tl.trans(dm_d), qi.to(tl.float32) * gq, input_precision='ieee')
        ) * gk_inv
    else:
        for col in range(BC):
            m_col = oi == col
            g_col = tl.sum(tl.where(m_col[:, None], gi, 0.0), 0)
            r_col = tl.sum(tl.where(m_col[:, None], ri, 0.0), 0)
            dl_col = tl.sum(tl.where(m_col[None, :], dl_d, 0.0), 1)
            dm_col = tl.sum(tl.where(m_col[None, :], dm_d, 0.0), 1)
            decay = tl.exp(tl.minimum(gi - g_col[None, :], 0.0))
            kg = r_col[None, :] * decay
            vq += dm_col[:, None] * kg
            vr += dl_col[:, None] * kg
            vk_col = tl.sum((dl_col[:, None] * ri + dm_col[:, None] * qi.to(tl.float32)) * decay, 0)
            vk += tl.where(m_col[:, None], vk_col[None, :], 0.0)

    # accumulate into the saved first-axis gradients
    vg = qi * vq + ri * vr - ki * vk
    tl.store(outq + pi, tl.load(dq + pi, mi, other=0.0) + vq, mi)
    erase = tl.load(dr + pi, mi, other=0.0) + vr
    tl.store(outk + pi, tl.load(dk + pi, mi, other=0.0) + vk + erase, mi)
    tl.store(outg + pi, tl.load(dg + pi, mi, other=0.0) + vg, mi)


def chunk_triadic_kda_bwd(v, beta, q, k, q2, k2, gk, g2, Ai, Aqk, states, u, Lk, Mk, h0, cu, indices, do, dht, scale, keys):
    B, T, H, K = k.shape
    E, V = k2.shape[-1], v.shape[-1]
    N = len(cu) - 1 if cu is not None else B
    nc = len(indices) if cu is not None else triton.cdiv(T, CHUNK_SIZE)
    offsets = prepare_chunk_offsets(cu, CHUNK_SIZE) if cu is not None else None
    inter_block = 32
    nk = triton.cdiv(K, inter_block)

    if states is None:
        keys = [torch.empty_like(k, dtype=torch.bfloat16) for _ in range(3)]
        Akk = k.new_empty(B, T, H, CHUNK_SIZE, dtype=torch.float32)
        Lk, Mk = fwd_intra(q, k, q2, k2, beta, gk, g2, cu, indices, Akk=Akk, Aqk=Aqk, keys=keys)
        _, _, states, u = fwd_h(v, beta, q2, k2, gk, g2, Ai, Aqk, keys, h0, cu,
                                indices, scale, save=True, output=False, final=False)

    pq, pkp, pr = keys
    dx, dh = torch.empty_like(u), torch.empty_like(states)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    BV = 16 if E > 8 else (32 if (V >= 32 and triton.cdiv(V, 32) * N * H >= 96) else 16)
    chunk_triadic_kda_bwd_kernel_h[(triton.cdiv(V, BV), N * H)](
        q2=q2,
        k2=k2,
        beta=beta,
        gk=gk,
        g2=g2,
        Ai=Ai,
        Aqk=Aqk,
        pq=pq,
        pkp=pkp,
        pr=pr,
        do=do,
        dx=dx,
        dh=dh,
        dht=dht,
        dh0=dh0,
        cu=cu,
        offsets=offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        E=E,
        BT=CHUNK_SIZE,
        BK=max(16, triton.next_power_of_2(K)),
        BV=BV,
        VARLEN=cu is not None,
        INITIAL=h0 is not None,
        FINAL=dht is not None,
        num_warps=8,
        num_stages=1,
    )

    inter_k = [torch.empty_like(k, dtype=torch.float32) for _ in range(4)]
    inter_e = [k.new_empty(nk, B, T, H, E, dtype=torch.float32) for _ in range(4)]
    bv_inter = 32 if V >= 32 else 16
    chunk_triadic_kda_bwd_kernel_inter[(nk, nc, B * H)](
        q=q,
        k=k,
        q2=q2,
        k2=k2,
        beta=beta,
        gk=gk,
        g2=g2,
        states=states,
        dh=dh,
        do=do,
        dx=dx,
        u=u,
        dq=inter_k[0],
        dk=inter_k[1],
        dr=inter_k[2],
        dgk=inter_k[3],
        dq2=inter_e[0],
        dk2=inter_e[1],
        dr2=inter_e[2],
        dg2=inter_e[3],
        cu=cu,
        indices=indices,
        scale=scale,
        T=T,
        TOTAL_T=B * T,
        H=H,
        K=K,
        V=V,
        E=E,
        BT=CHUNK_SIZE,
        BK=inter_block,
        BV=bv_inter,
        VARLEN=cu is not None,
        num_warps=4,
        num_stages=2,
    )

    dA, dM = [torch.empty_like(Aqk, dtype=torch.float32) for _ in range(2)]
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta, dtype=torch.float32)
    dq2, dk2, dg2 = torch.empty_like(k2), torch.empty_like(k2), torch.empty_like(k2)
    chunk_triadic_kda_bwd_kernel_matrices[(nc, B * H)](
        do=do,
        dx=dx,
        u=u,
        v=v,
        beta=beta,
        dv=dv,
        dbeta_v=dbeta,
        dA=dA,
        dM=dM,
        cu=cu,
        indices=indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=CHUNK_SIZE,
        BV=32,
        VARLEN=cu is not None,
        num_warps=8,
        num_stages=1,
    )

    dq, dk, dgk = torch.empty_like(q), torch.empty_like(k), torch.empty_like(gk)
    chunk_triadic_kda_bwd_kernel_scores_e[(nc, B * H)](
        q2=q2,
        k2=k2,
        beta=beta,
        g2=g2,
        Lk=Lk,
        Mk=Mk,
        dA=dA,
        dM=dM,
        dq=inter_e[0],
        dk=inter_e[1],
        dr=inter_e[2],
        dg=inter_e[3],
        outq=dq2,
        outk=dk2,
        dbeta_v=dbeta,
        outg=dg2,
        cu=cu,
        indices=indices,
        T=T,
        H=H,
        D=E,
        BT=CHUNK_SIZE,
        VARLEN=cu is not None,
        NK=nk,
        TOTAL_T=B * T,
        num_warps=8,
    )
    chunk_triadic_kda_bwd_kernel_scores_k[(nc * 4, B * H, nk)](
        q=q,
        k=k,
        g=gk,
        q2=q2,
        k2=k2,
        beta=beta,
        g2=g2,
        dA=dA,
        dM=dM,
        dq=inter_k[0],
        dk=inter_k[1],
        dr=inter_k[2],
        dg=inter_k[3],
        outq=dq,
        outk=dk,
        outg=dgk,
        cu=cu,
        indices=indices,
        T=T,
        H=H,
        D=K,
        E=E,
        BD=32,
        BT=CHUNK_SIZE,
        BC=16,
        VARLEN=cu is not None,
        num_warps=4,
        num_stages=1,
    )

    dgk = chunk_local_cumsum(dgk, chunk_size=CHUNK_SIZE, reverse=True, cu_seqlens=cu, chunk_indices=indices)
    return dq, dk, dv, dk2, dq2, dgk, dg2, dbeta, dh0
