# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets, solve_tril

CHUNK_SIZE = 64


@triton.jit
def state_dot(a, b):
    return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), out_dtype=tl.float32)


@triton.jit
def chunk_position(i_c, i_b, T, cu, indices, BT: tl.constexpr, VARLEN: tl.constexpr):
    if VARLEN:
        n = tl.load(indices + i_c * 2).to(tl.int64)
        c = tl.load(indices + i_c * 2 + 1).to(tl.int64)
        start = tl.load(cu + n).to(tl.int64)
        end = tl.load(cu + n + 1).to(tl.int64)
        global_chunk = i_c
    else:
        c = i_c
        start, end = i_b * T, (i_b + 1) * T
        global_chunk = i_b * tl.cdiv(T, BT) + i_c
    return start + c * BT, end, global_chunk


@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_fwd_kernel_scores_k(
    q,
    k,
    g,
    L_axis,
    M_axis,
    pq,
    pk,
    pr,
    cu,
    indices,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    VARLEN: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64)
    c, block = block // (BT // BC), block % (BT // BC)
    bh = tl.program_id(1).to(tl.int64)
    start, end, _ = chunk_position(c, bh // H, T, cu, indices, BT, VARLEN)
    h = bh % H

    oi = tl.arange(0, BC).to(tl.int64)
    ti = start + block * BC + oi
    if start + block * BC >= end:
        return

    # strict lower-triangular sub-blocks (jb < block)
    for jb in range(0, block):
        oj = jb * BC + tl.arange(0, BC).to(tl.int64)
        tj = start + oj
        erase_off = tl.zeros((BC, BC), tl.float32)
        query_off = tl.zeros((BC, BC), tl.float32)
        for ik in range(tl.cdiv(D, BK)):
            d = ik * BK + tl.arange(0, BK).to(tl.int64)
            mi = (ti < end)[:, None] & (d < D)[None, :]
            mj = (tj < end)[:, None] & (d < D)[None, :]
            pi = (ti[:, None] * H + h) * D + d[None, :]
            pj = (tj[:, None] * H + h) * D + d[None, :]
            qi = tl.load(q + pi, mi, other=0.0).to(tl.float32)
            ki = tl.load(k + pi, mi, other=0.0).to(tl.float32)
            gi = tl.load(g + pi, mi, other=0.0)
            gn = tl.load(g + ((start + block * BC) * H + h) * D + d, d < D, other=0.0)
            fgq = tl.where(mi, tl.exp(tl.minimum(gi - gn[None, :], 0.0)), 0.0)
            kj = tl.load(k + pj, mj, other=0.0).to(tl.float32)
            gj = tl.load(g + pj, mj, other=0.0)
            kgt = tl.trans(kj * tl.where(mj, tl.exp(tl.minimum(gn[None, :] - gj, 0.0)), 0.0))
            erase_off = tl.dot(ki * fgq, kgt, erase_off, input_precision='ieee')
            query_off = tl.dot(qi * fgq, kgt, query_off, input_precision='ieee')
        pa = (ti[:, None] * H + h) * BT + oj[None, :]
        ma = (ti < end)[:, None] & (tj < end)[None, :]
        tl.store(L_axis + pa, erase_off, ma)
        tl.store(M_axis + pa, query_off, ma)

    # upper-triangular sub-blocks (jb > block)
    for jb in range(block + 1, BT // BC):
        oj = jb * BC + tl.arange(0, BC).to(tl.int64)
        pa = (ti[:, None] * H + h) * BT + oj[None, :]
        tl.store(L_axis + pa, tl.zeros((BC, BC), tl.float32), (ti < end)[:, None])
        tl.store(M_axis + pa, tl.zeros((BC, BC), tl.float32), (ti < end)[:, None])

    # diagonal [BC, BC] sub-block and bf16 decayed keys
    erase_d = tl.zeros((BC, BC), tl.float32)
    query_d = tl.zeros((BC, BC), tl.float32)
    m_causal = oi[:, None] >= oi[None, :]
    last = tl.minimum(start + BT, end) - 1
    for ik in range(tl.cdiv(D, BK)):
        d = ik * BK + tl.arange(0, BK).to(tl.int64)
        mi = (ti < end)[:, None] & (d < D)[None, :]
        pi = (ti[:, None] * H + h) * D + d[None, :]
        qi = tl.load(q + pi, mi, other=0.0).to(tl.float32)
        ki = tl.load(k + pi, mi, other=0.0).to(tl.float32)
        gi = tl.load(g + pi, mi, other=0.0)

        gl = tl.load(g + (last * H + h) * D + d, d < D, other=0.0)
        exp_g = tl.exp(gi)
        tl.store(pq + pi, (qi * exp_g).to(tl.bfloat16), mi)
        tl.store(pr + pi, (ki * exp_g).to(tl.bfloat16), mi)
        tl.store(pk + pi, (ki * tl.exp(tl.minimum(gl[None, :] - gi, 0.0))).to(tl.bfloat16), mi)

        first_g = tl.max(tl.where(oi[:, None] == 0, gi, float('-inf')), 0)
        last_g = tl.min(tl.where(mi, gi, float('inf')), 0)
        anchor = 0.5 * (first_g + last_g)
        gq = tl.where(mi, tl.exp(gi - anchor[None, :]), 0.0)
        gk_inv = tl.where(mi, tl.exp(anchor[None, :] - gi), 0.0)
        kgt = tl.trans(ki * gk_inv)
        erase_d = tl.dot(ki * gq, kgt, erase_d, input_precision='ieee')
        query_d = tl.dot(qi * gq, kgt, query_d, input_precision='ieee')

    pa_diag = (ti[:, None] * H + h) * BT + (block * BC + oi)[None, :]
    tl.store(L_axis + pa_diag, tl.where(m_causal, erase_d, 0.0), (ti < end)[:, None])
    tl.store(M_axis + pa_diag, tl.where(m_causal, query_d, 0.0), (ti < end)[:, None])


@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_fwd_kernel_scores_e(
    q2,
    k2,
    beta,
    g2,
    Lk,
    Mk,
    Akk,
    Aqk,
    cu,
    indices,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    BT: tl.constexpr,
    VARLEN: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) % BT
    c = tl.program_id(0).to(tl.int64) // BT
    bh = tl.program_id(1).to(tl.int64)
    start, end, _ = chunk_position(c, bh // H, T, cu, indices, BT, VARLEN)
    h = bh % H
    if start + i >= end:
        return

    j = tl.arange(0, BT).to(tl.int64)
    d = tl.arange(0, BE).to(tl.int64)
    row = (start + i) * H + h
    pi = row * E + d
    pj = ((start + j[:, None]) * H + h) * E + d[None, :]
    qi = tl.load(q2 + pi, d < E, other=0.0)
    ki = tl.load(k2 + pi, d < E, other=0.0)
    ri = ki * tl.load(beta + row).to(tl.float32)
    gi = tl.load(g2 + pi, d < E, other=0.0)
    mask = (j <= i)[:, None] & (d < E)[None, :]
    kj = tl.load(k2 + pj, mask, other=0.0)
    gj = tl.load(g2 + pj, mask, other=0.0)

    decay = tl.exp(tl.minimum(gi[None, :] - gj, 0.0))
    erase_score = tl.sum(kj * decay * ri[None, :], 1)
    query_score = tl.sum(kj * decay * qi[None, :], 1)

    p = row * BT + j
    tl.store(Akk + p, tl.where(i > j, erase_score * tl.load(Lk + p), 0.0))
    tl.store(Aqk + p, query_score * tl.load(Mk + p))


def fwd_intra(q, k, q2, k2, beta, gk, g2, cu, indices, Akk, Aqk, keys):
    B, T, H, K = k.shape
    E = k2.shape[-1]
    nc = len(indices) if cu is not None else triton.cdiv(T, CHUNK_SIZE)
    Lk, Mk = [k.new_empty(B, T, H, CHUNK_SIZE, dtype=torch.float32) for _ in range(2)]
    pq, pk, pr = keys

    chunk_triadic_kda_fwd_kernel_scores_k[(nc * 4, B * H)](
        q=q,
        k=k,
        g=gk,
        L_axis=Lk,
        M_axis=Mk,
        pq=pq,
        pk=pk,
        pr=pr,
        cu=cu,
        indices=indices,
        T=T,
        H=H,
        D=K,
        BK=32,
        BT=CHUNK_SIZE,
        BC=16,
        VARLEN=cu is not None,
        num_warps=2,
    )
    chunk_triadic_kda_fwd_kernel_scores_e[(nc * CHUNK_SIZE, B * H)](
        q2=q2,
        k2=k2,
        beta=beta,
        g2=g2,
        Lk=Lk,
        Mk=Mk,
        Akk=Akk,
        Aqk=Aqk,
        cu=cu,
        indices=indices,
        T=T,
        H=H,
        E=E,
        BE=triton.next_power_of_2(E),
        BT=CHUNK_SIZE,
        VARLEN=cu is not None,
        num_warps=4,
    )
    return Lk, Mk


@triton.jit(do_not_specialize=['T'])
def chunk_triadic_kda_fwd_kernel_h(
    v,
    beta,
    q2,
    k2,
    gk,
    g2,
    Ai,
    Aqk,
    pq,
    pkp,
    pr,
    o,
    u,
    states,
    h0,
    ht,
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
    SAVE: tl.constexpr,
    OUTPUT: tl.constexpr,
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

    # one live [BV, K] tile per E slice, rotated across the chunk loop
    hs = ()
    for e in tl.static_range(E):
        if INITIAL:
            value = tl.load(h0 + (nh * E + e) * K * V + ph, mh, other=0.0)
        else:
            value = tl.zeros((BV, BK), tl.float32)
        hs += (value,)

    for c in tl.range(nc):
        tt = start + c * BT + t
        mt = tt < end
        pk = (tt[:, None] * H + h) * K + d[None, :]
        mk = mt[:, None] & (d < K)[None, :]
        pv = (tt[None, :] * H + h) * V + vv[:, None]
        mv = (vv < V)[:, None] & mt[None, :]
        pe = (tt * H + h) * E
        last = tl.minimum(start + (c + 1) * BT, end) - 1

        if OUTPUT:
            output = tl.zeros((BV, BT), tl.float32)
        b_beta = tl.load(beta + tt * H + h, mt, other=0.0).to(tl.float32)
        residual = tl.load(v + pv, mv, other=0.0).to(tl.float32) * b_beta[None, :]

        # read every slice: subtract the erase contribution and accumulate the query output
        rg_t = tl.trans(tl.load(pr + pk, mk, other=0.0))
        if OUTPUT:
            qg_t = tl.trans(tl.load(pq + pk, mk, other=0.0))
        for read_e in tl.range(E):
            ge = tl.load(g2 + pe + read_e, mt, other=0.0)
            ke = tl.load(k2 + pe + read_e, mt, other=0.0)
            re = ke * b_beta * tl.exp(ge)
            hp = hs[0].to(tl.bfloat16)

            if SAVE:
                tl.store(states + (((first + c) * H + h) * E + read_e) * K * V + ph, hp, mh)

            residual -= tl.dot(hp, rg_t, out_dtype=tl.float32) * re[None, :]
            if OUTPUT:
                qe = tl.load(q2 + pe + read_e, mt, other=0.0) * tl.exp(ge)
                output += tl.dot(hp, qg_t, out_dtype=tl.float32) * qe[None, :]

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

        # intra-chunk inverse, then the query score on the same residual
        pa = (tt[None, :] * H + h) * BT + t[:, None]
        ai = tl.load(Ai + pa, mt[None, :], other=0.0)
        residual = state_dot(residual, ai)

        if SAVE:
            tl.store(u + pv, residual, mv)

        if OUTPUT:
            m = tl.load(Aqk + pa, mt[None, :], other=0.0)
            intra = state_dot(residual, m)
            output = (output + intra) * scale
            tl.store(o + pv, output, mv)

        # decay the current slice and rotate the updated state to the end
        gl = tl.load(gk + (last * H + h) * K + d, d < K, other=0.0)
        kg = tl.load(pkp + pk, mk, other=0.0)
        for write_e in tl.range(E):
            ge = tl.load(g2 + pe + write_e, mt, other=0.0)
            gel = tl.load(g2 + (last * H + h) * E + write_e)
            ke = tl.load(k2 + pe + write_e, mt, other=0.0) * tl.exp(tl.minimum(gel - ge, 0.0))
            decay = tl.exp(gl + gel)
            update = tl.dot((residual * ke[None, :]).to(tl.bfloat16), kg, out_dtype=tl.float32)
            value = hs[0] * decay[None, :] + update

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

    if FINAL:
        for e in tl.static_range(E):
            tl.store(ht + (nh * E + e) * K * V + ph, hs[e], mh)


def fwd_h(v, beta, q2, k2, gk, g2, Ai, Aqk, keys, h0, cu, indices, scale, save=True, output=True, final=False):
    B, T, H, V = v.shape
    E, K = k2.shape[-1], gk.shape[-1]
    N = len(cu) - 1 if cu is not None else B
    nc = len(indices) if cu is not None else B * triton.cdiv(T, CHUNK_SIZE)
    offsets = prepare_chunk_offsets(cu, CHUNK_SIZE) if cu is not None else None
    pq, pkp, pr = keys

    o = torch.empty_like(v) if output else None
    u = torch.empty_like(v, dtype=torch.float32) if save else None
    states = v.new_empty(nc, H, E, K, V, dtype=torch.bfloat16) if save else None
    ht = v.new_empty(N, H, E, K, V, dtype=torch.float32) if final else None
    BV = 16 if E > 8 else (32 if (V >= 32 and triton.cdiv(V, 32) * N * H >= 96) else 16)

    chunk_triadic_kda_fwd_kernel_h[(triton.cdiv(V, BV), N * H)](
        v=v,
        beta=beta,
        q2=q2,
        k2=k2,
        gk=gk,
        g2=g2,
        Ai=Ai,
        Aqk=Aqk,
        pq=pq,
        pkp=pkp,
        pr=pr,
        o=o,
        u=u,
        states=states,
        h0=h0,
        ht=ht,
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
        FINAL=final,
        SAVE=save,
        OUTPUT=output,
        num_warps=8,
        num_stages=1,
    )
    return o, ht, states, u


def chunk_triadic_kda_fwd(
    q,
    k,
    v,
    k2,
    q2,
    gk,
    g2,
    beta,
    h0,
    cu,
    scale,
    save,
    final,
    indices=None,
    cumsum_gates=True,
):
    B, T, H, _ = k.shape
    if cu is not None and indices is None:
        indices = prepare_chunk_indices(cu, CHUNK_SIZE)
    Akk = k.new_empty(B, T, H, CHUNK_SIZE, dtype=torch.float32)
    Aqk = k.new_empty(B, T, H, CHUNK_SIZE, dtype=torch.bfloat16)
    if cumsum_gates:
        gk = chunk_local_cumsum(gk, chunk_size=CHUNK_SIZE, cu_seqlens=cu, chunk_indices=indices)
        g2 = chunk_local_cumsum(g2, chunk_size=CHUNK_SIZE, cu_seqlens=cu, chunk_indices=indices)

    keys = [torch.empty_like(k, dtype=torch.bfloat16) for _ in range(3)]
    Lk, Mk = fwd_intra(q, k, q2, k2, beta, gk, g2, cu, indices, Akk=Akk, Aqk=Aqk, keys=keys)
    Ai = solve_tril(A=Akk, cu_seqlens=cu, chunk_indices=indices, output_dtype=torch.bfloat16)
    o, ht, states, u = fwd_h(v, beta, q2, k2, gk, g2, Ai, Aqk, keys, h0, cu, indices, scale, save=save, output=True, final=final)
    if not save:
        keys = [None] * 3

    saved = (v, beta, q, k, q2, k2, gk, g2, Ai, Aqk, states, u, Lk, Mk, *keys, h0, cu, indices)
    return o, ht, saved
