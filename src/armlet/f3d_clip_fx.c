/*
 * arMLEt - f3d_clip_fx.c: the F3D clipper (overlay 1) as a pure function; docs/ucode-f3d.md 9
 *
 * A triangle that crosses a plane of the guard-band frustum is clipped against the six planes in the microcode's order; each cut makes up to two new vertices (clip coordinates interpolated with a
 * 16-bit weight from a two-stage Newton reciprocal, colours and texture coordinates with the same weights) that go through the vertex pipeline's tail (screen coordinates, flags, fog); the
 * polygon becomes a fan of triangles. The model is the one verified against the JIT (docs/ucode-f3d.md 9.6).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <string.h>

#include "f3d_fx.h"
#include "hle_fx.h"
#include "f3d_fx_int.h"

#ifndef F3D_FX_MUTATE
#define F3D_FX_MUTATE 0
#endif
#define FMUT(n, ok, bad) (F3D_FX_MUTATE == (n) ? (bad) : (ok))

typedef uint16_t u16;
typedef int16_t s16;

void f3d_vtx_tail(const struct f3d_vctx *c, const uint16_t hi[4], const uint16_t lo[4], struct f3d_vtx *o, uint16_t shi_out[4], uint16_t slo_out[4]);
uint8_t f3d_fog_byte(const struct f3d_vctx *c, uint16_t shi2, uint16_t slo2);

/* (hi : lo) per lane summed over the four lanes as the microcode's two VADDC / VADD rounds do: lanes (0 + 0?, ...) see docs 9.2.3 S1 */
static void vsum(const u16 hi[4], const u16 lo[4], u16 *rhi, u16 *rlo)
{
    u16 l26[4], h27[4];
    for (int k = 0; k < 4; k++) {
        int src = k & 2;
        uint32_t s = (uint32_t)lo[k] + lo[src];
        l26[k] = (u16)s;
        h27[k] = (u16)hle_clamp16((int32_t)(s16)hi[k] + (s16)hi[src] + (int32_t)(s >> 16));
    }
    uint32_t s = (uint32_t)l26[3] + l26[1];
    *rlo = (u16)s;
    *rhi = (u16)hle_clamp16((int32_t)(s16)h27[3] + (s16)h27[1] + (int32_t)(s >> 16));
}

static inline u16 lo_of(hle_acc a) { return hle_uclamp(a); }
static inline u16 hi_of(hle_acc a) { return hle_sclamp(a); }

/* the new vertex on plane coefficients cf from P (outside) and Q (inside) */
static void new_vertex(const struct f3d_vctx *ctx, const struct f3d_vtx *P, const struct f3d_vtx *Q, const s16 cf[4], struct f3d_vtx *o)
{
    u16 v11[4], v12[4], v29d[4], v28d[4];
    for (int k = 0; k < 4; k++) {
        s16 negc = hle_clamp16(-(int32_t)cf[k]);
        hle_acc acc = hle_wrap48((hle_acc)Q->cf[k] * cf[k] + (((hle_acc)Q->cx[k] * cf[k]) << 16));
        v11[k] = hi_of(acc); v12[k] = lo_of(acc);
        acc = hle_wrap48(acc + (hle_acc)P->cf[k] * negc + (((hle_acc)P->cx[k] * negc) << 16));
        v29d[k] = hi_of(acc); v28d[k] = lo_of(acc);
    }
    u16 Dhi, Dlo, dQhi, dQlo;
    vsum(v29d, v28d, &Dhi, &Dlo);
    vsum(v11, v12, &dQhi, &dQlo);
    int32_t D32 = (int32_t)((uint32_t)Dhi << 16 | Dlo);
    int32_t R = (int32_t)hle_rcp32(D32);
    hle_acc acc = (hle_acc)(u16)R * 2;
    u16 v3 = lo_of(acc);
    acc = hle_wrap48(acc + (((hle_acc)(s16)(R >> 16) * 2) << 16));
    u16 v7 = hi_of(acc);
    if ((s16)Dhi < 0) {
        acc = hle_wrap48(-(hle_acc)v3);
        v3 = lo_of(acc);
        acc = hle_wrap48(acc + (((hle_acc)(s16)v7 * -1) << 16));
        v7 = hi_of(acc);
    }
    u16 R16 = v7 == 0 ? v3 : 0xffff;
    int32_t X = (int32_t)((uint32_t)hi_of(hle_mac_32xu16(D32, R16)) << 16 | lo_of(hle_mac_32xu16(D32, R16)));
    struct hle_pair Rx = f3d_recip_refine(X);
    int32_t dQ = (int32_t)((uint32_t)dQhi << 16 | dQlo), Rx32 = hle_pair32(Rx);
    u16 w1lo = lo_of(hle_mac_32x32_before_h(dQ, Rx32)), w1hi = hi_of(hle_mac_32x32(dQ, Rx32));
    hle_acc ua = hle_wrap48(FMUT(1, ctx->clip_acc && F3D_FX_MUTATE != 40 ? ctx->clip_acc : 3, 4) + hle_mac_32xu16((int32_t)((uint32_t)w1hi << 16 | w1lo), R16));
    u16 uhi = hi_of(ua), ulo = lo_of(ua);
    u16 wP = uhi == 0 ? ulo : 0xffff;
    if (wP == 0) wP = 1;
    u16 wQ = (u16)(FMUT(2, 0, 1) - wP);
    u16 nhi[4], nlo[4];
    for (int k = 0; k < 4; k++) {
        hle_acc a = (hle_acc)(((uint32_t)Q->cf[k] * wQ) >> 16) + (hle_acc)Q->cx[k] * wQ + (hle_acc)(((uint32_t)P->cf[k] * wP) >> 16) + (hle_acc)P->cx[k] * wP;
        a = hle_wrap48(a);
        nhi[k] = hi_of(a);
        nlo[k] = lo_of(a);
    }
    /* colours (bytes << 7) and texture coordinates */
    s16 att[6];
    for (int i = 0; i < 6; i++) {
        s16 p, q;
        if (i < 4) { p = (s16)(P->col[i] << 7); q = (s16)(Q->col[i] << 7); }
        else if (i == 4) { p = P->s; q = Q->s; }
        else { p = P->t; q = Q->t; }
        hle_acc a = hle_wrap48((hle_acc)p * wP + (hle_acc)q * wQ);
        att[i] = (s16)hi_of(FMUT(5, a, hle_wrap48(a + 0x8000)));            /* mutation 5: rounded */
    }
    u16 shi[4], slo[4];
    f3d_vtx_tail(ctx, nhi, nlo, o, shi, slo);
    if (ctx->flags_variant == F3D_FLAGS_PW && F3D_FX_MUTATE != 22)   /* mutation 22: the near-plane bit kept */
        o->flags &= (uint16_t)~0x40;                       /* a new vertex is the first of a pair in the vertex code */
    for (int i = 0; i < 4; i++) o->col[i] = (uint8_t)(att[i] >> 7);
    o->s = att[4];
    o->t = att[5];
    if (ctx->geom & F3D_G_FOG)
        o->col[3] = f3d_fog_byte(ctx, shi[2], slo[2]);
}

void f3d_clip_tri_scalar(const struct f3d_vctx *c, const struct f3d_vtx *const v[3], struct f3d_clip_out *o)
{
    const struct f3d_vtx *poly[16], *out[16];
    unsigned n = 3;
    poly[0] = v[2]; poly[1] = v[1]; poly[2] = v[0];                   /* the list starts reversed: third, second, first */
    o->ntri = 0;
    o->nnew = 0;
    for (int pi = 0; pi < 6; pi++) {
        const int p = F3D_FX_MUTATE == 4 ? 5 - pi : pi;                      /* mutation 4: the planes in reverse order */
        const u16 mask = c->plane_mask[p];
        int outside[16], nout = 0, first = -1;
        for (unsigned i = 0; i < n; i++) {
            outside[i] = (poly[i]->flags & mask) != 0;
            nout += outside[i];
            if (outside[i] && first < 0) first = (int)i;
        }
        if (nout == 0)
            continue;                                                  /* the plane is a no-op */
        if (nout == (int)n)
            return;                                                    /* everything is outside: the triangle is discarded */
        unsigned j = ((unsigned)first + 1) % n;
        while (outside[j]) j = (j + 1) % n;
        unsigned k = (j + 1) % n;
        while (!outside[k]) k = (k + 1) % n;
        unsigned m = 0;
        struct f3d_vtx *nv = &o->nv[o->nnew++];
        new_vertex(c, poly[(j + n - 1) % n], poly[j], c->plane[p], nv);
        out[m++] = nv;
        for (unsigned i = j; i != k; i = (i + 1) % n)
            out[m++] = poly[i];
        nv = &o->nv[o->nnew++];
        new_vertex(c, poly[k], poly[(k + n - 1) % n], c->plane[p], nv);
        out[m++] = nv;
        memcpy(poly, out, m * sizeof(poly[0]));
        n = m;
    }
    /* the fan: (poly[i+2], poly[i+1], poly[0]) */
    for (unsigned i = 0; i + 2 < n && o->ntri < 7; i++) {
        o->tri[o->ntri][0] = poly[i + 2];
        o->tri[o->ntri][1] = F3D_FX_MUTATE == 3 ? poly[(i + 3) % n] : poly[i + 1];   /* mutation 3 */
        o->tri[o->ntri][2] = poly[0];
        o->ntri++;
    }
}

void f3d_clip_tri_seq(const struct f3d_vctx *c, const struct f3d_vtx *const v[3], struct f3d_clip_out *o)
{
    const struct f3d_vtx *poly[12], *out[12];
    unsigned n = 3;
    poly[0] = v[0]; poly[1] = v[1]; poly[2] = v[2];
    o->ntri = 0;
    o->nnew = 0;
    o->npoly = 0;
    for (int p = 0; p < 6; p++) {
        const u16 mask = c->plane_mask[p];
        unsigned m = 0, idx = 1;
        u16 a2 = (u16)(poly[0]->flags & mask);
        const struct f3d_vtx *a3 = poly[0];
        if (!a2 && F3D_FX_MUTATE != 41)
            out[m++] = poly[0];
        for (;;) {
            const struct f3d_vtx *a0 = idx < n ? poly[idx] : NULL, *sp = a0 ? a0 : poly[0];
            u16 v1 = (u16)(sp->flags & mask);
            idx++;
            if (v1 != a2) {                /* mutation 39: the planes separated by a boolean, not by the masked value */
                const struct f3d_vtx *in = (v1 != 0) != (F3D_FX_MUTATE == 38) ? a3 : sp, *outv = (v1 != 0) != (F3D_FX_MUTATE == 38) ? sp : a3;      /* mutation 38: anchor swapped */
                struct f3d_vtx *nv;
                a2 = v1;
                if (o->nnew >= F3D_CLIP_MAX_NEW)
                    return;
                nv = &o->nv[o->nnew++];
                new_vertex(c, outv, in, c->plane[p], nv);
                out[m++] = nv;
            }
            if (!a0)
                break;
            a3 = sp;
            if (!a2)
                out[m++] = sp;
            if (m >= (F3D_FX_MUTATE == 39 ? 5 : 10))                   /* mutation 39: a smaller room for the polygon */
                return;
        }
        if (!m)
            return;                                                    /* everything is outside: the triangle is discarded */
        memcpy(poly, out, m * sizeof(poly[0]));
        n = m;
    }
    for (unsigned i = 0; i < n; i++)
        o->poly[i] = poly[i];
    o->npoly = n;
    for (unsigned i = 0; i + 2 < n && o->ntri < 7; i++) {
        o->tri[o->ntri][0] = poly[0];
        o->tri[o->ntri][1] = poly[F3D_FX_MUTATE == 37 ? i + 2 : i + 1];      /* mutation 37: fan order */
        o->tri[o->ntri][2] = poly[F3D_FX_MUTATE == 37 ? i + 1 : i + 2];
        o->ntri++;
    }
}

/* the NEON forms of these functions are not written yet: the scalar form serves */
void f3d_clip_tri(const struct f3d_vctx *c, const struct f3d_vtx *const v[3], struct f3d_clip_out *o) { f3d_clip_tri_scalar(c, v, o); }
