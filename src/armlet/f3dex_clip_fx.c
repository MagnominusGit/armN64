/*
 * arMLEt - f3dex_clip_fx.c: the F3DEX 1.x / F3DLX triangle clipper (scalar reference)
 *
 * Sutherland-Hodgman over the six planes in the microcode's order, on lists of vertex pointers; docs/ucode-f3dex/clip.md (the numbers in the comments are its sections). A new vertex is the
 * crossing of an edge with a plane: the weight comes from a rough 16-bit reciprocal and one refined reciprocal of the plane values (6.2), the clip position, the colour bytes and the texture
 * coordinates are interpolated with it (6.3), and the vertex tail of f3dex_vtx_fx.c (flags, 1/w, screen position, fog) completes the record (7). The result polygon is submitted as a fan.
 * The sums are hle_fx.h's closed forms of the multiply-accumulate chains; the places where the microcode reads the low word of a chain before the last product is added are marked "lo3".
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <string.h>

#include "f3dex_fx_int.h"

#ifdef F3DEX_CLIP_COVER
/* coverage counters of the tests (-DF3DEX_CLIP_COVER): printed at exit */
#include <stdio.h>
#include <stdlib.h>
struct f3dex_clip_cover {
    unsigned calls, plane_new[6], plane_noop[6], plane_disc[6], nnew[16], npoly[16], cf_sat, x_zero, sf_sat, sf1, sf0, xneg;
};
struct f3dex_clip_cover f3dex_clip_cov;
static void cov_dump(void)
{
    const struct f3dex_clip_cover *k = &f3dex_clip_cov;
    fprintf(stderr, "clip coverage: calls %u\n  planes near far y+ x+ y- x- (order of the microcode)\n  new-vertex pairs:", k->calls);
    for (int i = 0; i < 6; i++) fprintf(stderr, " %u", k->plane_new[i]);
    fprintf(stderr, "\n  no-op:");
    for (int i = 0; i < 6; i++) fprintf(stderr, " %u", k->plane_noop[i]);
    fprintf(stderr, "\n  discard:");
    for (int i = 0; i < 6; i++) fprintf(stderr, " %u", k->plane_disc[i]);
    fprintf(stderr, "\n  new vertices per call 0..12:");
    for (int i = 0; i < 13; i++) fprintf(stderr, " %u", k->nnew[i]);
    fprintf(stderr, "\n  final polygon size 3..9:");
    for (int i = 3; i < 10; i++) fprintf(stderr, " %u", k->npoly[i]);
    fprintf(stderr, "\n  cf saturated %u, X == 0 %u, X < 0 %u, weight saturated (Si != 0) %u, sf == 1 %u, sf == 0 (clamped) %u\n", k->cf_sat, k->x_zero, k->xneg, k->sf_sat, k->sf1, k->sf0);
}
static void cov_reg(void)
{
    static int done;
    if (!done) {
        done = 1;
        atexit(cov_dump);
    }
}
#define COV(x) do { x; } while (0)
#else
#define COV(x) ((void)0)
#endif

/* ---- 6.1 plane values -------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* the four lanes summed as the microcode's two rounds of VADDC / VADD do: stage 1 t_l = v_l + v_(l & ~1), stage 2 t_3 + t_1 (the integer halves saturate, the carry of the fractions goes in) */
static int32_t lane3_sum(const struct hle_pair v[4])
{
    int32_t t[4];
    for (int l = 0; l < 4; l++)
        t[l] = hle_pair32(hle_add32(hle_pair32(v[l]), hle_pair32(v[l & ~1])));
    return hle_pair32(hle_add32(t[3], FMUT(56, t[1], t[0])));
}

/* X = D(Q) - D(O) and DQ = D(Q), the plane c . P of the 32-bit clip coordinates */
static void plane_values(const int16_t c[4], const struct f3dex_vtx *O, const struct f3dex_vtx *Q, int32_t *X, int32_t *DQ)
{
    struct hle_pair pQ[4], dX[4];
    for (int l = 0; l < 4; l++) {
        int16_t nc = hle_clamp16(-(int32_t)c[l]);
        hle_acc aq = hle_wrap48((hle_acc)Q->cf[l] * c[l] + (((hle_acc)Q->cx[l] * c[l]) << 16));            /* VMUDN VMADH */
        hle_acc ax = hle_wrap48(aq + (hle_acc)O->cf[l] * nc + (((hle_acc)O->cx[l] * nc) << 16));            /* VMADN VMADH with -c */
        pQ[l] = hle_pair_acc(aq);
        dX[l] = hle_pair_acc(ax);
    }
    *X = lane3_sum(dX);
    *DQ = lane3_sum(pQ);
}

/* ---- 6.2 the weights --------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* sf = the weight of the outside vertex, tf = 65536 - sf that of the inside vertex (both 1 .. 65535): s = D(Q) / (D(Q) - D(O)) as a 16-bit fraction */
static void edge_weights(int32_t X, int32_t DQ, uint16_t *sf, uint16_t *tf)
{
    int16_t Xi = hle_fx_i(X);
    /* a rough |1/X| as an unsigned fraction cf: the doubled reciprocal, negated for negative X; 0xffff when it does not fit */
    uint32_t R = hle_rcp32(X);
    uint16_t Rf = (uint16_t)R;
    int16_t Ri = (int16_t)(R >> 16);
    hle_acc a2 = hle_wrap48((hle_acc)Rf * 2 + (((hle_acc)Ri * 2) << 16));
    uint16_t v3 = (uint16_t)((hle_acc)Rf * 2);
    int16_t v7 = hle_sclamp(a2);
    if (FMUT(54, Xi < 0, 0)) {
        hle_acc a3 = hle_wrap48(-(hle_acc)v3 - ((hle_acc)v7 << 16));
        v3 = (uint16_t)(-(hle_acc)v3);
        v7 = hle_sclamp(a3);
    }
    uint16_t cf = v7 == 0 ? v3 : FMUT(49, 0xffff, 0x7fff);
    COV(f3dex_clip_cov.cf_sat += v7 != 0; f3dex_clip_cov.x_zero += X == 0; f3dex_clip_cov.xneg += Xi < 0);
    /* X' = X * cf and its refined reciprocal */
    hle_acc ax = hle_mac_32xu16(X, cf);
    struct hle_pair Inv = f3dex_recip_refine(hle_pair32(hle_pair_acc(ax)));
    /* T = D(Q) * Inv (the fraction read before the high term: lo3), then s = 3 + T * cf: the 3 is the rounding bias of the microcode's VMUDL v28,v31,v31[13] */
    int32_t Inv32 = hle_pair32(Inv);
    uint16_t Tf = hle_uclamp(hle_mac_32x32_before_h(DQ, Inv32));
    int16_t Ti = hle_sclamp(hle_mac_32x32(DQ, Inv32));
    hle_acc as = hle_wrap48(FMUT(41, 3, 2) + hle_mac_32xu16(hle_fx(Ti, Tf), cf));
    int16_t Si = hle_sclamp(as);
    uint16_t Sf = hle_uclamp(as);
    uint16_t s = Si != 0 ? 0xffff : Sf;                                      /* inconsistent flags, guard-band planes, D(Q) < 0: saturated, nothing branches */
    COV(f3dex_clip_cov.sf_sat += Si != 0; f3dex_clip_cov.sf0 += s == 0; f3dex_clip_cov.sf1 += s == 1);
    if (s == 0)
        s = FMUT(42, 1, 2);
    *sf = s;
    *tf = (uint16_t)(0x10000 - s);
}

/* ---- 6.3 + 7 one new vertex -------------------------------------------------------------------------------------------------------------------------------------------------- */
/* the crossing of the edge from the outside vertex O to the inside vertex Q with the plane c: N = O + t (Q - O) */
static void new_vertex(const struct f3dex_vctx *ctx, const int16_t c[4], const struct f3dex_vtx *O, const struct f3dex_vtx *Q, struct f3dex_vtx *nv)
{
    int32_t X, DQ;
    uint16_t sf, tf;
    plane_values(c, O, Q, &X, &DQ);
    edge_weights(X, DQ, &sf, &tf);
    if (F3DEX_FX_MUTATE == 43) {                                             /* mutation 43: the weights go to the other vertex */
        uint16_t t = sf;
        sf = tf;
        tf = t;
    }
    /* the clip position: floor(Q * tf / 65536) + floor(O * sf / 65536), each term floored separately */
    int16_t Ni[4];
    uint16_t Nf[4];
    for (int l = 0; l < 4; l++) {
        hle_acc a = hle_wrap48(hle_mac_32xu16(hle_fx(Q->cx[l], Q->cf[l]), tf) + hle_mac_32xu16(hle_fx(O->cx[l], O->cf[l]), sf));
        Ni[l] = hle_sclamp(a);
        Nf[l] = hle_uclamp(a);
    }
    /* the colour bytes (<< 7) and the texture coordinates; (O * sf + Q * tf) >> 16, not scaled again */
    uint8_t col[4];
    for (int k = 0; k < 4; k++) {
        hle_acc a = hle_wrap48((hle_acc)(O->col[k] << 7) * sf + (hle_acc)(Q->col[k] << 7) * tf + FMUT(50, 0, 0x8000));
        col[k] = (uint8_t)(hle_sclamp(a) >> 7);
    }
    int16_t s = hle_sclamp(hle_wrap48((hle_acc)O->s * sf + (hle_acc)Q->s * tf));
    int16_t t = hle_sclamp(hle_wrap48((hle_acc)O->t * sf + (hle_acc)Q->t * tf));
    if (F3DEX_FX_MUTATE == 51) {                                             /* mutation 51: scaled by G_TEXTURE's scale a second time */
        s = hle_sclamp((hle_acc)s * ctx->tscale[0]);
        t = hle_sclamp((hle_acc)t * ctx->tscale[1]);
    }
    /* the flags (3, 7.2) from the new position; the vertex tail: 1/w, projection, viewport; fog */
    struct f3dex_flagparts fp;
    f3dex_vtx_flagparts(ctx, Ni, Nf, &fp);
    if (F3DEX_FX_MUTATE == 55 && ctx->b->non) {                              /* mutation 55: F3DEX.NoN flags with the z-low bit of the other builds */
        struct f3dex_build bb = *ctx->b;
        struct f3dex_vctx cc = *ctx;
        bb.non = 0;
        cc.b = &bb;
        f3dex_vtx_flagparts(&cc, Ni, Nf, &fp);
    }
    nv->flags = (uint16_t)FMUT(44, fp.f2 | fp.f1 << 4, fp.f1 | fp.f2 << 4);
    struct f3dex_tail_out to;
    f3dex_vtx_tail(ctx, Ni, Ni[3], Nf, nv, &to);
    if (ctx->b->dlx)
        nv->sy = (int16_t)FMUT(45, nv->sy, nv->sy + 4);                       /* mutation 45: F3DLX screen y off by a step */
    for (int l = 0; l < 4; l++) {
        nv->cx[l] = Ni[l];
        nv->cf[l] = Nf[l];
    }
    memcpy(nv->col, col, 4);
    nv->s = s;
    nv->t = t;
    if ((ctx->geom & F3DEX_G_FOG) && F3DEX_FX_MUTATE != 52)
        nv->col[3] = f3dex_fog_byte(ctx, to.qzi, to.qzf);
}

/* ---- 5 the fan ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* the triangles ($1, $2, $3) = (P[k+2], P[k+1], P[0]) of the polygon P[0 .. n); sets ntri and tri only. Not part of f3dex_fx.h: the spec adapter (fan re-entry step) uses it. */
void f3dex_clip_fan_scalar(const struct f3dex_vtx *const *P, unsigned n, struct f3dex_clip_out *o);
void f3dex_clip_fan_scalar(const struct f3dex_vtx *const *P, unsigned n, struct f3dex_clip_out *o)
{
    o->ntri = 0;
    for (unsigned k = 0; k + 2 < n && o->ntri < 7; k++) {
        if (F3DEX_FX_MUTATE == 46) {                                         /* mutation 46: the orientation flipped */
            o->tri[o->ntri][0] = P[0];
            o->tri[o->ntri][1] = P[k + 1];
            o->tri[o->ntri][2] = P[k + 2];
        } else {
            o->tri[o->ntri][0] = P[k + 2];
            o->tri[o->ntri][1] = P[k + 1];
            o->tri[o->ntri][2] = P[0];
        }
        o->ntri++;
    }
}

/* ---- 5 control flow ----------------------------------------------------------------------------------------------------------------------------------------------------------- */
void f3dex_clip_tri_scalar(const struct f3dex_vctx *c, const struct f3dex_vtx *const v[3], struct f3dex_clip_out *o)
{
    const struct f3dex_vtx *L[16], *M[16];
    unsigned n = 3;
    L[0] = v[2];                                                             /* the list starts reversed: v3, v2, v1 */
    L[1] = v[1];
    L[2] = v[0];
    o->ntri = 0;
    o->nnew = 0;
    COV(cov_reg(); f3dex_clip_cov.calls++);
    for (int pi = 0; pi < 6; pi++) {                                         /* near, far, y+, x+, y-, x- (4) */
        const int p = FMUT(40, pi, 5 - pi);                                  /* mutation 40: the planes in reverse order */
        const uint16_t mask = c->plane_mask[p];
        int outside[16], nout = 0, first = -1, last = -1;
        for (unsigned i = 0; i < n; i++) {                                   /* side(v) = flags(v) & mask, 0 = inside */
            outside[i] = (L[i]->flags & mask) != 0;
            nout += outside[i];
            if (outside[i]) {
                if (first < 0) first = (int)i;
                last = (int)i;
            }
        }
        if (nout == 0) {                                                     /* 2. the plane is a no-op */
            COV(f3dex_clip_cov.plane_noop[pi]++);
            continue;
        }
        if (nout == (int)n) {                                                /* 1. no vertex inside: the polygon is discarded (the new vertices made so far stay counted) */
            COV(f3dex_clip_cov.plane_disc[pi]++; f3dex_clip_cov.nnew[o->nnew]++);
            return;
        }
        /* 3. o = the first outside vertex, q = the first inside one after it (cyclically), r = the first outside one after q */
        unsigned oi = (unsigned)FMUT(48, first, last);                       /* mutation 48: the last outside vertex instead of the first */
        unsigned q = (oi + 1) % n;
        while (outside[q]) q = (q + 1) % n;
        unsigned r = (q + 1) % n;
        while (!outside[r]) r = (r + 1) % n;
        unsigned m = 0;
        /* N1 on the edge (outside L[q-1], inside L[q]); the run of inside vertices L[q .. r-1]; N2 on the edge (outside L[r], inside L[r-1]) */
        new_vertex(c, c->plane[p], L[(q + n - 1) % n], L[q], &o->nv[o->nnew]);
        M[m++] = &o->nv[o->nnew++];
        for (unsigned i = q; i != r; i = (i + 1) % n)
            M[m++] = L[i];
        new_vertex(c, c->plane[p], L[r], L[FMUT(47, (r + n - 1) % n, (r + 1) % n)], &o->nv[o->nnew]);
        M[m++] = &o->nv[o->nnew++];
        COV(f3dex_clip_cov.plane_new[pi]++);
        memcpy(L, M, m * sizeof(L[0]));
        n = m;
    }
    COV(f3dex_clip_cov.nnew[o->nnew]++; f3dex_clip_cov.npoly[n]++);
    f3dex_clip_fan_scalar(L, n, o);
}

void f3dex_clip_tri(const struct f3dex_vctx *c, const struct f3dex_vtx *const v[3], struct f3dex_clip_out *o) { f3dex_clip_tri_scalar(c, v, o); }
