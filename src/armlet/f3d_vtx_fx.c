/*
 * arMLEt - f3d_vtx_fx.c: the F3D vertex pipeline (scalar reference) and the matrix multiply
 *
 * One vertex at a time, as lane arithmetic on its four lanes (x y z w); docs/ucode-f3d.md 6 and 7.1. The microcode computes two vertices at once; the lanes are independent.
 * The sums are hle_fx.h's closed forms of the multiply-accumulate chains. The places where the microcode reads the low word of a sum before the last product is added
 * ("lo3") are marked.
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

/* ---- clip flags: VCH on the high words, VCL on the low words, of one lane against W; bit 0 = "X <= -W", bit 1 = "X >= W" ---------------------------------------------------------- */
static inline unsigned f3d_vch_vcl(uint16_t xhi, uint16_t xlo, uint16_t whi, uint16_t wlo)
{
    int16_t vs = (int16_t)xhi, vt = (int16_t)whi;
    int sign = (vs ^ vt) < 0;
    int16_t negvt = sign ? (int16_t)(-vt) : vt;
    int16_t diff = (int16_t)(vs - negvt);
    int diff_zero = diff == 0, diff_gt = diff > 0, vt_neg = vt < 0;
    int ge = sign ? vt_neg : (diff_gt || diff_zero);
    int le = sign ? !diff_gt : vt_neg;
    int vce = sign && diff == -1;
    int ne = !(diff_zero || vce);
    uint16_t sneg = sign ? (uint16_t)(0 - wlo) : wlo;
    uint16_t ldiff = (uint16_t)(xlo - sneg);
    uint32_t sum = (uint32_t)xlo + wlo;
    int ncarry = ldiff == (sum > 0xffff ? 0xffff : sum);
    int le_eq = (ldiff == 0 && ncarry && !vce) || (vce && (ldiff == 0 || ncarry));
    int ge_eq = !(wlo > xlo);
    if (!ne && sign) le = le_eq;
    if (!(sign || ne)) ge = ge_eq;
    return (unsigned)le | ((unsigned)ge << 1);
}

/* the clip flag word of one vertex from its clip coordinates (hi / lo) and the K-scaled W */
static uint16_t f3d_clip_flag_word(const uint16_t hi[4], const uint16_t lo[4], uint16_t whi2, uint16_t wlo2)
{
    unsigned f = 0;
    for (int k = 0; k < 3; k++) {
        unsigned f1 = f3d_vch_vcl(hi[k], lo[k], hi[3], lo[3]);
        unsigned f2 = f3d_vch_vcl(hi[k], lo[k], whi2, wlo2);
        f |= (f2 & 1) << k | (f1 & 1) << (4 + k) | ((f2 >> 1) & 1) << (8 + k) | ((f1 >> 1) & 1) << (12 + k);
    }
    return (uint16_t)f;
}

/* ---- the part of the pipeline after the clip coordinates: shared with the clipper (new vertices go through the same tail) ------------------------------------------------------------ */
/* clip coordinates hi / lo (4 lanes) -> flags, 1/W, screen coordinates (x, y, z, zfrac), reciprocal; fills o->cx .. o->flags except the colour and texture coordinates */
void f3d_vtx_tail(const struct f3d_vctx *c, const uint16_t hi[4], const uint16_t lo[4], struct f3d_vtx *o, uint16_t shi_out[4], uint16_t slo_out[4])
{
    int16_t vscale[4];
    for (int k = 0; k < 4; k++)
        vscale[k] = hle_sclamp(((hle_acc)c->vscale[k] * (F3D_FX_MUTATE == 6 ? 1 : c->vsign[k])) << 16);   /* mutation 6: the sign vector ignored */
    /* K-scaled copy of the clip coordinates (flags only) */
    uint16_t hi2w, lo2w;
    {
        hle_acc a = hle_wrap48((hle_acc)lo[3] * c->clip_k);                 /* VMUDN: u16 x s16 */
        lo2w = hle_uclamp(a);
        a = hle_wrap48(a + (((hle_acc)(int16_t)hi[3] * c->clip_k) << 16));  /* VMADH */
        hi2w = hle_sclamp(a);
    }
    uint16_t flags = F3D_FX_MUTATE == 5 ? f3d_clip_flag_word(hi, lo, hi[3], lo[3]) : f3d_clip_flag_word(hi, lo, hi2w, lo2w);   /* mutation 5: no K scaling in the flags */
    /* the perspective-normalised W, its reciprocal */
    int32_t W = (int32_t)((uint32_t)hi[3] << 16 | lo[3]);
    struct hle_pair n = { 0, 0 };
    {
        hle_acc a = hle_mac_32xu16(W, c->persp);                           /* VMUDL VMADM */
        n.i = hle_sclamp(a);
        n.f = hle_uclamp(a);
    }
    struct hle_pair R = f3d_recip_refine(hle_pair32(n));
    uint16_t rhc = R.i >= 0 ? (uint16_t)R.i : 0x7fff;
    int32_t Rv = (int32_t)((uint32_t)rhc << 16 | R.f);
    int16_t sc[4];
    uint16_t szf = 0;
    for (int k = 0; k < 4; k++) {
        int32_t X = (int32_t)((uint32_t)hi[k] << 16 | lo[k]);
        if (F3D_FX_MUTATE == 3) {                                          /* mutation 3: the K-scaled copy is projected (the first draft's error) */
            hle_acc t = hle_wrap48((hle_acc)lo[k] * c->clip_k);
            uint16_t l2 = hle_uclamp(t);
            t = hle_wrap48(t + (((hle_acc)(int16_t)hi[k] * c->clip_k) << 16));
            X = (int32_t)((uint32_t)hle_sclamp(t) << 16 | l2);
        }
        hle_acc a = hle_mac_32x32(X, Rv);                                  /* VMUDL VMADM VMADN VMADH */
        uint16_t slo = FMUT(4, hle_uclamp(hle_mac_32x32_before_h(X, Rv)), hle_uclamp(a));   /* lo3; mutation 4: the full sum */
        uint16_t shi = hle_sclamp(a);
        /* times the perspective normalisation */
        hle_acc b = hle_mac_32xu16((int32_t)((uint32_t)shi << 16 | slo), c->persp);
        shi = hle_sclamp(b);
        slo = hle_uclamp(b);
        if (shi_out) shi_out[k] = shi;
        if (slo_out) slo_out[k] = slo;
        /* viewport: trans << 16 + scale * (shi:slo) */
        hle_acc v = hle_wrap48(((hle_acc)c->vtrans[k] << 16) + (hle_acc)(int32_t)((uint32_t)shi << 16 | slo) * vscale[k]);
        int16_t s = (int16_t)hle_sclamp(v);
        int16_t lo_lim = k < 2 ? c->lim[1] : c->lim[3], hi_lim = k < 2 ? c->lim[0] : c->lim[2];
        if (s < lo_lim) s = lo_lim;
        if (s > hi_lim) s = hi_lim;
        sc[k] = s;
        if (k == 2) szf = hle_uclamp(v);
    }
    for (int k = 0; k < 4; k++) {
        o->cx[k] = (int16_t)hi[k];
        o->cf[k] = lo[k];
    }
    o->sx = sc[0];
    o->sy = sc[1];
    o->sz = sc[2];
    o->szf = szf;
    o->rwi = R.i;
    o->rwf = R.f;
    o->flags = flags;
}

/* the fog value from the normalised z (shi:slo of lane 2) */
uint8_t f3d_fog_byte(const struct f3d_vctx *c, uint16_t shi2, uint16_t slo2)
{
    if (c->fog_wide) {
        /* z (32 bits) x the 32-bit multiplier (VMUDL VMADM VMADN VMADH: the low word is read before the last product), + the 32-bit offset (VADDC VADD), x the 16-bit factor (VMUDN VMADH), then
         * the integer half limited to [0, factor] */
        int32_t z = (int32_t)((uint32_t)shi2 << 16 | slo2), m = hle_fx(c->fogw[0], (uint16_t)c->fogw[1]);
        uint16_t lo = hle_uclamp(hle_mac_32x32_before_h(z, m));
        int16_t hi = (int16_t)hle_sclamp(hle_mac_32x32(z, m));
        struct hle_pair s = hle_add32(hle_fx(hi, lo), FMUT(31, hle_fx(c->fogw[2], F3D_FX_MUTATE == 43 ? 0 : (uint16_t)c->fogw[3]), 0));    /* mutation 31: no offset */
        int16_t v = (int16_t)hle_sclamp(hle_wrap48((hle_acc)(int32_t)hle_pair32(s) * (F3D_FX_MUTATE == 46 ? 1 : c->fogw[4])));
        if (v < 0 && F3D_FX_MUTATE != 44) v = 0;                 /* mutation 44: no lower limit */
        if (v >= c->fogw[4] && F3D_FX_MUTATE != 45) v = c->fogw[4];   /* mutation 45: no upper limit */
        return (uint8_t)v;
    }
    hle_acc a = hle_wrap48((hle_acc)(int32_t)((uint32_t)shi2 << 16 | slo2) * c->fog[0]);   /* VMUDN VMADH */
    int16_t v = (int16_t)hle_sclamp(a);
    int32_t w = (int32_t)v + c->fog[1];
    v = hle_clamp16(w);
    if (v < 0) v = 0;
    if (v > c->fog[2]) v = c->fog[2];
    return (uint8_t)FMUT(1, v, v + 1);
}

/* ---- one vertex ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */
static inline int16_t sadd16(int16_t a, int16_t b) { return hle_clamp16((int32_t)a + b); }
static inline int16_t vmulf16(int16_t a, int16_t b) { return (int16_t)hle_sclamp(hle_mulf(a, b)); }

/* lighting and texture generation for vertex g (0 / 1: bytes 4g .. 4g+3 of the slot byte rows) */
static void light_vertex(const struct f3d_vctx *c, const struct f3d_vin *in, unsigned g, uint8_t col[4], int16_t st[2])
{
    int16_t nrm[3] = { (int16_t)((int8_t)in->c[0] << 8), (int16_t)((int8_t)in->c[1] << 8), (int16_t)((int8_t)in->c[2] << 8) };
    int16_t acc[3] = { 0, 0, 0 };
    int16_t cur[3];                                                         /* the previous iteration's colour products, lanes r g b */
    for (int k = 0; k < 3; k++)
        cur[k] = (int16_t)(c->lcol[c->nslot + 1][4 * g + k] << 7);          /* the ambient colour, accumulated first */
    int16_t Dprev = 0, D = 0;
    int16_t pre[3] = { 0, 0, 0 };                                           /* the colour bytes << 7 of the last slot */
    for (int s = (int)c->nslot; s >= 0; s--) {
        for (int k = 0; k < 3; k++) {
            if (cur[k] < 0) cur[k] = 0;                                     /* VGE with 0 */
            acc[k] = sadd16(acc[k], cur[k]);
        }
        int16_t d[3];
        for (int k = 0; k < 3; k++)
            d[k] = vmulf16(nrm[k], (int16_t)(c->ldir[s][4 * g + k] << 8));
        Dprev = D;
        D = sadd16(sadd16(d[0], d[1]), d[2]);
        for (int k = 0; k < 3; k++) {
            pre[k] = (int16_t)(c->lcol[s][4 * g + k] << 7);
            cur[k] = vmulf16(pre[k], D);
        }
    }
    for (int k = 0; k < 3; k++)
        col[k] = (uint8_t)(acc[k] >> 7);
    if (c->geom & F3D_G_TEXGEN) {
        for (int j = 0; j < 2; j++) {
            /* the accumulator holds the last slot's product for lane j (colour byte j of the look-at Y slot, here the colour bytes row), plus the look-at X product */
            int16_t colbyte = (int16_t)(c->lcol[0][4 * g + j] << 7);
            hle_acc a = hle_wrap48(2 * (hle_acc)colbyte * D + 0x8000);
            a = hle_wrap48(a + (hle_acc)(uint16_t)(int16_t)(c->lk[4 * g + j] << 8) * Dprev);
            int16_t v = (int16_t)hle_sclamp(a);
            if (c->geom & F3D_G_TEXGEN_LINEAR) {
                int16_t x = vmulf16(v, v);
                x = vmulf16(x, v);
                hle_acc a2 = 2 * (hle_acc)x * c->tg[0] + 0x8000;
                a2 += 2 * (hle_acc)x * c->tg[2];
                a2 += 2 * (hle_acc)v * c->tg[1];
                v = (int16_t)hle_sclamp(a2);
            }
            st[j] = hle_clamp16((int32_t)v + FMUT(2, 0x4000, 0x4001));
        }
    }
}

static void vtx_one(const struct f3d_vctx *c, const struct f3d_vin *in, unsigned g, struct f3d_vtx *o)
{
    uint16_t hi[4], lo[4];
    const int16_t px[3] = { in->x, in->y, in->z };
    /* clip coordinates: rows 0..2 times x, y, z plus the fraction of row 3 (N and H chains); the low word is read before the integer part of row 3 is added */
    for (int k = 0; k < 4; k++) {
        hle_acc acc = 0;
        for (int j = 0; j < 3; j++)
            acc += (hle_acc)c->mp.f[j][k] * px[j] + (((hle_acc)c->mp.i[j][k] * px[j]) << 16);
        acc = hle_wrap48(acc + c->mp.f[3][k]);
        lo[k] = hle_uclamp(acc);
        hi[k] = hle_sclamp(hle_wrap48(acc + ((hle_acc)c->mp.i[3][k] << 16)));
        if (c->has_off) {                                                   /* Rare: VADDC + VADD of the offset vector */
            struct hle_pair s = hle_add32(hle_fx((int16_t)hi[k], lo[k]), hle_fx(c->off_i[k], c->off_f[k]));
            hi[k] = (uint16_t)(F3D_FX_MUTATE == 29 ? hle_clamp16((int16_t)hi[k] + c->off_i[k]) : s.i);   /* mutation 29: the carry of the low words dropped */
            lo[k] = s.f;
        }
    }
    uint16_t shi[4], slo[4];
    f3d_vtx_tail(c, hi, lo, o, shi, slo);
    int16_t st[2] = { in->s, in->t };
    memcpy(o->col, in->c, 4);
    if (c->geom & F3D_G_LIGHTING)
        light_vertex(c, in, g, o->col, st);
    o->s = (int16_t)hle_sclamp((hle_acc)st[0] * c->tscale[0]);
    o->t = (int16_t)hle_sclamp((hle_acc)st[1] * c->tscale[1]);
    if (c->geom & F3D_G_FOG)
        o->col[3] = f3d_fog_byte(c, shi[2], slo[2]);
}

/* Pilotwings' flag word, from the SM64 one (docs/ucode-f3d.md 10.5.1): the pairs are (0,1), (2,3), ... of the command; a last odd vertex is a first of a pair */
static void pw_flags(struct f3d_vtx *out, unsigned n)
{
    for (unsigned i = 0; i < n; i += 2) {
        unsigned sa = out[i].cx[3] < 0;
        out[i].flags &= (uint16_t)~0x40;
        if (i + 1 < n) {
            struct f3d_vtx *b = &out[i + 1];
            b->flags = (uint16_t)((b->flags & ~0x40) | (b->cx[3] < 0 && F3D_FX_MUTATE != 21 ? 0x40 : 0) | (sa && F3D_FX_MUTATE != 20 ? 0x4 : 0));   /* mutations 20, 21: a bit left out */
        }
    }
}

int f3d_vtx_xform_scalar(const struct f3d_vctx *c, const struct f3d_vin *in, unsigned n, struct f3d_vtx *out)
{
    if ((c->geom & F3D_G_LIGHTING) && (c->nslot < 1 || c->nslot > F3D_MAX_SLOTS))
        return 1;
    for (unsigned i = 0; i < n; i++)
        vtx_one(c, &in[i], FMUT(7, i & 1, 0), &out[i]);                                  /* mutation 7: the second vertex of a pair uses the first one's light bytes */
    if (c->flags_variant == F3D_FLAGS_PW)
        pw_flags(out, n);
    return 0;
}

/* ---- the matrix multiply ----------------------------------------------------------------------------------------------------------------------------------------------------- */
void f3d_mtx_mul_scalar(const struct f3d_mtx *a, const struct f3d_mtx *b, struct f3d_mtx *out)
{
    struct f3d_mtx r;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            hle_acc acc = 0;
            for (int k = 0; k < 4; k++) {
                acc += (hle_acc)(((uint32_t)b->f[k][j] * a->f[i][k] + FMUT(8, 0, 0x8000)) >> 16);   /* VMUDL: u16 x u16 >> 16; mutation 8: rounded */
                acc += (hle_acc)b->i[k][j] * a->f[i][k];                                  /* VMADM: s16 x u16 */
                acc += (hle_acc)b->f[k][j] * a->i[i][k];                                  /* VMADN: u16 x s16 */
                acc += ((hle_acc)b->i[k][j] * a->i[i][k]) << 16;                          /* VMADH: s16 x s16 << 16 */
            }
            acc = hle_wrap48(acc);
            r.i[i][j] = hle_sclamp(acc);
            r.f[i][j] = hle_uclamp(acc);
        }
    *out = r;
}

#ifdef __ARM_NEON
#include "f3d_vtx_fx_neon_kernels.h"

/* ---- NEON: two vertices at a time ---------------------------------------------------------------------------------------------------------------------------------------- */
static void neon_prepare(const struct f3d_vctx *c, struct vtx_neon_ctx *n)
{
    int32x4_t m[3];
    for (int j = 0; j < 3; j++)
        m[j] = vreinterpretq_s32_u32(vorrq_u32(vshll_n_u16(vld1_u16((const uint16_t *)c->mp.i[j]), 16), vmovl_u16(vld1_u16(c->mp.f[j]))));
    n->m[0] = m[0]; n->m[1] = m[1]; n->m[2] = m[2];
    uint32x4_t mf3 = vmovl_u16(vld1_u16(c->mp.f[3]));
    int32x4_t mi3 = vmovl_s16(vld1_s16(c->mp.i[3]));
    n->cpart[0] = vreinterpretq_s64_u64(vmovl_u32(vget_low_u32(mf3)));
    n->cpart[1] = vreinterpretq_s64_u64(vmovl_u32(vget_high_u32(mf3)));
    n->cint[0] = vshll_n_s32(vget_low_s32(mi3), 16);
    n->cint[1] = vshll_n_s32(vget_high_s32(mi3), 16);
    int16x4_t vs4 = vld1_s16(c->vscale), sg4 = vld1_s16(c->vsign);
    n->vscale = vcombine_s16(vqmovn_s32(vmull_s16(vs4, sg4)), vqmovn_s32(vmull_s16(vs4, sg4)));
    int16x4_t vt4 = vld1_s16(c->vtrans);
    n->vtrans = vcombine_s16(vt4, vt4);
    int16_t l0 = c->lim[0], l1 = c->lim[1], l2 = c->lim[2], l3 = c->lim[3];
    int16_t ll[8] = { l1, l1, l3, l3, l1, l1, l3, l3 }, lh[8] = { l0, l0, l2, l2, l0, l0, l2, l2 };
    n->lim_lo = vld1q_s16(ll);
    n->lim_hi = vld1q_s16(lh);
    n->persp = c->persp;
    int32x4_t big = vdupq_n_s32(0);
    for (int j = 0; j < 3; j++)
        big = vorrq_s32(big, vabsq_s32(vqabsq_s32(m[j])));
    n->wrap = vmaxvq_u32(vshrq_n_u32(vreinterpretq_u32_s32(big), 28)) != 0;
    n->ok32 = 1;
    int16x4_t lim16 = vdup_n_s16(16384);
    for (int j = 0; j < 4; j++) {
        uint16x4_t mi = vld1_u16((const uint16_t *)c->mp.i[j]), mf = vld1_u16(c->mp.f[j]);
        int16x4_t brw = vreinterpret_s16_u16(vshr_n_u16(mf, 15));
        int16x4_t mip = vadd_s16(vreinterpret_s16_u16(mi), brw);
        int16x4_t mfs = vreinterpret_s16_u16(mf);
        if (j < 3) {
            n->mip[j] = mip;
            n->mfs[j] = mfs;
            uint16x4_t big16 = vcgt_s16(vabs_s16(mip), lim16);
            uint16x4_t wrapped = vclt_s16(mip, vreinterpret_s16_u16(mi));
            n->ok32 &= vmaxv_u16(vorr_u16(big16, wrapped)) == 0;
        } else {
            n->h3 = vmovl_s16(brw);
            n->l3 = vmovl_s16(mfs);
            n->mi3 = vmovl_s16(vreinterpret_s16_u16(mi));
        }
    }
}

#define BC8(v, a, b) vcombine_s16(vdup_n_s16(vgetq_lane_s16(v, a)), vdup_n_s16(vgetq_lane_s16(v, b)))

struct f3d_vpre {
    struct vtx_neon_ctx n;
    int lit;
    int has_off;                            /* Rare: the offset vector, both halves */
    int16x8_t off_hi;
    uint16x8_t off_lo;
    int16x8_t lcol_amb;                     /* the ambient colour bytes << 7 */
    int16x8_t ldir[F3D_MAX_SLOTS + 1], lcol[F3D_MAX_SLOTS + 1];     /* slot nslot - k at index k */
};

static inline int16x8_t bytes_shl(const uint8_t *p, int sh) { return vreinterpretq_s16_u16(sh == 7 ? vshll_n_u8(vld1_u8(p), 7) : vshll_n_u8(vld1_u8(p), 8)); }

static void neon_prepare_all(const struct f3d_vctx *c, struct f3d_vpre *p)
{
    neon_prepare(c, &p->n);
    p->lit = (c->geom & F3D_G_LIGHTING) != 0;
    p->has_off = c->has_off;
    if (c->has_off) {
        p->off_hi = vcombine_s16(vld1_s16(c->off_i), vld1_s16(c->off_i));
        p->off_lo = vcombine_u16(vld1_u16(c->off_f), vld1_u16(c->off_f));
    }
    if (p->lit) {
        p->lcol_amb = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(c->lcol[c->nslot + 1]), 7));
        for (unsigned k = 0; k <= c->nslot; k++) {
            unsigned s = c->nslot - k;
            p->ldir[k] = vshll_n_s8(vld1_s8(c->ldir[s]), 8);
            p->lcol[k] = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(c->lcol[s]), 7));
        }
    }
}

/* two vertices; a[1] may be a copy of a[0] when only one is wanted (out[1] is then not written) */
static void neon_pair(const struct f3d_vctx *c, const struct f3d_vpre *P, const struct f3d_vin in[2], int two, struct f3d_vtx out[2])
{
    const struct vtx_neon_ctx *n = &P->n;
    int16_t px[2][3];
    int16_t st[2][2];
    uint8_t col[2][4];
    for (int g = 0; g < 2; g++) {
        px[g][0] = in[g].x; px[g][1] = in[g].y; px[g][2] = in[g].z;
        st[g][0] = in[g].s; st[g][1] = in[g].t;
        memcpy(col[g], in[g].c, 4);
    }
    if (P->lit) {
        uint8_t nb[8];
        memcpy(nb, in[0].c, 4); memcpy(nb + 4, in[1].c, 4);
        int16x8_t nrm = vshll_n_s8(vld1_s8((const int8_t *)nb), 8);
        const int16x8_t zero16 = vdupq_n_s16(0);
        int16x8_t acc27 = zero16, v7 = P->lcol_amb, v6b = zero16, v7pre = zero16;
        int16_t D0[2] = { 0, 0 }, D1[2] = { 0, 0 };
        for (unsigned k = 0; k <= c->nslot; k++) {
            D1[0] = D0[0]; D1[1] = D0[1];
            v7 = vmaxq_s16(v7, zero16);
            acc27 = vqaddq_s16(acc27, v7);
            v7pre = P->lcol[k];
            int16x8_t v6 = vqrdmulhq_s16(nrm, P->ldir[k]);
            int16x8_t v3 = vqaddq_s16(v6, vtrn2q_s16(v6, v6));
            v6b = vqaddq_s16(v3, BC8(v6, 2, 6));
            v7 = vqrdmulhq_s16(v7pre, BC8(v6b, 0, 4));
            D0[0] = vgetq_lane_s16(v6b, 0);
            D0[1] = vgetq_lane_s16(v6b, 4);
        }
        if (c->geom & F3D_G_TEXGEN) {
            int16_t pre[8], v18s[8];
            vst1q_s16(pre, v7pre);
            for (int i = 0; i < 8; i++) {
                int g = i >> 2;
                hle_acc la = 2 * (hle_acc)pre[i] * D0[g] + 0x8000;
                hle_acc a = hle_wrap48(la + (hle_acc)(uint16_t)(int16_t)(c->lk[i] << 8) * D1[g]);
                int16_t v18 = (int16_t)hle_sclamp(a);
                if (c->geom & F3D_G_TEXGEN_LINEAR) {
                    int16_t x = vmulf16(v18, v18);
                    x = vmulf16(x, v18);
                    hle_acc a2 = 2 * (hle_acc)x * c->tg[0] + 0x8000;
                    a2 += 2 * (hle_acc)x * c->tg[2];
                    a2 += 2 * (hle_acc)v18 * c->tg[1];
                    v18 = (int16_t)hle_sclamp(a2);
                }
                v18s[i] = hle_clamp16((int32_t)v18 + FMUT(2, 0x4000, 0x4001));
            }
            st[0][0] = v18s[0]; st[0][1] = v18s[1]; st[1][0] = v18s[4]; st[1][1] = v18s[5];
        }
        int8x8_t rgb = vshrn_n_s16(acc27, 7);
        uint8_t rb[8];
        vst1_s8((int8_t *)rb, rgb);
        for (int g = 0; g < 2; g++) { col[g][0] = rb[4 * g]; col[g][1] = rb[4 * g + 1]; col[g][2] = rb[4 * g + 2]; }
    }
    uint16x8_t hi, lo;
    unsigned vcc;
    {
        int big = 0;
        for (int g = 0; g < 2; g++)
            for (int k = 0; k < 3; k++) {
                int16_t v = px[g][k];
                int16_t av = v < 0 ? (int16_t)-v : v;
                if (av < 0) av = 32767;
                big |= av > 16384;
            }
        if (n->ok32 && !big) neon_transform32(n, px, &hi, &lo, &vcc);
        else neon_transform(n, px, &hi, &lo, &vcc);
    }
    if (P->has_off) {                       /* VADDC + VADD of the offset: the low words wrap and carry, the high words saturate; the flags are those of the sums */
        uint16x8_t lo2 = vaddq_u16(lo, P->off_lo);
        int16x8_t carry = vreinterpretq_s16_u16(vshrq_n_u16(vcltq_u16(lo2, lo), 15));
        if (F3D_FX_MUTATE == 29) carry = vdupq_n_s16(0);
        int32x4_t h0 = vaddq_s32(vaddl_s16(vget_low_s16(vreinterpretq_s16_u16(hi)), vget_low_s16(P->off_hi)), vmovl_s16(vget_low_s16(carry)));
        int32x4_t h1 = vaddq_s32(vaddl_s16(vget_high_s16(vreinterpretq_s16_u16(hi)), vget_high_s16(P->off_hi)), vmovl_s16(vget_high_s16(carry)));
        uint16x8_t whi, wlo, le, ge;
        static const uint16_t wt[8] = {1, 2, 4, 8, 16, 32, 64, 128};
        hi = vreinterpretq_u16_s16(vcombine_s16(vqmovn_s32(h0), vqmovn_s32(h1)));
        lo = lo2;
        neon_wlanes(hi, &whi);
        neon_wlanes(lo, &wlo);
        neon_clip_flags(hi, lo, whi, wlo, &le, &ge);
        if (F3D_FX_MUTATE != 30)                           /* mutation 30: the flags of the sums not recomputed (NEON form) */
            vcc = (unsigned)vaddvq_u16(vandq_u16(le, vld1q_u16(wt))) | ((unsigned)vaddvq_u16(vandq_u16(ge, vld1q_u16(wt))) << 8);
    }
    uint16_t hia[8], loa[8];
    vst1q_u16(hia, hi); vst1q_u16(loa, lo);
    /* flags: the exact group from the NEON compare, the K-scaled group the same when K = 1 */
    unsigned vcc2 = vcc;
    if (c->clip_k != 1) {
        vcc2 = 0;
        for (int g = 0; g < 2; g++) {
            hle_acc a = hle_wrap48((hle_acc)loa[4 * g + 3] * c->clip_k);
            uint16_t l2 = hle_uclamp(a);
            a = hle_wrap48(a + (((hle_acc)(int16_t)hia[4 * g + 3] * c->clip_k) << 16));
            uint16_t h2 = hle_sclamp(a);
            for (int k = 0; k < 3; k++) {
                unsigned f2 = f3d_vch_vcl(hia[4 * g + k], loa[4 * g + k], h2, l2);
                vcc2 |= (f2 & 1) << (4 * g + k) | ((f2 >> 1) & 1) << (8 + 4 * g + k);
            }
        }
    }
    uint16_t rhc[2], rhi[2], rlo[2];
    for (int g = 0; g < 2; g++) {
        int32_t h = (int16_t)hia[4 * g + 3];
        uint32_t l = loa[4 * g + 3];
        int32_t w = h * (int32_t)n->persp + (int32_t)((l * n->persp) >> 16);
        struct hle_pair R = f3d_recip_refine(w);
        rhi[g] = (uint16_t)R.i; rlo[g] = R.f;
        rhc[g] = R.i >= 0 ? (uint16_t)R.i : 0x7fff;
    }
    struct vtx_neon_out no;
    neon_screen(n, hi, lo, rhc, rlo, &no);
    unsigned t0 = ((vcc & 1799) << 4) | ((vcc & 28784) << 16) | (vcc2 & 1799) | ((vcc2 & 28784) << 12);
    uint16_t scr[8], vlo[8], shi[8], slo[8];
    vst1q_s16((int16_t *)scr, no.scr); vst1q_u16(vlo, no.vlo); vst1q_u16(shi, no.shi); vst1q_u16(slo, no.slo);
    for (int g = 0; g < (two ? 2 : 1); g++) {
        struct f3d_vtx *o = &out[g];
        for (int k = 0; k < 4; k++) { o->cx[k] = (int16_t)hia[4 * g + k]; o->cf[k] = loa[4 * g + k]; }
        memcpy(o->col, col[g], 4);
        o->s = (int16_t)hle_sclamp((hle_acc)st[g][0] * c->tscale[0]);
        o->t = (int16_t)hle_sclamp((hle_acc)st[g][1] * c->tscale[1]);
        o->sx = (int16_t)scr[4 * g]; o->sy = (int16_t)scr[4 * g + 1]; o->sz = (int16_t)scr[4 * g + 2];
        o->szf = vlo[4 * g + 2];
        o->rwi = (int16_t)rhi[g]; o->rwf = rlo[g];
        o->flags = (uint16_t)(t0 >> (16 * g));
        if (c->geom & F3D_G_FOG)
            o->col[3] = f3d_fog_byte(c, shi[4 * g + 2], slo[4 * g + 2]);
    }
}

int f3d_vtx_xform(const struct f3d_vctx *c, const struct f3d_vin *in, unsigned n, struct f3d_vtx *out)
{
    if ((c->geom & F3D_G_LIGHTING) && (c->nslot < 1 || c->nslot > F3D_MAX_SLOTS))
        return 1;
    struct f3d_vpre P;
    neon_prepare_all(c, &P);
    unsigned i = 0;
    for (; i + 1 < n; i += 2)
        neon_pair(c, &P, &in[i], 1, &out[i]);
    if (i < n) {
        struct f3d_vin two[2] = { in[i], in[i] };
        struct f3d_vtx o2[2];
        neon_pair(c, &P, two, 0, o2);
        out[i] = o2[0];
    }
    if (c->flags_variant == F3D_FLAGS_PW && F3D_FX_MUTATE != 23)   /* mutation 23: the NEON form leaves the SM64 flags */
        pw_flags(out, n);
    return 0;
}

/* the matrix product, 64-bit lanes: the four product forms per k */
void f3d_mtx_mul(const struct f3d_mtx *a, const struct f3d_mtx *b, struct f3d_mtx *out)
{
    struct f3d_mtx r;
    uint16x4_t Fb[4];
    int16x4_t Ib[4];
    for (int k = 0; k < 4; k++) { Fb[k] = vld1_u16(b->f[k]); Ib[k] = vld1_s16(b->i[k]); }
    for (int i = 0; i < 4; i++) {
        int64x2_t acc0 = vdupq_n_s64(0), acc1 = vdupq_n_s64(0);
        for (int k = 0; k < 4; k++) {
            const uint16_t fa = a->f[i][k];
            const int16_t ia = a->i[i][k];
            /* Fb * Fa >> 16 (u16 x u16) */
            uint32x4_t p1 = vshrq_n_u32(vmull_n_u16(Fb[k], fa), 16);
            /* Ib * Fa (s16 x u16): the signed product of Fa as s16 plus (Ib << 16) when Fa >= 0x8000 */
            int32x4_t p2 = vmull_n_s16(Ib[k], (int16_t)fa);
            int64x2_t p2l = vmovl_s32(vget_low_s32(p2)), p2h = vmovl_s32(vget_high_s32(p2));
            if (fa >= 0x8000 && F3D_FX_MUTATE != 19) {   /* mutation 19: the unsigned fraction treated as signed */
                int64x2_t sh_l = vshll_n_s32(vget_low_s32(vmovl_s16(Ib[k])), 16), sh_h = vshll_n_s32(vget_high_s32(vmovl_s16(Ib[k])), 16);
                p2l = vaddq_s64(p2l, sh_l);
                p2h = vaddq_s64(p2h, sh_h);
            }
            /* Fb * Ia (u16 x s16): the signed product of Fb as s16 plus (Ia << 16) where Fb >= 0x8000 */
            int32x4_t p3 = vmull_n_s16(vreinterpret_s16_u16(Fb[k]), ia);
            int64x2_t p3l = vmovl_s32(vget_low_s32(p3)), p3h = vmovl_s32(vget_high_s32(p3));
            uint32x4_t m3 = vmovl_u16(vshr_n_u16(Fb[k], 15));
            int64x2_t s3l = vmull_n_s32(vreinterpret_s32_u32(vget_low_u32(m3)), (int32_t)ia << 16), s3h = vmull_n_s32(vreinterpret_s32_u32(vget_high_u32(m3)), (int32_t)ia << 16);
            p3l = vaddq_s64(p3l, s3l);
            p3h = vaddq_s64(p3h, s3h);
            /* Ib * Ia << 16 */
            int32x4_t p4 = vmull_n_s16(Ib[k], ia);
            int64x2_t p4l = vshll_n_s32(vget_low_s32(p4), 16), p4h = vshll_n_s32(vget_high_s32(p4), 16);
            acc0 = vaddq_s64(acc0, vaddq_s64(vaddq_s64(vmovl_s32(vreinterpret_s32_u32(vget_low_u32(p1))), p2l), vaddq_s64(p3l, p4l)));
            acc1 = vaddq_s64(acc1, vaddq_s64(vaddq_s64(vmovl_s32(vreinterpret_s32_u32(vget_high_u32(p1))), p2h), vaddq_s64(p3h, p4h)));
        }
        int64_t t[4] = { vgetq_lane_s64(acc0, 0), vgetq_lane_s64(acc0, 1), vgetq_lane_s64(acc1, 0), vgetq_lane_s64(acc1, 1) };
        for (int j = 0; j < 4; j++) {
            hle_acc v = hle_wrap48(t[j]);
            r.i[i][j] = hle_sclamp(v);
            r.f[i][j] = hle_uclamp(v);
        }
    }
    *out = r;
}
#else
/* ---- the entry points without NEON: the scalar forms --------------------------------------------------------------------------------------------------------------------------- */
int f3d_vtx_xform(const struct f3d_vctx *c, const struct f3d_vin *in, unsigned n, struct f3d_vtx *out) { return f3d_vtx_xform_scalar(c, in, n, out); }
void f3d_mtx_mul(const struct f3d_mtx *a, const struct f3d_mtx *b, struct f3d_mtx *out) { f3d_mtx_mul_scalar(a, b, out); }
#endif
