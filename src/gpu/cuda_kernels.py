"""CUDA kernels for the device frame loop, compiled once per process. [Shrestha]

Every kernel here has a numpy reference elsewhere in the repository, and the
contract is BIT-IDENTICAL output, not "close": the map is hashed, and a device
map that differs by one cell is a different map. Three things make that true
rather than lucky, and each was measured before it was relied on:

⚑ `--fmad=false` ON EVERY KERNEL. NVRTC contracts `a*b + c` into a fused
  multiply-add by default. Measured on 1,000,000 random doubles, `x*0.1 + 0.7`
  differed from numpy on 289,150 of them with contraction and on 0 without.
  Every float expression below is written in the numpy reference's operation
  order, and this flag is what stops the compiler reordering it.

⚑ cupy's float `//` IS NOT numpy's. `1.0 // 0.1` is 9.0 in numpy and 10.0 in
  cupy: numpy computes floor division through `npy_divmod` (fmod first, then
  snap), cupy through `floor(a / b)`. 42,425 of 5,000,000 lattice-scale
  coordinates disagreed. `bin_points` floors every world coordinate onto the
  5 cm lattice this way, so `np_floor_divide` below is `npy_divmod`, line for
  line -- the integer lattice (math §2) is only one lattice if both sides
  agree on where x / 0.05 lands.

⚑ atan2 / asin: numpy's float32 results are CORRECTLY ROUNDED on this build
  (glibc 2.42), and a double-precision device result rounded to float32 matched
  them on all 3,709,039 points of 30 real seq 08 frames. Taking the float32
  device overloads directly instead put 116 of 24.5 M points in a different
  range-image pixel over 200 frames. So the projection computes in double and
  rounds -- and `scripts/gpu_parity.py` compares the whole range image every
  frame, so a disagreement cannot pass unnoticed.

The quantised-variance codec needs `log`, and device and host `log` are not
guaranteed to agree. Rather than hope, the codec is taken from the host
function itself: `variance_code_thresholds()` bisects Aakash's
`quantise_variance_cm2` over double bit patterns for the 255 code boundaries,
and the kernel finds the code by binary search. Exact by construction, and
cheaper than a log.
"""

import numpy as np

OPTIONS = ("--fmad=false",)

_PREAMBLE = r"""
// numpy's npy_divmod, as floor division. Line for line; see module docstring.
static inline double np_floor_divide(double a, double b) {
    if (b == 0.0) { return a / b; }
    double mod = fmod(a, b);
    double div = (a - mod) / b;
    if (mod != 0.0) {
        if ((b < 0.0) != (mod < 0.0)) { div -= 1.0; }
    }
    double floordiv;
    if (div != 0.0) {
        floordiv = floor(div);
        if (div - floordiv > 0.5) { floordiv += 1.0; }
    } else {
        floordiv = copysign(0.0, a / b);
    }
    return floordiv;
}
// Python/numpy integer floor division.
static inline long long floordiv_i64(long long a, long long b) {
    long long q = a / b;
    long long r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) { q -= 1; }
    return q;
}
"""

_cache = {}


def _kernel(name, in_params, out_params, body):
    k = _cache.get(name)
    if k is None:
        import cupy
        k = cupy.ElementwiseKernel(in_params, out_params, body, "vrgrid_" + name,
                                   preamble=_PREAMBLE, options=OPTIONS)
        _cache[name] = k
    return k


# --- perception ---------------------------------------------------------------

# Sort key for "closest return wins": pixel | range bits | point index. A
# positive float32's bit pattern orders exactly as its value, so sorting the
# key is sorting by (pixel, range, index) -- which is what JP's stable argsort
# by range followed by first-per-pixel selects. Unique keys, so the order is
# total and the device sort cannot pick a different winner on a tie.
KEY_IDX_BITS = 18      # points per frame < 262,144 (kernels.POINT_RADIX)
KEY_RANGE_BITS = 31    # a positive float32, sign bit dropped
KEY_PIX_BITS = 64 - KEY_RANGE_BITS - KEY_IDX_BITS   # 15: 32,768 pixels = 64 x 512
KEY_NONE = np.uint64(0xFFFFFFFFFFFFFFFF)


def project_keys():
    """`range_image.project`'s per-point arithmetic. float32 points only."""
    return _kernel("project_keys",
        "raw float32 pts, int64 ncols, float32 pi_f, float32 d_theta_f, "
        "float64 phi_max, float64 d_phi, int64 h, int64 w",
        "uint64 key, int8 fov",
        r"""
        float x = pts[ncols * i], y = pts[ncols * i + 1], z = pts[ncols * i + 2];
        float s = x * x + y * y;
        s = s + z * z;
        float r = sqrt(s);
        key = 0xFFFFFFFFFFFFFFFFull;
        fov = 0;
        if (r > 1e-6f) {
            float az = (float)atan2((double)y, (double)x);
            float zr = z / r;
            if (zr < -1.0f) zr = -1.0f;
            if (zr > 1.0f) zr = 1.0f;
            float el = (float)asin((double)zr);
            float t = az + pi_f;
            t = t / d_theta_f;
            long long u = (long long)floor(t);
            u = ((u % w) + w) % w;
            double vt = phi_max - (double)el;
            vt = vt / d_phi;
            long long v = (long long)floor(vt);
            if (v < 0) { fov = 1; v = 0; }
            else if (v >= h) { fov = 2; v = h - 1; }
            union { float f; unsigned int u; } bits;
            bits.f = r;
            unsigned long long pix = (unsigned long long)(v * w + u);
            key = (pix << 49) | ((unsigned long long)(bits.u & 0x7FFFFFFFu) << 18)
                  | (unsigned long long)i;
        }
        """)


def project_write():
    """Winner per pixel out of the sorted keys, written into the planes."""
    return _kernel("project_write",
        "raw uint64 keys, raw float32 pts, int64 ncols, int64 npix, "
        "raw float32 planes, raw int32 inverse",
        "",
        r"""
        unsigned long long k = keys[i];
        if (k == 0xFFFFFFFFFFFFFFFFull) continue;
        unsigned long long pix = k >> 49;
        if (i > 0) {
            unsigned long long prev = keys[i - 1];
            if (prev != 0xFFFFFFFFFFFFFFFFull && (prev >> 49) == pix) continue;
        }
        long long src = (long long)(k & 262143ull);
        union { float f; unsigned int u; } bits;
        bits.u = (unsigned int)((k >> 18) & 0x7FFFFFFFull);
        long long p = (long long)pix;
        const_cast<float&>(planes[p]) = bits.f;
        const_cast<float&>(planes[npix + p]) = pts[ncols * src];
        const_cast<float&>(planes[2 * npix + p]) = pts[ncols * src + 1];
        const_cast<float&>(planes[3 * npix + p]) = pts[ncols * src + 2];
        const_cast<float&>(planes[4 * npix + p]) = ncols >= 4 ? pts[ncols * src + 3] : 1.0f;
        const_cast<int&>(inverse[p]) = (int)src;
        """)


def reflectivity_to_points():
    """`reflectivity.normalise` (KITTI path: both compensations on, full scale
    1.0) then `scatter_to_points`, per pixel."""
    return _kernel("reflectivity",
        "raw float32 planes, raw int32 inverse, int64 npix, raw uint8 rho_pts",
        "",
        r"""
        double rng = (double)planes[i];
        double it = (double)planes[4 * npix + i];
        unsigned char b = 0;
        if (isfinite(rng) && !(rng <= 0.0) && isfinite(it)) {
            double q = it / 1.0;
            q = q * 255.0;
            q = nearbyint(q);
            if (q < 0.0) q = 0.0;
            if (q > 255.0) q = 255.0;
            b = (unsigned char)q;
        }
        int s = inverse[i];
        if (s >= 0) const_cast<unsigned char&>(rho_pts[s]) = b;
        """)


def semantics():
    """`semantics.semantic_labels` + `is_moving` through LUTs built from those
    functions, plus the engine's `where(semantic < 0, 0, semantic)` class."""
    return _kernel("semantics",
        "raw uint32 labels, raw int32 lut, raw bool moving_lut",
        "int32 sem, bool moving, uint8 cls",
        r"""
        unsigned int s = labels[i] & 0xFFFFu;
        sem = lut[s];
        moving = moving_lut[s];
        cls = sem < 0 ? (unsigned char)0 : (unsigned char)sem;
        """)


# --- map ----------------------------------------------------------------------

def payload(dtype):
    """`quantise_height(world z, datum)` and
    `quantise_weight(measurement_variance_cm2(max(range, 1e-3)))`, with the
    range taken in the POINT dtype exactly as `MapEngine.step` takes it."""
    from vrgrid.gpu.kernels import Z_MAX_CM, Z_MIN_CM

    c = "float" if np.dtype(dtype) == np.float32 else "double"
    t = np.dtype(dtype).name
    return _kernel(f"payload_{c}",
        f"raw {t} pts, int64 ncols, raw float64 world, float64 h_s, float64 sr2, "
        "float64 sp2, float64 cos2, float64 datum100, bool has_datum",
        "int16 zcm, int32 wq",
        rf"""
        const double ZMIN = {float(Z_MIN_CM)!r}, ZMAX = {float(Z_MAX_CM)!r};
        {c} x = pts[ncols * i], y = pts[ncols * i + 1], z = pts[ncols * i + 2];
        {c} s = x * x + y * y;
        s = s + z * z;
        {c} rng = sqrt(s);
        {c} lim = ({c})1e-3;
        {c} rm = (rng >= lim || isnan(rng)) ? rng : lim;
        double r = (double)rm;
        r = (r >= 1e-3 || isnan(r)) ? r : 1e-3;
        double t = h_s / r;
        double a = t * t;
        a = a * sr2;
        double b = r * r;
        b = b * sp2;
        double var = a + b;
        var = var / cos2;
        var = var * 1e4;
        double vv = (var >= 1e-9 || isnan(var)) ? var : 1e-9;
        double w = 1024.0 / vv;
        w = nearbyint(w);
        if (w < 1.0) w = 1.0;
        if (w > 1048576.0) w = 1048576.0;
        wq = (int)w;
        double zc = world[3 * i + 2] * 100.0;
        if (has_datum) zc = zc - datum100;
        zc = nearbyint(zc);
        if (zc < ZMIN || zc > ZMAX) wq = 0;   // kernels.out_of_band
        if (zc < ZMIN) zc = ZMIN;
        if (zc > ZMAX) zc = ZMAX;
        zcm = (short)zc;
        """)


def bin_points():
    """`grid.lattice.bin_points` (ring_of_into + lattice index + toroidal slot)
    for one point. `tab` is 7 x n_rings int64: k, side, x0, y0, offset, x0 mod
    side, y0 mod side -- `_fill_ring_tables`, packed. `per` is 3 x n_rings
    float64: block-centre offset x, offset y, heading half-extent --
    `lattice._descent_constants`, packed.

    The ring is decided per world-lattice block, coarse to fine, exactly as
    `ring_of_into` does: every float operation below is that function's, in
    its order, so a block on a boundary lands in the same ring on both."""
    return _kernel("bin_block",
        "raw float64 world, raw float64 radii, int64 n_rings, float64 a_f, "
        "float64 a_s, float64 a_r, int64 floor_ring, float64 rear_floor_m, "
        "float64 c0, float64 cy, float64 sy, raw float64 per, raw int64 tab",
        "int64 idx",
        r"""
        const long long N = n_rings;
        long long fx = (long long)np_floor_divide(world[3 * i], c0);
        long long fy = (long long)np_floor_divide(world[3 * i + 1], c0);

        long long tx = floordiv_i64(fx, tab[N - 1]) - tab[2 * N + N - 1];
        long long ty = floordiv_i64(fy, tab[N - 1]) - tab[3 * N + N - 1];
        long long Wt = tab[N + N - 1];
        long long lvl = (tx >= 0 && tx < Wt && ty >= 0 && ty < Wt) ? N - 1 : -1;

        for (long long M = N - 1; M >= 1 && lvl == M; --M) {
            long long kM = tab[M], WP = tab[N + M - 1];
            long long r = kM / tab[M - 1];
            long long bx = floordiv_i64(fx, kM), by = floordiv_i64(fy, kM);
            long long ex = bx * r - tab[2 * N + M - 1];
            long long ey = by * r - tab[3 * N + M - 1];
            bool fits = ex >= 0 && ex <= WP - r && ey >= 0 && ey <= WP - r;

            double h = per[3 * M + 2];
            double xc = (double)(bx * kM);
            xc = xc * c0;
            xc = xc + per[3 * M];
            double yc = (double)(by * kM);
            yc = yc * c0;
            yc = yc + per[3 * M + 1];
            double u = xc * cy;
            double t = yc * sy;
            u = u + t;
            double v = yc * cy;
            t = xc * sy;
            v = v - t;
            double fwd = u - h;
            fwd = fwd >= 0.0 ? fwd : 0.0;
            fwd = fwd / a_f;
            double rear = u + h;
            rear = -rear;
            rear = rear >= 0.0 ? rear : 0.0;
            rear = rear / a_r;
            double dd = fwd >= rear ? fwd : rear;
            double side = fabs(v);
            side = side - h;
            side = side >= 0.0 ? side : 0.0;
            side = side / a_s;
            dd = dd >= side ? dd : side;
            bool admit = dd < radii[M - 1];
            if (floor_ring >= 0 && M > floor_ring && u < 0.0 && fabs(u) < rear_floor_m)
                admit = true;
            if (fits && admit) lvl = M - 1;
        }

        long long lv = lvl > 0 ? lvl : 0;
        long long k = tab[lv], W = tab[N + lv];
        long long ix = floordiv_i64(fx, k);
        long long iy = floordiv_i64(fy, k);
        ix -= tab[2 * N + lv];
        bool live = ix >= 0 && ix < W;
        iy -= tab[3 * N + lv];
        live = live && iy >= 0 && iy < W;
        ix += tab[5 * N + lv];
        if (ix >= W) ix -= W;
        iy += tab[6 * N + lv];
        if (iy >= W) iy -= W;
        long long slot = iy * W + ix + tab[4 * N + lv];
        idx = (live && lvl >= 0) ? slot : -1;
        """)


def fuse():
    """`grid.fusion.fuse` for one touched cell: Kalman update (§3.3), ceiling,
    occupancy hit (§10.1), Boyer-Moore class (§10.2), reflectivity (§10.3) and
    counts. Cells are unique, so each thread owns its slot."""
    return _kernel("fuse",
        "raw int64 cells, raw int64 wz_sum, raw int64 w_sum, raw int32 cnt, "
        "raw int16 ceil_cm, raw int32 refl_sum, raw uint8 class_id, "
        "raw int16 gh, raw int16 ch, raw uint8 hv, raw int8 lo, raw uint8 sc, "
        "raw uint8 rf, raw uint8 oc, raw uint8 fss, raw float64 deq, raw float64 thr, "
        "float64 qdt, int32 hit, int32 lo_min, int32 lo_max",
        "",
        r"""
        long long s = cells[i];
        double pv = deq[hv[s]] + qdt;
        double pmu = (double)gh[s];

        long long wsum = w_sum[i], wz = wz_sum[i];
        long long wm = wsum >= 1 ? wsum : 1;
        long long sg = (wz > 0) - (wz < 0);
        long long mag = wz < 0 ? -wz : wz;
        int mh = (int)(sg * ((2 * mag + wm) / (2 * wm)));
        double z = (double)mh;
        double wd = (double)wsum;
        double mv = 1024.0 / (wd >= 1.0 ? wd : 1.0);

        double gain = pv / (pv + mv);
        double post_mu = z - pmu;
        post_mu = gain * post_mu;
        post_mu = pmu + post_mu;
        double post_var = 1.0 - gain;
        post_var = post_var * pv;
        if (oc[s] == 0) { post_mu = z; post_var = mv; }
        if (!(wsum > 0)) { post_mu = pmu; post_var = pv; }

        double rr = nearbyint(post_mu);
        if (rr < -32768.0) rr = -32768.0;
        if (rr > 32767.0) rr = 32767.0;
        const_cast<short&>(gh[s]) = (short)rr;

        int lo_c = 0, hi_c = 255;
        while (lo_c < hi_c) {
            int mid = (lo_c + hi_c + 1) / 2;
            if (post_var <= thr[mid - 1]) lo_c = mid; else hi_c = mid - 1;
        }
        const_cast<unsigned char&>(hv[s]) = (unsigned char)lo_c;

        short c0 = ch[s], c1 = ceil_cm[i];
        const_cast<short&>(ch[s]) = c0 <= c1 ? c0 : c1;

        int l = (int)lo[s] + hit;
        l = l < lo_min ? lo_min : (l > lo_max ? lo_max : l);
        const_cast<signed char&>(lo[s]) = (signed char)l;

        int p = sc[s], cand = p >> 3, cn = p & 7, ob = class_id[i], nc, nk;
        if (cn == 0) { nc = ob; nk = 1; }
        else if (cand == ob) { nc = cand; nk = cn + 1 < 7 ? cn + 1 : 7; }
        else { nc = cand; nk = cn - 1; }
        const_cast<unsigned char&>(sc[s]) = (unsigned char)((nc << 3) | nk);

        int nn = cnt[i] >= 1 ? cnt[i] : 1;
        int rv = refl_sum[i] / nn;
        rv = rv < 0 ? 0 : (rv > 255 ? 255 : rv);
        const_cast<unsigned char&>(rf[s]) = (unsigned char)rv;

        int o = (int)oc[s] + cnt[i];
        const_cast<unsigned char&>(oc[s]) = (unsigned char)(o < 255 ? o : 255);
        const_cast<unsigned char&>(fss[s]) = 0;
        """)


def occupied_mask():
    """`occupancy_state(...) == OCC_OCCUPIED`, per slot (§10.1)."""
    return _kernel("occupied",
        "int8 lo, uint8 oc, uint8 fl, int32 l_occ, int32 unknown_below, uint8 blind",
        "bool m",
        "m = ((int)lo > l_occ) && ((int)oc >= unknown_below) && ((fl & blind) == 0);")


def apply_miss():
    """`visibility.apply_miss`: clamped log-odds decrement where seen through."""
    return _kernel("apply_miss",
        "raw int64 slots, raw bool see_through, raw int8 lo, int32 miss, "
        "int32 lo_min, int32 lo_max",
        "",
        r"""
        if (see_through[i]) {
            signed char& L = const_cast<signed char&>(lo[slots[i]]);
            int v = (int)L + miss;
            v = v < lo_min ? lo_min : (v > lo_max ? lo_max : v);
            L = (signed char)v;
        }
        """)


def rebase_heights():
    """`shift.track_datum`'s in-range re-base: ground always, ceiling only
    where one was seen (the sentinel is not a height), and a ground height that
    leaves the band loses its evidence (variance code 0)."""
    return _kernel("rebase",
        "int32 delta, int32 zmin, int32 zmax, int16 none",
        "int16 g, int16 c, uint8 var",
        r"""
        int v = (int)g - delta;
        if (v < zmin || v > zmax) var = 0;
        v = v < zmin ? zmin : (v > zmax ? zmax : v);
        g = (short)v;
        if (c != none) {
            int u = (int)c - delta;
            u = u < zmin ? zmin : (u > zmax ? zmax : u);
            c = (short)u;
        }
        """)


# --- host-side tables, derived from the reference functions themselves ---------

def dequantise_lut() -> np.ndarray:
    """The 256 variances `dequantise_variance_cm2` decodes, from that function."""
    from vrgrid.grid.quantise import dequantise_variance_cm2
    return np.asarray(dequantise_variance_cm2(np.arange(256)), np.float64)


def variance_code_thresholds() -> np.ndarray:
    """thr[c-1] = the LARGEST double v with `quantise_variance_cm2(v) >= c`.

    Found by bisection over positive double bit patterns, which order exactly
    as the values do, evaluating Aakash's function itself. Valid because the
    codec is monotone non-increasing in v, and that is checked on the result
    rather than assumed: every threshold must encode to >= c and its successor
    to < c, or this raises.
    """
    from vrgrid.grid.quantise import quantise_variance_cm2 as q

    c = np.arange(1, 256)
    lo = np.full(255, np.float64(1e-300)).view(np.uint64).copy()
    hi = np.full(255, np.float64(1e300)).view(np.uint64).copy()
    for _ in range(80):
        if np.all(hi - lo <= 1):
            break
        mid = lo + (hi - lo) // np.uint64(2)
        ok = q(mid.view(np.float64)).astype(np.int64) >= c
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    thr = lo.view(np.float64).copy()
    nxt = (lo + np.uint64(1)).view(np.float64)
    if not (np.all(q(thr).astype(np.int64) >= c) and np.all(q(nxt).astype(np.int64) < c)):
        raise RuntimeError("quantise_variance_cm2 is not monotone; the device "
                           "codec table cannot represent it")
    if np.any(np.diff(thr) > 0):
        raise RuntimeError("variance code thresholds are not non-increasing")
    return thr
