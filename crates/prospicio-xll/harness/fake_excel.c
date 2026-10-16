/* A stand-in for Excel that loads the XLL and drives it through the C API,
 * for a smoke test without Excel (docs/design/xll.md). It exports
 * MdCallBack12 as Excel does, answers xlGetName, xlfRegister and xlFree,
 * then calls worksheet functions with XLOPER12 arguments and prints what
 * comes back.
 *
 *   x86_64-w64-mingw32-gcc -O2 -o fake_excel.exe fake_excel.c
 *   wine fake_excel.exe prospicio.xll
 *
 * Exits non-zero when a registration or a check fails. */
#include <windows.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#define xltypeNum 0x0001
#define xltypeStr 0x0002
#define xltypeErr 0x0010
#define xltypeMulti 0x0040
#define xltypeMissing 0x0080
#define xltypeNil 0x0100
#define xltypeInt 0x0800
#define xlbitDLLFree 0x4000
#define xlFree 0x4000
#define xlGetName (9 | 0x4000)
#define xlfRegister 149

typedef struct xloper12 {
    union {
        double num;
        wchar_t *str;
        int xbool;
        int err;
        int w;
        struct {
            struct xloper12 *lparray;
            int rows;
            int columns;
        } array;
        unsigned char size[24];
    } val;
    unsigned int xltype;
} XLOPER12;

typedef XLOPER12 *(*Fn0)(void);
typedef XLOPER12 *(*Fn9)(XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *,
                         XLOPER12 *, XLOPER12 *, XLOPER12 *);
typedef XLOPER12 *(*Fn8)(XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *,
                         XLOPER12 *, XLOPER12 *);
typedef XLOPER12 *(*Fn7)(XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *,
                         XLOPER12 *);
typedef XLOPER12 *(*Fn5)(XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *);
typedef XLOPER12 *(*Fn6)(XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *);
typedef void (*Free12)(XLOPER12 *);

static int registered = 0, failures = 0;
static wchar_t dll_name[] = L"\x0F" L"C:\\prospicio.xll";

static void narrow(const wchar_t *s, char *out, size_t cap) {
    size_t n = (size_t)s[0];
    if (n >= cap) n = cap - 1;
    for (size_t i = 0; i < n; i++) out[i] = (char)s[i + 1];
    out[n] = 0;
}

__declspec(dllexport) int __stdcall MdCallBack12(int xlfn, int count, XLOPER12 **args, XLOPER12 *res) {
    if (xlfn == xlGetName) {
        res->xltype = xltypeStr;
        res->val.str = dll_name;
        return 0;
    }
    if (xlfn == xlFree) return 0;
    if (xlfn == xlfRegister) {
        char proc[256], types[256], name[256], argnames[256];
        narrow(args[1]->val.str, proc, sizeof proc);
        narrow(args[2]->val.str, types, sizeof types);
        narrow(args[3]->val.str, name, sizeof name);
        narrow(args[4]->val.str, argnames, sizeof argnames);
        int nargs = (int)strlen(types) - 2; /* result type and '$' */
        int ok = args[0]->xltype == xltypeStr && args[5]->xltype == xltypeInt &&
                 count == 10 + nargs;
        printf("register %-16s %-8s %-28s (%d args) %s\n", name, types, argnames, count,
               ok ? "ok" : "BAD");
        if (!ok) failures++;
        registered++;
        res->xltype = xltypeNum;
        res->val.num = registered;
        return 0;
    }
    return 1;
}

static void print_value(const XLOPER12 *x) {
    char buf[512];
    switch (x->xltype & 0x0FFF) {
    case xltypeNum: printf("%14.4f", x->val.num); break;
    case xltypeStr: narrow(x->val.str, buf, sizeof buf); printf("%14s", buf); break;
    case xltypeErr: printf("%14s", "#ERR"); break;
    default: printf("%14s", "?");
    }
}

static void print_result(const char *label, XLOPER12 *x) {
    printf("%s:\n", label);
    if ((x->xltype & 0x0FFF) == xltypeMulti) {
        for (int r = 0; r < x->val.array.rows; r++) {
            for (int c = 0; c < x->val.array.columns; c++)
                print_value(&x->val.array.lparray[r * x->val.array.columns + c]);
            printf("\n");
        }
    } else {
        print_value(x);
        printf("\n");
    }
    if (!(x->xltype & xlbitDLLFree)) {
        printf("  result not marked xlbitDLLFree\n");
        failures++;
    }
}

static void check(const char *what, int ok) {
    printf("check %-50s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) failures++;
}

static const double RAA[10][10] = {
    {5012, 8269, 10907, 11805, 13539, 16181, 18009, 18608, 18662, 18834},
    {106, 4285, 5396, 10666, 13782, 15599, 15496, 16169, 16704},
    {3410, 8992, 13873, 16141, 18735, 22214, 22863, 23466},
    {5655, 11555, 15766, 21266, 23425, 26083, 27067},
    {1092, 9565, 15836, 22169, 25955, 26180},
    {1513, 6445, 11702, 12935, 15852},
    {557, 4020, 10946, 12314},
    {1351, 6947, 13112},
    {3133, 5395},
    {2063},
};

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "prospicio.xll";
    HMODULE xll = LoadLibraryA(path);
    if (!xll) {
        printf("cannot load %s (error %lu)\n", path, GetLastError());
        return 2;
    }
    int (*open)(void) = (int (*)(void))GetProcAddress(xll, "xlAutoOpen");
    Free12 free12 = (Free12)GetProcAddress(xll, "xlAutoFree12");
    check("xlAutoOpen returns 1", open && open() == 1);
    check("forty functions registered", registered == 40);

    /* RAA as a Q argument: a 10 x 10 multi with Nil below the diagonal. */
    XLOPER12 cells[100], tri, missing, simple, tail;
    for (int r = 0; r < 10; r++)
        for (int c = 0; c < 10; c++) {
            XLOPER12 *x = &cells[r * 10 + c];
            if (c < 10 - r) {
                x->xltype = xltypeNum;
                x->val.num = RAA[r][c];
            } else {
                x->xltype = xltypeNil;
            }
        }
    tri.xltype = xltypeMulti;
    tri.val.array.lparray = cells;
    tri.val.array.rows = 10;
    tri.val.array.columns = 10;
    missing.xltype = xltypeMissing;
    static wchar_t simple_s[] = L"\x06" L"simple";
    simple.xltype = xltypeStr;
    simple.val.str = simple_s;
    tail.xltype = xltypeNum;
    tail.val.num = 1.05;

    XLOPER12 *v = ((Fn0)GetProcAddress(xll, "pro_version"))();
    print_result("PRO.VERSION()", v);
    free12(v);

    XLOPER12 *ldf = ((Fn6)GetProcAddress(xll, "pro_cl_ldf"))(&tri, &missing, &tail, &missing, &missing, &missing);
    print_result("PRO.CL.LDF(RAA,,1.05)", ldf);
    check("LDF is 1 x 10", (ldf->xltype & 0xFFF) == xltypeMulti && ldf->val.array.rows == 1 &&
                               ldf->val.array.columns == 10);
    check("last LDF is the tail", ldf->val.array.lparray[9].val.num == 1.05);
    free12(ldf);

    Fn8 mack = (Fn8)GetProcAddress(xll, "pro_mack");
    XLOPER12 *m = mack(&tri, &missing, &missing, &missing, &missing, &missing, &missing, &missing);
    print_result("PRO.MACK(RAA)", m);
    XLOPER12 *total = &m->val.array.lparray[11 * 8];
    check("total reserve 52135.228 (R ChainLadder)", fabs(total[3].val.num - 52135.2282612101) < 1e-6);
    check("total Mack SE 26880.740 (R ChainLadder)", fabs(total[6].val.num - 26880.7403298914) < 1e-6);
    free12(m);

    /* Mack's rule for the last sigma: R's MackChainLadder(est.sigma = "Mack"). */
    static wchar_t mack_s[] = L"\x04" L"mack";
    XLOPER12 rule;
    rule.xltype = xltypeStr;
    rule.val.str = mack_s;
    XLOPER12 *rs = mack(&tri, &missing, &missing, &missing, &missing, &missing, &missing, &rule);
    check("Mack SE with Mack's sigma rule 26909.011 (R ChainLadder)",
          fabs(rs->val.array.lparray[11 * 8 + 6].val.num - 26909.0111556436) < 1e-6);
    free12(rs);

    XLOPER12 *bad = mack(&tri, &tail, &missing, &missing, &missing, &missing, &missing, &missing);
    print_result("PRO.MACK(RAA, 1.05)", bad);
    check("a bad argument comes back as text", (bad->xltype & 0xFFF) == xltypeStr);
    free12(bad);

    XLOPER12 *s = ((Fn7)GetProcAddress(xll, "pro_cl"))(&tri, &simple, &missing, &missing, &missing, &missing, &missing);
    print_result("PRO.CL(RAA, \"simple\")", s);
    check("simple-average reserve 93643.031 (R ChainLadder)",
          fabs(s->val.array.lparray[11 * 5 + 4].val.num - 93643.0313432212) < 1e-6);
    free12(s);

    /* Many calls, freed each time: a leak or a double free shows here. */
    for (int i = 0; i < 2000; i++) free12(mack(&tri, &missing, &missing, &missing, &missing, &missing, &missing, &missing));
    check("2000 calls and frees", 1);

    /* Layer pricing: Pareto(t = 500, alpha = 2), a column of attachments
     * against a row of limits, the last "unlimited" (R package Pareto:
     * 4000 xs 1000 is 200; unlimited xs 1000 is 250). */
    static wchar_t pareto_s[] = L"\x06" L"pareto";
    static wchar_t unlimited_s[] = L"\x09" L"unlimited";
    XLOPER12 att_cells[2], lim_cells[2], att, lim, fam, t, alpha;
    att_cells[0].xltype = att_cells[1].xltype = xltypeNum;
    att_cells[0].val.num = 1000;
    att_cells[1].val.num = 5000;
    att.xltype = xltypeMulti;
    att.val.array.lparray = att_cells;
    att.val.array.rows = 2;
    att.val.array.columns = 1;
    lim_cells[0].xltype = xltypeNum;
    lim_cells[0].val.num = 4000;
    lim_cells[1].xltype = xltypeStr;
    lim_cells[1].val.str = unlimited_s;
    lim.xltype = xltypeMulti;
    lim.val.array.lparray = lim_cells;
    lim.val.array.rows = 1;
    lim.val.array.columns = 2;
    fam.xltype = xltypeStr;
    fam.val.str = pareto_s;
    t.xltype = xltypeNum;
    t.val.num = 500;
    alpha.xltype = xltypeNum;
    alpha.val.num = 2;
    XLOPER12 *layer = ((Fn6)GetProcAddress(xll, "pro_layer"))(&lim, &att, &fam, &t, &alpha, &missing);
    print_result("PRO.LAYER({4000,\"unlimited\"}, {1000;5000}, \"pareto\", 500, 2)", layer);
    check("layer table is 2 x 2", (layer->xltype & 0xFFF) == xltypeMulti && layer->val.array.rows == 2 &&
                                       layer->val.array.columns == 2);
    check("4000 xs 1000 is 200 (R package Pareto)", fabs(layer->val.array.lparray[0].val.num - 200) < 1e-9);
    check("unlimited xs 1000 is 250", fabs(layer->val.array.lparray[1].val.num - 250) < 1e-9);
    free12(layer);
    XLOPER12 basic, lim1m;
    basic.xltype = xltypeNum;
    basic.val.num = 1000;
    lim1m.xltype = xltypeNum;
    lim1m.val.num = 1000;
    XLOPER12 *f = ((Fn6)GetProcAddress(xll, "pro_ilf"))(&lim1m, &basic, &fam, &t, &alpha, &missing);
    print_result("PRO.ILF(1000, 1000, \"pareto\", 500, 2)", f);
    check("ILF at the basic limit is 1", (f->xltype & 0xFFF) == xltypeNum && f->val.num == 1.0);
    free12(f);

    /* ODP bootstrap: 20000 simulations, seed 1; RAA's total within four
     * standard errors of R ChainLadder's BootChainLadder (100000 sims). */
    XLOPER12 sims, seed;
    sims.xltype = xltypeNum;
    sims.val.num = 20000;
    seed.xltype = xltypeNum;
    seed.val.num = 1;
    DWORD t0 = GetTickCount();
    XLOPER12 *o = ((Fn7)GetProcAddress(xll, "pro_odp"))(&tri, &sims, &seed, &missing, &missing, &missing, &missing);
    DWORD ms = GetTickCount() - t0;
    print_result("PRO.ODP(RAA, 20000, 1)", o);
    printf("  (%lu ms)\n", ms);
    check("ODP table is 12 x 9", (o->xltype & 0xFFF) == xltypeMulti && o->val.array.rows == 12 &&
                                     o->val.array.columns == 9);
    XLOPER12 *ot = &o->val.array.lparray[11 * 9];
    check("ODP total mean 53842 +- 589 (R BootChainLadder)", fabs(ot[2].val.num - 53842.27) < 589.4);
    check("ODP total SD 18965 +- 484 (R BootChainLadder)", fabs(ot[3].val.num - 18965.35) < 483.9);
    free12(o);

    /* The draws: 100000 rows of 11 cells and a header, allocated by the
     * XLL and freed by xlAutoFree12. */
    sims.val.num = 100000;
    t0 = GetTickCount();
    XLOPER12 *d = ((Fn6)GetProcAddress(xll, "pro_odp_sims"))(&tri, &sims, &seed, &missing, &missing, &missing);
    ms = GetTickCount() - t0;
    printf("PRO.ODP.SIMS(RAA, 100000, 1): %d x %d (%lu ms)\n", d->val.array.rows, d->val.array.columns, ms);
    check("draws are 100001 x 11", (d->xltype & 0xFFF) == xltypeMulti && d->val.array.rows == 100001 &&
                                       d->val.array.columns == 11);
    char head[16];
    narrow(d->val.array.lparray[10].val.str, head, sizeof head);
    check("last column is the total", strcmp(head, "Total") == 0);
    double sum = 0;
    for (int r = 1; r <= 100000; r++) sum += d->val.array.lparray[r * 11 + 10].val.num;
    check("mean total within R's tolerance of 53842", fabs(sum / 100000 - 53842.27) < 589.4);
    free12(d);

    /* Mack's bootstrap: 20000 simulations; the total SD reconciles with
     * Mack's sqrt(process^2 + 0.981 parameter^2) = 26845 (R ChainLadder's
     * risks), within five Monte Carlo standard errors (about 1000). */
    sims.val.num = 20000;
    XLOPER12 *mb = ((Fn9)GetProcAddress(xll, "pro_mack_boot"))(&tri, &sims, &seed, &missing, &missing, &missing,
                                                               &missing, &missing, &missing);
    print_result("PRO.MACK.BOOT(RAA, 20000, 1)", mb);
    check("Mack bootstrap table is 12 x 9", (mb->xltype & 0xFFF) == xltypeMulti && mb->val.array.rows == 12 &&
                                                mb->val.array.columns == 9);
    XLOPER12 *mt = &mb->val.array.lparray[11 * 9];
    check("Mack bootstrap total SD 26845 +- 1000 (Mack, R ChainLadder)", fabs(mt[3].val.num - 26845) < 1000);
    check("Mack bootstrap total mean 52135 +- 950 (chain ladder)", fabs(mt[2].val.num - 52135.23) < 950);
    free12(mb);

    /* Bornhuetter-Ferguson on RAA with a flat exposure of 20000 and a
     * priori 1: the oldest origin's ultimate is its latest, and the total
     * reserve sums the a priori ultimate's unreported shares. */
    XLOPER12 exp_cells[10], expo, apriori;
    for (int i = 0; i < 10; i++) {
        exp_cells[i].xltype = xltypeNum;
        exp_cells[i].val.num = 20000;
    }
    expo.xltype = xltypeMulti;
    expo.val.array.lparray = exp_cells;
    expo.val.array.rows = 10;
    expo.val.array.columns = 1;
    apriori.xltype = xltypeNum;
    apriori.val.num = 1.0;
    XLOPER12 *bf = ((Fn6)GetProcAddress(xll, "pro_bf"))(&tri, &expo, &apriori, &missing, &missing, &missing);
    print_result("PRO.BF(RAA, 20000 each, 1)", bf);
    check("BF table is 12 x 7", (bf->xltype & 0xFFF) == xltypeMulti && bf->val.array.rows == 12 &&
                                    bf->val.array.columns == 7);
    double expect = 0;
    for (int i = 1; i <= 10; i++) {
        double cdf = bf->val.array.lparray[i * 7 + 3].val.num;
        expect += 20000 * (1 - 1 / cdf);
    }
    check("BF reserve is the unreported share of 20000 each",
          fabs(bf->val.array.lparray[11 * 7 + 6].val.num - expect) < 1e-6);
    free12(bf);

    /* Merz-Wuthrich: RAA's total CDR(1) SE 25166.303 (R ChainLadder's CDR). */
    XLOPER12 *cd = ((Fn8)GetProcAddress(xll, "pro_mack_cdr"))(&tri, &missing, &missing, &missing, &missing,
                                                              &missing, &missing, &missing);
    print_result("PRO.MACK.CDR(RAA)", cd);
    check("total CDR(1) SE 25166.303 (R ChainLadder)",
          fabs(cd->val.array.lparray[11 * 4 + 2].val.num - 25166.3025264382) < 1e-6);
    free12(cd);
    sims.val.num = 2000;
    XLOPER12 *oc = ((Fn7)GetProcAddress(xll, "pro_odp_cdr"))(&tri, &sims, &seed, &missing, &missing, &missing,
                                                             &missing);
    check("ODP one-year table is 12 x 8", (oc->xltype & 0xFFF) == xltypeMulti && oc->val.array.rows == 12 &&
                                              oc->val.array.columns == 8);
    free12(oc);

    /* Clark's LDF method on RAA, log-logistic to infinity: total reserve
     * 111006.63 (R ChainLadder's ClarkLDF run to convergence, 1e-5). */
    XLOPER12 *ck = ((Fn5)GetProcAddress(xll, "pro_clark"))(&tri, &missing, &missing, &missing, &missing);
    print_result("PRO.CLARK(RAA)", ck);
    check("Clark LDF total reserve 111006.63 (R ChainLadder)",
          fabs(ck->val.array.lparray[11 * 7 + 3].val.num - 111006.625765179) < 1.2);
    free12(ck);

    /* Triangle helpers: incremental and back is the triangle again. */
    typedef XLOPER12 *(*Fn1)(XLOPER12 *);
    XLOPER12 *inc = ((Fn1)GetProcAddress(xll, "pro_tri_incremental"))(&tri);
    XLOPER12 *cum = ((Fn1)GetProcAddress(xll, "pro_tri_cumulative"))(inc);
    int same = (cum->xltype & 0xFFF) == xltypeMulti && cum->val.array.rows == 10 && cum->val.array.columns == 10;
    for (int i = 0; same && i < 100; i++) {
        XLOPER12 *a = &cells[i], *b = &cum->val.array.lparray[i];
        same = (a->xltype == xltypeNil) ? (b->xltype & 0xFFF) == xltypeStr : b->val.num == a->val.num;
    }
    check("PRO.TRI.CUMULATIVE(PRO.TRI.INCREMENTAL(RAA)) is RAA", same);
    free12(inc);
    free12(cum);

    /* A gamma severity's cdf at its quantile: the exponential's 1 - e^-1. */
    static wchar_t gamma_s[] = L"\x05" L"gamma";
    XLOPER12 gfam, one, thousand;
    gfam.xltype = xltypeStr;
    gfam.val.str = gamma_s;
    one.xltype = thousand.xltype = xltypeNum;
    one.val.num = 1;
    thousand.val.num = 1000;
    XLOPER12 *gc = ((Fn5)GetProcAddress(xll, "pro_sev_cdf"))(&thousand, &gfam, &one, &thousand, &missing);
    check("PRO.SEV.CDF(1000, \"gamma\", 1, 1000) is 1 - 1/e",
          (gc->xltype & 0xFFF) == xltypeNum && fabs(gc->val.num - (1 - exp(-1.0))) < 1e-15);
    free12(gc);

    /* Pareto alpha between two frequencies: ln(4) / ln(2) = 2. */
    XLOPER12 t1, f1, t2, f2;
    t1.xltype = f1.xltype = t2.xltype = f2.xltype = xltypeNum;
    t1.val.num = 1e6;
    f1.val.num = 4;
    t2.val.num = 2e6;
    f2.val.num = 1;
    XLOPER12 *pa = ((Fn5)GetProcAddress(xll, "pro_pareto_alpha_freqs"))(&t1, &f1, &t2, &f2, &missing);
    check("PRO.PARETO.ALPHA.FREQS(1e6, 4, 2e6, 1) is 2", (pa->xltype & 0xFFF) == xltypeNum && fabs(pa->val.num - 2) < 1e-14);
    free12(pa);

    /* Pareto(1, 3), unlimited xs 0: E[X^2] - E[X]^2 = 3 - 1.5^2. */
    XLOPER12 unl, zero, three;
    unl.xltype = xltypeStr;
    unl.val.str = unlimited_s;
    zero.xltype = three.xltype = xltypeNum;
    zero.val.num = 0;
    three.val.num = 3;
    XLOPER12 *lv = ((Fn6)GetProcAddress(xll, "pro_layer_var"))(&unl, &zero, &fam, &one, &three, &missing);
    check("PRO.LAYER.VAR(\"unlimited\", 0, \"pareto\", 1, 3) is 0.75",
          (lv->xltype & 0xFFF) == xltypeNum && fabs(lv->val.num - 0.75) < 1e-12);
    free12(lv);

    printf(failures ? "\n%d FAILED\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
