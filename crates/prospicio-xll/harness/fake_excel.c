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
typedef XLOPER12 *(*Fn19)(XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *,
                          XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *,
                          XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *,
                          XLOPER12 *);
typedef XLOPER12 *(*Fn15)(XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *,
                          XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *,
                          XLOPER12 *, XLOPER12 *, XLOPER12 *);
typedef XLOPER12 *(*Fn10)(XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *,
                          XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *);
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
    check("ninety-six functions registered", registered == 96);

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
    XLOPER12 *o = ((Fn8)GetProcAddress(xll, "pro_odp"))(&tri, &sims, &seed, &missing, &missing, &missing, &missing,
                                                       &missing);
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
    XLOPER12 *d = ((Fn7)GetProcAddress(xll, "pro_odp_sims"))(&tri, &sims, &seed, &missing, &missing, &missing, &missing);
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
    XLOPER12 *oc = ((Fn8)GetProcAddress(xll, "pro_odp_cdr"))(&tri, &sims, &seed, &missing, &missing, &missing,
                                                             &missing, &missing);
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

    /* The ODP GLM on RAA: the chain ladder's total reserve, 52135.23. */
    typedef XLOPER12 *(*Fn2)(XLOPER12 *, XLOPER12 *);
    XLOPER12 *og = ((Fn2)GetProcAddress(xll, "pro_odp_glm"))(&tri, &missing);
    check("PRO.ODP.GLM total reserve is the chain ladder's 52135.23",
          (og->xltype & 0xFFF) == xltypeMulti && og->val.array.rows == 12 && og->val.array.columns == 8 &&
              fabs(og->val.array.lparray[11 * 8 + 3].val.num - 52135.2283) < 0.01);
    free12(og);

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

    /* Swiss Re c = 3: G(1) = 1, and the c recovered from G's mean rate. */
    static wchar_t swiss_s[] = L"\x08" L"swiss_re";
    XLOPER12 sfam;
    sfam.xltype = xltypeStr;
    sfam.val.str = swiss_s;
    XLOPER12 *g1 = ((Fn5)GetProcAddress(xll, "pro_exposure_curve"))(&one, &sfam, &three, &missing, &missing);
    check("PRO.EXPOSURE.CURVE(1, \"swiss_re\", 3) is 1",
          (g1->xltype & 0xFFF) == xltypeNum && fabs(g1->val.num - 1) < 1e-15);
    free12(g1);

    /* Square root rule: a quarter of the standard gives half credibility. */
    XLOPER12 quarter, full;
    quarter.xltype = full.xltype = xltypeNum;
    quarter.val.num = 250;
    full.val.num = 1000;
    XLOPER12 *cz = ((Fn2)GetProcAddress(xll, "pro_cred_z"))(&quarter, &full);
    check("PRO.CRED.Z(250, 1000) is 0.5", (cz->xltype & 0xFFF) == xltypeNum && fabs(cz->val.num - 0.5) < 1e-15);
    free12(cz);

    /* An Excel date on the half-month grid: 30 June 2013 is 2013.5. */
    XLOPER12 june30;
    june30.xltype = xltypeNum;
    june30.val.num = 41455;
    XLOPER12 *yf = ((Fn1)GetProcAddress(xll, "pro_yearfraction"))(&june30);
    check("PRO.YEARFRACTION(DATE(2013,6,30)) is 2013.5", (yf->xltype & 0xFFF) == xltypeNum && yf->val.num == 2013.5);
    free12(yf);

    /* Werner and Modlin, ch. 8: LR 65%, V 25%, Q 10%, F 6.5% is +10%. */
    typedef XLOPER12 *(*Fn4)(XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *);
    XLOPER12 lr, vv, qq, ff;
    lr.xltype = vv.xltype = qq.xltype = ff.xltype = xltypeNum;
    lr.val.num = 0.65;
    vv.val.num = 0.25;
    qq.val.num = 0.10;
    ff.val.num = 0.065;
    XLOPER12 *ic = ((Fn4)GetProcAddress(xll, "pro_indicated_change"))(&lr, &vv, &qq, &ff);
    check("PRO.INDICATED.CHANGE(0.65, 0.25, 0.10, 0.065) is +10%",
          (ic->xltype & 0xFFF) == xltypeNum && fabs(ic->val.num - 0.10) < 1e-12);
    free12(ic);

    /* aggregate's oep: Poisson 2 of a lognormal with mean 100 and CV 2,
     * the 100-year occurrence PML is 1171.4979 (aggregate 1.0.1). */
    static wchar_t lncv_s[] = L"\x0c" L"lognormal_cv";
    XLOPER12 lncv, two, hundred;
    lncv.xltype = xltypeStr;
    lncv.val.str = lncv_s;
    two.xltype = hundred.xltype = xltypeNum;
    two.val.num = 2;
    hundred.val.num = 100;
    XLOPER12 *pml = ((Fn9)GetProcAddress(xll, "pro_agg_pml"))(&two, &missing, &lncv, &hundred, &two, &missing,
                                                             &hundred, &missing, &missing);
    check("PRO.AGG.PML 100-year occurrence PML 1171.4979 (aggregate)",
          (pml->xltype & 0xFFF) == xltypeMulti && pml->val.array.rows == 2 && pml->val.array.columns == 6 &&
              fabs(pml->val.array.lparray[8].val.num / 1171.497873505582 - 1) < 1e-10 &&
              pml->val.array.lparray[9].val.num > pml->val.array.lparray[8].val.num);
    free12(pml);

    /* A stop loss of 500 xs 400 on the same year, net of 500 xs 100 per
     * occurrence: a column of nine rows, the cover's mean below the
     * subject's and its chance of exhausting below that of attaching. */
    XLOPER12 sl_l, sl_r, occ_l, occ_a;
    sl_l.xltype = sl_r.xltype = occ_l.xltype = occ_a.xltype = xltypeNum;
    sl_l.val.num = 500;
    sl_r.val.num = 400;
    occ_l.val.num = 500;
    occ_a.val.num = 100;
    XLOPER12 *sl = ((Fn15)GetProcAddress(xll, "pro_agg_cover"))(&two, &missing, &lncv, &hundred, &two, &missing,
                                                                &sl_l, &sl_r, &missing, &occ_l, &occ_a, &missing,
                                                                &missing, &missing, &missing);
    check("PRO.AGG.COVER stop loss net of an occurrence layer",
          (sl->xltype & 0xFFF) == xltypeMulti && sl->val.array.rows == 9 && sl->val.array.columns == 2 &&
              sl->val.array.lparray[9].val.num < sl->val.array.lparray[3].val.num &&
              sl->val.array.lparray[7].val.num < sl->val.array.lparray[5].val.num &&
              sl->val.array.lparray[7].val.num > 0);
    free12(sl);

    /* aggregate's book C priced by PH 0.7 on its grid (step 1, 2^14
     * points, the severity limited to 1,000): the gross premium is
     * 297.44319 and 500 xs 100's 139.35875 (aggregate 1.0.1). */
    static wchar_t ph_s[] = L"\x02" L"ph";
    XLOPER12 ph, rho, cap, n14;
    ph.xltype = xltypeStr;
    ph.val.str = ph_s;
    rho.xltype = cap.xltype = n14.xltype = xltypeNum;
    rho.val.num = 0.7;
    cap.val.num = 1000;
    n14.val.num = 16384;
    XLOPER12 *tp = ((Fn19)GetProcAddress(xll, "pro_tower_price"))(
        &two, &missing, &lncv, &hundred, &two, &missing, &cap, &missing, &occ_l, &occ_a, &missing, &missing,
        &missing, &missing, &missing, &ph, &rho, &one, &n14);
    check("PRO.TOWER.PRICE book C under PH 0.7 (aggregate)",
          (tp->xltype & 0xFFF) == xltypeMulti && tp->val.array.rows == 4 && tp->val.array.columns == 8 &&
              fabs(tp->val.array.lparray[12].val.num / 297.44318995353353 - 1) < 2e-8 &&
              fabs(tp->val.array.lparray[20].val.num / 139.35875318947413 - 1) < 2e-8);
    free12(tp);

    /* The Bayesian ODP chain ladder by NUTS on RAA, its chains on worker
     * threads: the diagnostics' first row is the largest R-hat. */
    static wchar_t diag_s[] = L"\x0b" L"diagnostics";
    XLOPER12 diag;
    diag.xltype = xltypeStr;
    diag.val.str = diag_s;
    XLOPER12 *bd = ((Fn10)GetProcAddress(xll, "pro_bayes_odp"))(&tri, &diag, &missing, &missing, &missing,
                                                                 &missing, &missing, &missing, &missing, &missing);
    int bd_ok = (bd->xltype & 0xFFF) == xltypeMulti && bd->val.array.columns == 2 &&
                (bd->val.array.lparray[1].xltype & 0xFFF) == xltypeNum;
    double rhat = bd_ok ? bd->val.array.lparray[1].val.num : 99;
    printf("PRO.BAYES.ODP(RAA, \"diagnostics\"): max R-hat %.4f\n", rhat);
    check("PRO.BAYES.ODP samples RAA with R-hat under 1.01", bd_ok && rhat < 1.01);
    free12(bd);

    /* A lognormal fitted to a column of losses: meanlog is the mean of the
     * logs, in the params table's second row. */
    XLOPER12 sf_cells[4], sf_col, sf_fam;
    double sf_y[4] = {100, 200, 400, 800};
    for (int i = 0; i < 4; i++) {
        sf_cells[i].xltype = xltypeNum;
        sf_cells[i].val.num = sf_y[i];
    }
    sf_col.xltype = xltypeMulti;
    sf_col.val.array.rows = 4;
    sf_col.val.array.columns = 1;
    sf_col.val.array.lparray = sf_cells;
    static wchar_t ln_s[] = L"\x09" L"lognormal";
    sf_fam.xltype = xltypeStr;
    sf_fam.val.str = ln_s;
    XLOPER12 *sf = ((Fn9)GetProcAddress(xll, "pro_sev_fit"))(&sf_col, &sf_fam, &missing, &missing, &missing,
                                                              &missing, &missing, &missing, &missing);
    check("PRO.SEV.FIT lognormal: meanlog is the mean of the logs",
          (sf->xltype & 0xFFF) == xltypeMulti && sf->val.array.columns == 3 &&
              fabs(sf->val.array.lparray[4].val.num - log(100 * pow(2, 1.5))) < 1e-6);
    free12(sf);

    /* Bahnemann's P2.16: a gamma by minimum chi-square on six bands, the
     * top one open (a blank last bound): shape 4.7432. */
    XLOPER12 gb_cells[7], gc_cells[6], gbnd, gcnt, g_fam, g_meth;
    double gb_x[6] = {0, 1000, 1500, 2000, 2500, 3000}, gc_x[6] = {42, 61, 47, 26, 14, 10};
    for (int i = 0; i < 6; i++) {
        gb_cells[i].xltype = xltypeNum;
        gb_cells[i].val.num = gb_x[i];
        gc_cells[i].xltype = xltypeNum;
        gc_cells[i].val.num = gc_x[i];
    }
    gb_cells[6].xltype = xltypeNil;
    gbnd.xltype = gcnt.xltype = xltypeMulti;
    gbnd.val.array.rows = gcnt.val.array.rows = 1;
    gbnd.val.array.columns = 7;
    gcnt.val.array.columns = 6;
    gbnd.val.array.lparray = gb_cells;
    gcnt.val.array.lparray = gc_cells;
    static wchar_t gfam_s[] = L"\x05" L"gamma";
    static wchar_t chisq_s[] = L"\x0a" L"chi_square";
    g_fam.xltype = g_meth.xltype = xltypeStr;
    g_fam.val.str = gfam_s;
    g_meth.val.str = chisq_s;
    XLOPER12 *gf = ((Fn5)GetProcAddress(xll, "pro_sev_fit_grouped"))(&gbnd, &gcnt, &g_fam, &g_meth, &missing);
    check("PRO.SEV.FIT.GROUPED gamma by minimum chi-square: shape 4.7432 (Bahnemann P2.16)",
          (gf->xltype & 0xFFF) == xltypeMulti && gf->val.array.columns == 3 &&
              fabs(gf->val.array.lparray[4].val.num - 4.7432) < 5e-5);
    free12(gf);

    /* Hill's estimate from the 3 largest of 1, 2, 4, 8, 16: the mean log
     * of 16/2, 8/2, 4/2 is 2 ln 2. */
    XLOPER12 hk;
    double hy[5] = {1, 2, 4, 8, 16};
    XLOPER12 h_cells[5];
    for (int i = 0; i < 5; i++) {
        h_cells[i].xltype = xltypeNum;
        h_cells[i].val.num = hy[i];
    }
    sf_col.val.array.rows = 5;
    sf_col.val.array.lparray = h_cells;
    hk.xltype = xltypeNum;
    hk.val.num = 3;
    XLOPER12 *hl = ((Fn2)GetProcAddress(xll, "pro_evt_hill"))(&sf_col, &hk);
    check("PRO.EVT.HILL from the 3 largest of 1, 2, 4, 8, 16 is 2 ln 2",
          (hl->xltype & 0xFFF) == xltypeMulti && hl->val.array.rows == 2 &&
              fabs(hl->val.array.lparray[6].val.num - 2 * log(2.0)) < 1e-12);
    free12(hl);

    /* One payment after a year at 10%: the discount factor is 1 / 1.1. */
    XLOPER12 py_t, py_r;
    py_t.xltype = py_r.xltype = xltypeNum;
    py_t.val.num = 1;
    py_r.val.num = 0.1;
    XLOPER12 *ps = ((Fn5)GetProcAddress(xll, "pro_payout_stats"))(&py_t, &py_t, &py_r, &missing, &missing);
    check("PRO.PAYOUT.STATS one payment after a year at 10% is 1 / 1.1",
          (ps->xltype & 0xFFF) == xltypeMulti && ps->val.array.rows == 3 &&
              fabs(ps->val.array.lparray[1].val.num - 1 / 1.1) < 1e-15);
    free12(ps);

    /* A risk premium of 87 with 3% expenses and 10% brokerage grosses up
     * to a technical premium of 100. */
    XLOPER12 q_e, q_r, q_x, q_b;
    q_e.xltype = q_r.xltype = q_x.xltype = q_b.xltype = xltypeNum;
    q_e.val.num = 50;
    q_r.val.num = 87;
    q_x.val.num = 0.03;
    q_b.val.num = 0.10;
    XLOPER12 *qt = ((Fn8)GetProcAddress(xll, "pro_quote"))(&q_e, &q_r, &missing, &q_x, &q_b, &missing, &missing,
                                                           &missing);
    check("PRO.QUOTE grosses a risk premium of 87 up to 100",
          (qt->xltype & 0xFFF) == xltypeMulti && qt->val.array.columns == 2 &&
              fabs(qt->val.array.lparray[3].val.num - 100) < 1e-12);
    free12(qt);

    /* The proportional hazard with rho 0.5 at s = 0.25: g = 0.25^0.5. */
    static wchar_t phd_s[] = L"\x02" L"ph";
    XLOPER12 d_s, d_f, d_p;
    d_s.xltype = d_p.xltype = xltypeNum;
    d_s.val.num = 0.25;
    d_p.val.num = 0.5;
    d_f.xltype = xltypeStr;
    d_f.val.str = phd_s;
    XLOPER12 *dg = ((Fn4)GetProcAddress(xll, "pro_distortion"))(&d_s, &d_f, &d_p, &missing);
    check("PRO.DISTORTION(0.25, \"ph\", 0.5) is 0.5",
          (dg->xltype & 0xFFF) == xltypeNum && fabs(dg->val.num - 0.5) < 1e-15);
    free12(dg);

    /* A three-year term at 50% of the profit over a threshold of 80 (premium
     * 100, allowance 20%): losses 120, 70, 30 carry a deficit of 40, of
     * which 30 is left after year 2, so year 3 pays 0.5 x (50 - 30) = 10. */
    XLOPER12 tm_cells[3], tm_row, tm_p, tm_s, tm_a;
    double tm_y[3] = {120, 70, 30};
    for (int i = 0; i < 3; i++) {
        tm_cells[i].xltype = xltypeNum;
        tm_cells[i].val.num = tm_y[i];
    }
    tm_row.xltype = xltypeMulti;
    tm_row.val.array.rows = 1;
    tm_row.val.array.columns = 3;
    tm_row.val.array.lparray = tm_cells;
    tm_p.xltype = tm_s.xltype = tm_a.xltype = xltypeNum;
    tm_p.val.num = 100;
    tm_s.val.num = 0.5;
    tm_a.val.num = 0.2;
    XLOPER12 *pct = ((Fn5)GetProcAddress(xll, "pro_profit_commission_term"))(&tm_row, &tm_p, &tm_s, &tm_a, &missing);
    check("PRO.PROFIT.COMMISSION.TERM carries a deficit into year 3",
          (pct->xltype & 0xFFF) == xltypeMulti && pct->val.array.columns == 3 &&
              fabs(pct->val.array.lparray[2].val.num - 10) < 1e-12);
    free12(pct);

    /* The Gaussian copula at (1/2, 1/2) with rho 1/2: 1/4 + asin(1/2) / (2 pi) = 1/3. */
    static wchar_t gau_s[] = L"\x08" L"gaussian";
    XLOPER12 cu_cells[2], cu_row, cu_f, cu_r;
    for (int i = 0; i < 2; i++) {
        cu_cells[i].xltype = xltypeNum;
        cu_cells[i].val.num = 0.5;
    }
    cu_row.xltype = xltypeMulti;
    cu_row.val.array.rows = 1;
    cu_row.val.array.columns = 2;
    cu_row.val.array.lparray = cu_cells;
    cu_f.xltype = xltypeStr;
    cu_f.val.str = gau_s;
    cu_r.xltype = xltypeNum;
    cu_r.val.num = 0.5;
    XLOPER12 *cc = ((Fn4)GetProcAddress(xll, "pro_copula_cdf"))(&cu_row, &cu_f, &cu_r, &missing);
    check("PRO.COPULA.CDF Gaussian at (1/2, 1/2), rho 1/2, is 1/3",
          (cc->xltype & 0xFFF) == xltypeNum && fabs(cc->val.num - 1.0 / 3.0) < 1e-15);
    free12(cc);

    /* A profit commission of 25% after a 10% allowance on a premium of
     * 100 at a loss of 50: 0.25 x (90 - 50) = 10. */
    XLOPER12 pc_x, pc_p, pc_s, pc_a;
    pc_x.xltype = pc_p.xltype = pc_s.xltype = pc_a.xltype = xltypeNum;
    pc_x.val.num = 50;
    pc_p.val.num = 100;
    pc_s.val.num = 0.25;
    pc_a.val.num = 0.1;
    XLOPER12 *pcm = ((Fn4)GetProcAddress(xll, "pro_profit_commission"))(&pc_x, &pc_p, &pc_s, &pc_a);
    check("PRO.PROFIT.COMMISSION(50, 100, 0.25, 0.1) is 10",
          (pcm->xltype & 0xFFF) == xltypeNum && fabs(pcm->val.num - 10) < 1e-12);
    free12(pcm);

    printf(failures ? "\n%d FAILED\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
