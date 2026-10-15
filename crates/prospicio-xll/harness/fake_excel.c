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
typedef XLOPER12 *(*Fn3)(XLOPER12 *, XLOPER12 *, XLOPER12 *);
typedef XLOPER12 *(*Fn4)(XLOPER12 *, XLOPER12 *, XLOPER12 *, XLOPER12 *);
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
    check("eight functions registered", registered == 8);

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

    XLOPER12 *ldf = ((Fn3)GetProcAddress(xll, "pro_cl_ldf"))(&tri, &missing, &tail);
    print_result("PRO.CL.LDF(RAA,,1.05)", ldf);
    check("LDF is 1 x 10", (ldf->xltype & 0xFFF) == xltypeMulti && ldf->val.array.rows == 1 &&
                               ldf->val.array.columns == 10);
    check("last LDF is the tail", ldf->val.array.lparray[9].val.num == 1.05);
    free12(ldf);

    Fn4 mack = (Fn4)GetProcAddress(xll, "pro_mack");
    XLOPER12 *m = mack(&tri, &missing, &missing, &missing);
    print_result("PRO.MACK(RAA)", m);
    XLOPER12 *total = &m->val.array.lparray[11 * 8];
    check("total reserve 52135.228 (R ChainLadder)", fabs(total[3].val.num - 52135.2282612101) < 1e-6);
    check("total Mack SE 26880.740 (R ChainLadder)", fabs(total[6].val.num - 26880.7403298914) < 1e-6);
    free12(m);

    XLOPER12 *bad = mack(&tri, &tail, &missing, &missing);
    print_result("PRO.MACK(RAA, 1.05)", bad);
    check("a bad argument comes back as text", (bad->xltype & 0xFFF) == xltypeStr);
    free12(bad);

    XLOPER12 *s = ((Fn4)GetProcAddress(xll, "pro_cl"))(&tri, &simple, &missing, &missing);
    print_result("PRO.CL(RAA, \"simple\")", s);
    check("simple-average reserve 93643.031 (R ChainLadder)",
          fabs(s->val.array.lparray[11 * 5 + 4].val.num - 93643.0313432212) < 1e-6);
    free12(s);

    /* Many calls, freed each time: a leak or a double free shows here. */
    for (int i = 0; i < 2000; i++) free12(mack(&tri, &missing, &missing, &missing));
    check("2000 calls and frees", 1);

    printf(failures ? "\n%d FAILED\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}
