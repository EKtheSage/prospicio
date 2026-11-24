"""Self-test of rerun_references.compare on CSVs without tolerance columns (#935).

    python3 validation/scripts/test_rerun_references.py

CI runs it with the other validation script self-tests.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import rerun_references as rr  # noqa: E402

SOBOL_OLD = b"dim,coord,points,source\n1,0,0;1;2,SciPy 1.14.1\n2,0,0;1;3,SciPy 1.14.1\n"
SOBOL_NEW = SOBOL_OLD.replace(b"1.14.1", b"1.18.1")
HAWKES_OLD = (
    b"case,horizon,periods,baseline,alpha,beta,log_likelihood,source\n"
    b"a,1.0,0.5|0.7;0.2,1.5210146113899532,9.0735959257929863,288.23210269653163,-3.5,SciPy 1.14.1\n"
)


def hawkes_new(baseline, beta, label="SciPy 1.18.1", ll=b"-3.5"):
    return (
        b"case,horizon,periods,baseline,alpha,beta,log_likelihood,source\n"
        b"a,1.0,0.5|0.7000000000000001;0.2,%s,9.0735956889067779,%s,%s,%s\n"
        % (baseline, beta, ll, label.encode())
    )


def check(got, want, case):
    assert got == want, f"{case}: {got!r}, want {want!r}"


TOL_HEAD = b"distribution,params,quantity,arg,expected,abs_tol,rel_tol,source\n"


def main():
    # A float in a key moving in its last digit is the same row, and its value is still compared (#1046).
    old = TOL_HEAD + b"d,p,cdf,56.79453034175243,0.5,0.0,1e-12,scipy 1\n"
    new = TOL_HEAD + b"d,p,cdf,56.794530341752406,0.5,0.0,1e-12,scipy 2\n"
    check(rr.compare("x.csv", old, new), ([], ["scipy 1 -> scipy 2"]), "key float drift")
    diffs, _ = rr.compare("x.csv", old, new.replace(b",0.5,", b",0.6,"))
    assert len(diffs) == 1 and "0.5 -> 0.6" in diffs[0], diffs
    # A key that really moved is still a row removed and another added.
    diffs, _ = rr.compare("x.csv", old, new.replace(b"56.79", b"57.79"))
    assert len(diffs) == 2, diffs
    # A version label alone is a moved source, not a difference.
    check(rr.compare("sobol_scipy.csv", SOBOL_OLD, SOBOL_NEW), ([], ["SciPy 1.14.1 -> SciPy 1.18.1"]), "label only")
    # A changed value in a file with a source column is still a difference.
    changed = SOBOL_NEW.replace(b"0;1;3", b"0;1;4")
    diffs, moved = rr.compare("sobol_scipy.csv", SOBOL_OLD, changed)
    assert len(diffs) == 1 and moved == ["SciPy 1.14.1 -> SciPy 1.18.1"], (diffs, moved)
    # Sobol points are exact 32-bit integers: one moved by 1 (well inside 1e-9 relative) is a difference.
    big = b"dim,coord,points,source\n1,0,0;2147483648;4026531840,SciPy 1.14.1\n"
    diffs, _ = rr.compare("sobol_scipy.csv", big, big.replace(b"2147483648", b"2147483649"))
    check(diffs, ["line 2: 0;2147483648;4026531840 -> 0;2147483649;4026531840"], "sobol point by 1")
    # Event times that moved in the last ulp and an optimiser's last digits (3e-8 to 9e-8 relative) pass for hawkes_scipy.csv.
    got = rr.compare("hawkes_scipy.csv", HAWKES_OLD, hawkes_new(b"1.5210145700988877", b"288.23212707844078"))
    check(got, ([], ["SciPy 1.14.1 -> SciPy 1.18.1"]), "hawkes drift")
    # A real change (1e-4 relative) does not.
    diffs, _ = rr.compare("hawkes_scipy.csv", HAWKES_OLD, hawkes_new(b"1.5211668", b"288.23210269653163"))
    assert len(diffs) == 1, diffs
    # The log-likelihood keeps the Rust test's absolute bound (1e-7, inside its 1e-6):
    # a move of 1e-4 (1.5e-7 relative at 660, inside 1e-6 relative) is a difference.
    hawkes_ll = HAWKES_OLD.replace(b",-3.5,", b",660.00505458772454,")
    ll_moved = hawkes_new(b"1.5210145700988877", b"288.23212707844078", ll=b"660.00515458772454")
    diffs, _ = rr.compare("hawkes_scipy.csv", hawkes_ll, ll_moved)
    check(diffs, ["line 2: 660.00505458772454 -> 660.00515458772454"], "hawkes log-likelihood by 1e-4")
    # The last-digit noise of a stable log-likelihood (1e-13) still passes.
    ll_noise = hawkes_new(b"1.5210145700988877", b"288.23212707844078", ll=b"660.00505458772464")
    check(rr.compare("hawkes_scipy.csv", hawkes_ll, ll_noise), ([], ["SciPy 1.14.1 -> SciPy 1.18.1"]), "hawkes ll noise")
    # Another file keeps the 1e-9 comparison, and its list cells (the periods) are exact text.
    diffs, _ = rr.compare("other.csv", HAWKES_OLD, hawkes_new(b"1.5210145700988877", b"288.23212707844078"))
    # It is not in SOURCE_LABELS, so its `source` column is compared too.
    assert len(diffs) == 5 and diffs[0] == "line 2: 0.5|0.7;0.2 -> 0.5|0.7000000000000001;0.2", diffs
    assert diffs[-1] == "line 2: SciPy 1.14.1 -> SciPy 1.18.1", diffs
    # In the pnl_*_sources.csv files `source` is a data key the Rust tests filter rows on:
    # a changed key is a difference, not a moved source.
    builder = b"source,value,prob\nex3a,1.0,0.5\nex3a,3.0,0.5\nex5,2.0,1.0\n"
    got = rr.compare("pnl_builder_sources.csv", builder, builder.replace(b"ex3a,3.0", b"ex9z,3.0", 1))
    check(got, (["line 3: ex3a -> ex9z"], []), "pnl builder key")
    occurrence = b"source,gross,ceded,prob\nex4,10.0,2.0,1.0\nex7,5.0,1.0,1.0\n"
    got = rr.compare("pnl_occurrence_sources.csv", occurrence, occurrence.replace(b"ex4", b"zzz"))
    check(got, (["line 2: ex4 -> zzz"], []), "pnl occurrence key")
    print("rerun_references self-test passed")


if __name__ == "__main__":
    main()
