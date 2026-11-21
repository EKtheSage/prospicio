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


def hawkes_new(baseline, beta, label="SciPy 1.18.1"):
    return (
        b"case,horizon,periods,baseline,alpha,beta,log_likelihood,source\n"
        b"a,1.0,0.5|0.7000000000000001;0.2,%s,9.0735956889067779,%s,-3.5,%s\n"
        % (baseline, beta, label.encode())
    )


def check(got, want, case):
    assert got == want, f"{case}: {got!r}, want {want!r}"


def main():
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
    # Another file keeps the 1e-9 comparison, and its list cells (the periods) are exact text.
    diffs, _ = rr.compare("other.csv", HAWKES_OLD, hawkes_new(b"1.5210145700988877", b"288.23212707844078"))
    assert len(diffs) == 4 and diffs[0] == "line 2: 0.5|0.7;0.2 -> 0.5|0.7000000000000001;0.2", diffs
    print("rerun_references self-test passed")


if __name__ == "__main__":
    main()
