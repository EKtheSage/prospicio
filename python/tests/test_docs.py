"""The public API is documented, and its docstring examples run.

Docstrings come from the Rust `///` comments in crates/act-python, so these
tests guard what the stubs and the docs site are generated from.
"""

import doctest
import inspect

import pytest

import actuarialrs as ar

NAMESPACES = [ar.distributions, ar.aggregate]


def public_objects():
    for ns in NAMESPACES:
        for name in ns.__all__:
            obj = getattr(ns, name)
            yield f"{ns.__name__}.{name}", obj
            if inspect.isclass(obj):
                for member in vars(obj):
                    if not member.startswith("_"):
                        # getattr, not the raw class dict: on Python 3.9 a
                        # staticmethod wrapper reports its own type's docstring.
                        yield f"{ns.__name__}.{name}.{member}", getattr(obj, member)


@pytest.mark.parametrize("path, obj", list(public_objects()), ids=lambda x: x if isinstance(x, str) else "")
def test_public_api_has_docstring(path, obj):
    doc = inspect.getdoc(obj)
    assert doc and doc.strip(), f"{path} has no docstring"


@pytest.mark.parametrize("ns", NAMESPACES, ids=lambda ns: ns.__name__)
def test_docstring_examples_run(ns):
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    for name in ns.__all__:
        for test in finder.find(getattr(ns, name), f"{ns.__name__}.{name}"):
            runner.run(test)
    results = runner.summarize(verbose=False)
    assert results.attempted > 0, f"{ns.__name__} has no docstring examples"
    assert results.failed == 0
