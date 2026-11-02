"""Check that the analysis dependencies import and that HiGHS can solve a small LP.

Run with the repo's virtual environment:

    .venv\\Scripts\\python.exe scripts\\check_env.py
"""

from __future__ import annotations

import importlib
import sys

MODULES = [
    "pandas",
    "numpy",
    "pyarrow",
    "highspy",
    "pyomo",
    "pytest",
    "scipy",
    "numpy_financial",
    "pdfplumber",
    "matplotlib",
    "requests",
]


def check_imports() -> list[str]:
    failed = []
    for name in MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report whatever went wrong
            print(f"FAIL  {name:18s} {exc}")
            failed.append(name)
        else:
            print(f"ok    {name:18s} {getattr(module, '__version__', '?')}")
    return failed


def check_highs() -> bool:
    """Solve max 3x + 2y s.t. x + y <= 4, x <= 3, y <= 3; optimum is x=3, y=1, obj=11."""
    import highspy
    import numpy as np

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.addVars(2, np.array([0.0, 0.0]), np.array([3.0, 3.0]))
    h.changeColsCost(2, np.array([0, 1], dtype=np.int32), np.array([3.0, 2.0]))
    h.addRow(-highspy.kHighsInf, 4.0, 2, np.array([0, 1], dtype=np.int32), np.array([1.0, 1.0]))
    h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    h.run()

    status = h.getModelStatus()
    obj = h.getObjectiveValue()
    ok = status == highspy.HighsModelStatus.kOptimal and abs(obj - 11.0) < 1e-9
    print(f"{'ok   ' if ok else 'FAIL '} highs LP          status={h.modelStatusToString(status)} obj={obj:.6f}")
    return ok


def main() -> int:
    print(sys.version)
    print(f"interpreter: {sys.executable}\n")

    failed = check_imports()
    print()
    highs_ok = check_highs()

    if failed or not highs_ok:
        if failed:
            print(f"\nmissing or broken: {', '.join(failed)}")
        return 1
    print("\nenvironment ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
