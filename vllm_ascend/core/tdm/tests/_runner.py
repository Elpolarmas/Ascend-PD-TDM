"""Minimal pytest substitute: scans every test_*.py in this dir, runs every
top-level callable named ``test_*``, and reports pass/fail. Lets us verify
TDM logic without a pytest install (offline env)."""
import importlib
import pkgutil
import sys
import traceback
from pathlib import Path


def main() -> int:
    here = Path(__file__).parent
    pkg = "vllm_ascend.core.tdm.tests"
    failures = []
    passed = 0
    for mod_info in pkgutil.iter_modules([str(here)]):
        if not mod_info.name.startswith("test_"):
            continue
        mod = importlib.import_module(f"{pkg}.{mod_info.name}")
        for attr in dir(mod):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            label = f"{mod_info.name}::{attr}"
            try:
                fn()
                passed += 1
                print(f"PASS  {label}")
            except Exception:
                failures.append(label)
                print(f"FAIL  {label}")
                traceback.print_exc()
    print(f"\n{passed} passed, {len(failures)} failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
