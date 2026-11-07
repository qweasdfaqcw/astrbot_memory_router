"""不依赖 pytest 的测试运行器（便于离线环境自检）。

优先推荐 ``pytest``；本脚本仅使用标准库，逐个执行 ``tests/test_*.py`` 中的
``test_*`` 函数，并为需要 ``tmp_path`` 的测试创建独立的 ``.test_tmp`` 目录。
"""

from __future__ import annotations

import importlib
import inspect
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / ".test_tmp"


def main() -> int:
    shutil.rmtree(BASE, ignore_errors=True)
    BASE.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str, str]] = []
    total = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        module = importlib.import_module(f"tests.{path.stem}")
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            total += 1
            case_dir = BASE / f"{path.stem}_{name}"
            shutil.rmtree(case_dir, ignore_errors=True)
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                if "tmp_path" in inspect.signature(func).parameters:
                    func(case_dir)
                else:
                    func()
                print(f"PASS {path.name}::{name}")
            except Exception:  # noqa: BLE001
                failures.append((path.name, name, traceback.format_exc()))
                print(f"FAIL {path.name}::{name}")
    print(f"\n{total - len(failures)}/{total} passed")
    for path, name, tb in failures:
        print(f"\n--- {path}::{name}\n{tb}")
    shutil.rmtree(BASE, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
