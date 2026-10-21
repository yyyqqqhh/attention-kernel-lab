"""Run source checks and tests, keeping skipped tests visible in a JSON report."""

import argparse
import ast
import contextlib
import io
import json
import platform
from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/validation.json"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sources = sorted(path for directory in ("src", "tests", "tools")
                     for path in (root / directory).rglob("*.py"))
    syntax_errors = []
    for source in sources:
        try:
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except SyntaxError as error:
            syntax_errors.append({"file": source.relative_to(root).as_posix(),
                                  "line": error.lineno, "message": error.msg})
    log = io.StringIO()
    # Test imports sometimes emit warnings. Keep report JSON separate from stdout.
    with contextlib.redirect_stdout(log):
        suite = unittest.defaultTestLoader.discover(str(root / "tests"))
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "syntax": {"files_checked": len(sources), "errors": syntax_errors,
                   "scope": "Python AST parse only; no Triton JIT compilation"},
        "tests": {"run": result.testsRun, "passed": result.testsRun - len(result.skipped)
                  - len(result.failures) - len(result.errors) - len(result.expectedFailures),
                  "skipped": [{"test": test.id(), "reason": reason}
                              for test, reason in result.skipped],
                  "failures": [{"test": test.id(), "detail": detail}
                               for test, detail in result.failures],
                  "errors": [{"test": test.id(), "detail": detail}
                             for test, detail in result.errors]},
        "log": log.getvalue(),
        "success": result.wasSuccessful() and not syntax_errors,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(log.getvalue(), end="")
    print(f"AST files: {len(sources)}; report: {arguments.output}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
