"""Run one isolated experiment callable with a standard temporal window.

This is intentionally a manual/CI utility, not a production model endpoint.
Future experiments should live under ``backend.app.ml.experiments`` and expose a
callable accepting training_start, training_end and test_end (or a subset).
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
from typing import Any, Callable

_ALLOWED = re.compile(r"^backend\.app\.ml\.experiments\.[a-z0-9_]+:[a-z_][a-z0-9_]*$")


def load_experiment(spec: str) -> Callable[..., Any]:
    if not _ALLOWED.fullmatch(spec):
        raise ValueError(
            "Experiment callable must be backend.app.ml.experiments.<module>:<function>; "
            "arbitrary modules are not allowed."
        )
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if function is None or not callable(function):
        raise ValueError(f"Experiment callable does not exist: {spec}")
    return function


def invoke(function: Callable[..., Any], *, start: int, end: int, test_end: int) -> Any:
    signature = inspect.signature(function)
    available = {
        "training_start": int(start),
        "training_end": int(end),
        "test_end": int(test_end),
    }
    kwargs = {name: available[name] for name in signature.parameters if name in available}
    required = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    if required:
        raise ValueError(
            "Experiment callable has unsupported required parameters: " + ", ".join(required)
        )
    return function(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--callable", required=True, dest="callable_spec")
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--test-end", required=True, type=int)
    args = parser.parse_args()
    if args.start > args.end:
        raise SystemExit("training start cannot be after training end")
    if args.test_end <= args.end:
        raise SystemExit("test-end must be after training end")

    function = load_experiment(args.callable_spec)
    report = invoke(function, start=args.start, end=args.end, test_end=args.test_end)
    summary = {
        "callable": args.callable_spec,
        "training_period": [args.start, args.end],
        "testing_period": [args.end + 1, args.test_end],
        "experiment": report.get("experiment") if isinstance(report, dict) else None,
        "run_id": report.get("run_id") if isinstance(report, dict) else None,
        "status": report.get("status") if isinstance(report, dict) else None,
        "decision": report.get("decision") if isinstance(report, dict) else None,
    }
    print("EXPERIMENT_RUN_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
