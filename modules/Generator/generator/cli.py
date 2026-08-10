"""SoftModel command line.

No model types, no schemas: create takes just a name; the MODEL shapes
itself (features, numeric vs categorical output, vocabulary, capacity) from
the data it is taught.

Examples:
    python -m generator.cli create risk --holdout examples/risk/holdout.jsonl
    python -m generator.cli teach  risk --data examples/risk/train.jsonl
    python -m generator.cli infer  risk '{"amount": 800, "night": 0, "foreign": 0}'
    python -m generator.cli eval   risk
    python -m generator.cli versions risk
    python -m generator.cli discoveries risk           # readable regularities
    python -m generator.cli rollback risk v1
    python -m generator.cli card   risk
    python -m generator.cli demo

Drift awareness (M2):
    python -m generator.cli add-holdout risk --data fresh_reality.jsonl
    python -m generator.cli drift risk --recent 8
    python -m generator.cli teach risk --data new_cases.jsonl --window 12
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .factory import SoftModelFactory
from .spec import ModelSpec
from .data import read_jsonl


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="generator", description="SoftModel Model factory CLI")
    p.add_argument("--backend", default=None, help="mlp | mock (overrides env)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("create", help="create a SoftModel model (no types, no schemas)")
    sp.add_argument("model_id")
    sp.add_argument("--description", default="")
    sp.add_argument("--holdout", default=None, help="jsonl of {input,target}")

    sp = sub.add_parser("teach", help="teach (train + gated promote) a model")
    sp.add_argument("model_id")
    sp.add_argument("--data", required=True, help="jsonl of {input,target}")
    sp.add_argument("--mode", default="sft")
    sp.add_argument("--window", type=int, default=None,
                    help="train on only the most recent N stored examples (M2)")
    sp.add_argument("--recent", type=int, default=None,
                    help="judge the gate on only the most recent N held-out examples (M2)")

    sp = sub.add_parser("add-holdout", help="append fresh labeled reality to held-out (M2)")
    sp.add_argument("model_id")
    sp.add_argument("--data", required=True, help="jsonl of {input,target}")

    sp = sub.add_parser("drift", help="check whether the active version has drifted (M2)")
    sp.add_argument("model_id")
    sp.add_argument("--recent", type=int, default=None,
                    help="score on the most recent N held-out examples")

    sp = sub.add_parser("infer", help="run the model")
    sp.add_argument("model_id")
    sp.add_argument("input")
    sp.add_argument("--version", default=None)

    sp = sub.add_parser("eval", help="evaluate a version on held-out")
    sp.add_argument("model_id")
    sp.add_argument("--version", default=None)

    for name in ("versions", "card"):
        sp = sub.add_parser(name, help=f"{name} of a model")
        sp.add_argument("model_id")

    sp = sub.add_parser("discoveries",
                        help="validated regularities of a Type-D model (M3)")
    sp.add_argument("model_id")
    sp.add_argument("--version", default=None)

    sp = sub.add_parser("rollback", help="set the active version")
    sp.add_argument("model_id")
    sp.add_argument("version")

    sub.add_parser("demo", help="run the built-in control-loop demo (mock)")

    args = p.parse_args(argv)

    if args.cmd == "demo":
        try:
            import os
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root not in sys.path:
                sys.path.insert(0, root)
            from scripts import demo_m0  # type: ignore
        except Exception as e:  # packaged install without scripts/
            print(f"demo requires the repo checkout (scripts/demo_m0.py): {e}")
            return 1
        return demo_m0.main()

    cfg = Config.from_env(**({"backend": args.backend} if args.backend else {}))
    f = SoftModelFactory(cfg)

    if args.cmd == "create":
        _print(f.create(ModelSpec.from_files(
            args.model_id, holdout_path=args.holdout,
            description=args.description)))
    elif args.cmd == "teach":
        _print(f.teach(args.model_id, read_jsonl(args.data), mode=args.mode,
                       window=args.window, recent_n=args.recent))
    elif args.cmd == "add-holdout":
        _print(f.add_holdout(args.model_id, read_jsonl(args.data)))
    elif args.cmd == "drift":
        _print(f.check_drift(args.model_id, recent_n=args.recent))
    elif args.cmd == "infer":
        try:
            parsed = json.loads(args.input)   # feature object / vector
        except (json.JSONDecodeError, ValueError):
            parsed = args.input               # plain text (mock backend)
        _print(f.infer(args.model_id, parsed, version=args.version))
    elif args.cmd == "eval":
        _print(f.evaluate(args.model_id, version=args.version))
    elif args.cmd == "versions":
        _print(f.versions(args.model_id))
    elif args.cmd == "discoveries":
        _print(f.discoveries(args.model_id, version=args.version))
    elif args.cmd == "card":
        _print(f.card(args.model_id))
    elif args.cmd == "rollback":
        _print(f.rollback(args.model_id, args.version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
