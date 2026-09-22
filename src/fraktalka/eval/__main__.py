"""Run the make-or-break experiment:  python -m fraktalka.eval [--seed N] [--models N]"""

from __future__ import annotations

import argparse

from .generalization import format_report, run


def main() -> int:
    parser = argparse.ArgumentParser(description="fraktalka Layer-0 falsification harness")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", type=int, default=24)
    args = parser.parse_args()
    print(format_report(run(seed=args.seed, n_models=args.models)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
