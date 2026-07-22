#!/usr/bin/env python3
"""Command-line entry point for the paper's numerical tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper.appendix import run_e1, run_e2, run_e3
from paper.section5 import (
    run_bm19,
    run_budget_calibration,
    run_fixed_scores,
    run_options,
    run_reserve_family,
)
from paper.tables import table_1, table_2, table_3, table_4, table_6, table_7


TARGETS = {
    "budget": "Table 1: joint versus decoupled calibration",
    "fixed": "Tables 2--3: fixed affine-signature expectation and probability",
    "options": "Table 4: Asian and lookback options",
    "reserve": "Table 6: four reserve-path functionals",
    "bm19": "Table 7: BM19 J1 comparison",
    "tables": "assemble numerical Tables 1--4 and 6--7",
    "e1": "Appendix E.1 parametric geometry and figures",
    "e2": "Appendix E.2 model-library geometry and figures",
    "e3": "Appendix E.3 penalty/depth ablations",
    "section5": "all Section 5 experiments and tables",
    "appendix": "all Appendix E experiments",
    "all": "the complete paper",
}


def build_tables(root: Path, profile: str) -> None:
    output = root / "paper_tables"; output.mkdir(parents=True, exist_ok=True)
    expected = tuple(range(5)) if profile == "paper" else (0,)
    frames = {
        "01_budget_calibration_decoupled_vs_joint.csv": table_1(root, expected),
        "02_linear_risk_factor_expectation.csv": table_2(root, expected),
        "03_linear_tail_probability.csv": table_3(root, expected),
        "04_path_option_vol_sweep.csv": table_4(root, expected),
        "06_reserve_risk_family.csv": table_6(root, expected),
        "07_bm19_tailored_comparison.csv": table_7(root, expected),
    }
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)


def run(target: str, profile: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    section5 = target in {"section5", "all"}
    appendix = target in {"appendix", "all"}
    if target == "budget" or section5:
        run_budget_calibration(profile, root)
    if target == "fixed" or section5:
        run_fixed_scores(profile, root)
    if target == "options" or section5:
        run_options(profile, root)
    if target == "reserve" or section5:
        run_reserve_family(profile, root)
    if target == "bm19" or section5:
        run_bm19(profile, root)
    if target == "tables" or section5:
        build_tables(root, profile)
    if target == "e1" or appendix:
        run_e1(profile, root)
    if target == "e2" or appendix:
        run_e2(profile, root)
    if target == "e3" or appendix:
        run_e3(profile, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", nargs="?", default="all", choices=TARGETS)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke",
                        help="smoke uses reduced samples; paper uses the reported experiment settings")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--list", action="store_true", help="show available targets and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print("\n".join(f"{name:10s} {description}" for name, description in TARGETS.items()))
        return
    run(args.target, args.profile, args.run_dir.resolve())
    print(f"\nCompleted {args.target!r} ({args.profile}) under {args.run_dir.resolve()}")


if __name__ == "__main__":
    main()
