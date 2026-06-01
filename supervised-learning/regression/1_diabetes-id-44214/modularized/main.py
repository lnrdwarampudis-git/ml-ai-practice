"""
main.py — Entrypoint for the Diabetes ML Pipeline (OpenML 44214).

Usage
-----
# Run all 50 phases, output to ./diabetes_output (default)
python main.py

# Custom output directory
python main.py --output-dir /path/to/your/folder

# Specific phases only
python main.py --output-dir ./my_results --phases 1 2 3

# Change seed / CV folds
python main.py --seed 123 --cv-folds 10 --output-dir ./run2

# Full help
python main.py --help
"""

import sys
import time
import traceback

# ── stdlib path fix so sub-packages resolve regardless of cwd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import parse_args
from phases import (
    phase1_foundations,
    phase2_data_readiness,
    phase3_modeling,
    phase4_production,
    phase5_advanced,
    summary_dashboard,
)


PHASE_MAP = {
    # Foundations (1-10)
    1:  ("phase1", None),
    2:  ("phase1", None),
    3:  ("phase1", None),
    4:  ("phase1", None),
    5:  ("phase1", None),
    6:  ("phase1", None),
    7:  ("phase1", None),
    8:  ("phase1", None),
    9:  ("phase1", None),
    10: ("phase1", None),
    # Data Readiness (11-20)
    11: ("phase2", None),
    12: ("phase2", None),
    13: ("phase2", None),
    14: ("phase2", None),
    15: ("phase2", None),
    16: ("phase2", None),
    17: ("phase2", None),
    18: ("phase2", None),
    19: ("phase2", None),
    20: ("phase2", None),
    # Modeling (21-30)
    21: ("phase3", None),
    22: ("phase3", None),
    23: ("phase3", None),
    24: ("phase3", None),
    25: ("phase3", None),
    26: ("phase3", None),
    27: ("phase3", None),
    28: ("phase3", None),
    29: ("phase3", None),
    30: ("phase3", None),
    # Production (31-40)
    31: ("phase4", None),
    32: ("phase4", None),
    33: ("phase4", None),
    34: ("phase4", None),
    35: ("phase4", None),
    36: ("phase4", None),
    37: ("phase4", None),
    38: ("phase4", None),
    39: ("phase4", None),
    40: ("phase4", None),
    # Advanced (41-50)
    41: ("phase5", None),
    42: ("phase5", None),
    43: ("phase5", None),
    44: ("phase5", None),
    45: ("phase5", None),
    46: ("phase5", None),
    47: ("phase5", None),
    48: ("phase5", None),
    49: ("phase5", None),
    50: ("phase5", None),
}

# Which module group each phase number belongs to
_GROUP_ORDER = ["phase1", "phase2", "phase3", "phase4", "phase5"]
_GROUP_MODULE = {
    "phase1": phase1_foundations,
    "phase2": phase2_data_readiness,
    "phase3": phase3_modeling,
    "phase4": phase4_production,
    "phase5": phase5_advanced,
}


def _groups_needed(phases: list[int]) -> list[str]:
    """Return the ordered list of module groups required by the selected phases."""
    needed = set(PHASE_MAP[p][0] for p in phases if p in PHASE_MAP)
    return [g for g in _GROUP_ORDER if g in needed]


def main():
    cfg, phases = parse_args()

    # ── Create output directories
    cfg.make_dirs()

    print("=" * 60)
    print("  DIABETES ML PIPELINE  —  OpenML 44214")
    print("=" * 60)
    print(f"  Output dir : {cfg.base_dir.resolve()}")
    print(f"  Figures    : {cfg.figures_dir.resolve()}")
    print(f"  Artifacts  : {cfg.artifacts_dir.resolve()}")
    print(f"  Models     : {cfg.models_dir.resolve()}")
    print(f"  Seed       : {cfg.random_seed}")
    print(f"  CV folds   : {cfg.cv_folds}")
    print(f"  Phases     : {sorted(phases)}")
    print("=" * 60)

    t0 = time.perf_counter()
    ctx = {}

    groups = _groups_needed(sorted(phases))

    # Phase groups run in order; each depends on the previous ctx
    # Phase 1 must always run first (provides data splits)
    if "phase1" not in groups:
        print("\n⚠  Phase 1 (data loading) is required for all other phases.")
        print("   Adding phase 1 automatically.\n")
        groups = ["phase1"] + groups

    for group in groups:
        try:
            if group == "phase1":
                ctx = _GROUP_MODULE[group].run(cfg)
            else:
                ctx = _GROUP_MODULE[group].run(cfg, ctx)
        except Exception as exc:
            print(f"\n❌ Error in {group}: {exc}")
            traceback.print_exc()
            print("   Continuing with next group…")

    # Summary dashboard — only if enough context is available
    required_for_dashboard = {
        "zoo_df", "gbr_cv", "gbr_preds", "best_name", "coverage",
        "test_preds", "cp_lower", "cp_upper", "qr_models",
        "errors_A", "errors_B", "p_val", "cv_scores", "ridge_cv",
    }
    if required_for_dashboard.issubset(ctx.keys()):
        try:
            summary_dashboard.run(cfg, ctx)
        except Exception as exc:
            print(f"\n⚠ Dashboard skipped: {exc}")
    else:
        missing = required_for_dashboard - ctx.keys()
        print(f"\nℹ  Summary dashboard skipped (need phases 1-5; missing: {missing})")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"  ✅  All selected phases complete in {elapsed:.1f}s")
    print(f"  📊  {len(list(cfg.figures_dir.glob('*.png')))} figures → {cfg.figures_dir.resolve()}")
    print(f"  💾  {len(list(cfg.artifacts_dir.glob('*')))} artifacts → {cfg.artifacts_dir.resolve()}")
    print(f"  🤖  {len(list(cfg.models_dir.glob('*')))} models    → {cfg.models_dir.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
