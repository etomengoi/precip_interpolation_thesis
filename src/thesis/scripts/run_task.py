"""Default entry-point called by Docker entrypoint.sh.

Dispatches to the appropriate pipeline script based on the TASK env var.
Default: run_cv (cross-validation).
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    task = os.environ.get("TASK", "cv")

    if task == "cv":
        from thesis.scripts.run_cv import main as run_cv
        run_cv()
    elif task == "variogram":
        from thesis.scripts.run_variogram import main as run_variogram
        run_variogram()
    elif task == "monthly_quota_grid":
        from thesis.scripts.build_monthly_grids import main as build_monthly_grids
        build_monthly_grids()
    elif task == "dem":
        from thesis.scripts.run_dem import main as run_dem
        run_dem()
    elif task == "viz_day":
        from thesis.scripts.run_viz_day import main as run_viz_day
        run_viz_day()
    elif task == "grk_hparam":
        from thesis.scripts.run_grk_hparam_search import main as run_grk_hparam
        run_grk_hparam()
    elif task == "grk_lgb_hparam":
        from thesis.scripts.run_grk_lgb_hparam_search import main as run_grk_lgb_hparam
        run_grk_lgb_hparam()
    elif task == "grk_kfold_cv":
        from thesis.scripts.run_grk_kfold_cv import main as run_grk_kfold_cv
        run_grk_kfold_cv()
    else:
        print(f"Unknown TASK={task!r}. Available: cv, variogram, monthly_quota_grid, dem, viz_day, grk_hparam, grk_lgb_hparam, grk_kfold_cv")
        sys.exit(1)


if __name__ == "__main__":
    main()
