"""Upload local outputs to S3 with structured folder layout."""
from __future__ import annotations

import os
from pathlib import Path

from thesis.scripts._common import S3_BUCKET


def sync_to_s3(local_dir: str | Path, s3_prefix: str) -> None:
    """Use ``aws s3 sync`` for efficient incremental uploads."""
    local_dir = Path(local_dir)
    cmd = f'aws s3 sync "{local_dir}" "s3://{S3_BUCKET}/{s3_prefix}" --quiet'
    print(f"  sync: {local_dir} → s3://{S3_BUCKET}/{s3_prefix}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"aws s3 sync failed with code {rc}")
