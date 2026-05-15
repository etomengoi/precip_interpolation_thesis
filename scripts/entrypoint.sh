#!/usr/bin/env bash
set -euo pipefail

cd /app

export PATH="/app/.venv/bin:$PATH"

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

# ── Sync code from git (overrides whatever was baked into image) ──────────────
# Image carries deps + last-known-good code. Runtime git pull = latest from main
# without rebuilding image. Override branch with GIT_BRANCH env var.
# Set SKIP_GIT_SYNC=1 to use only the code baked into the image (offline / debug).
# ─────────────────────────────────────────────────────────────────────────────
if [ "${SKIP_GIT_SYNC:-0}" != "1" ]; then
  REPO_URL="https://github.com/etomengoi/precip_interpolation_thesis.git"
  REPO_BRANCH="${GIT_BRANCH:-main}"

  if [ ! -d ".git" ]; then
    echo "[entrypoint] Initialising git repo for runtime code sync..."
    git init -q
    git remote add origin "$REPO_URL"
  fi

  echo "[entrypoint] Pulling latest code from $REPO_BRANCH..."
  git fetch origin "$REPO_BRANCH" --quiet
  git reset --hard "origin/$REPO_BRANCH"
  echo "[entrypoint] Code synced to commit: $(git rev-parse --short HEAD)"

  # Reinstall package in editable mode (idempotent if pyproject unchanged).
  uv pip install -e . --quiet 2>&1 | tail -3 || \
    echo "[entrypoint] WARNING: editable install failed — using image-baked deps"
fi

: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID is not set}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY is not set}"
: "${AWS_DEFAULT_REGION:=eu-central-1}"

aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "$AWS_DEFAULT_REGION"

mkdir -p /app/data/dem
mkdir -p /app/data/rekis
mkdir -p /app/data/soilgrids
mkdir -p /app/outputs
mkdir -p /app/outputs/grk

if [ "${SYNC_DATA_FROM_S3:-1}" = "1" ]; then
  echo "Sync DEM_data -> /app/data/dem"
  aws s3 sync "s3://thesis-data-ismaktam/data/dem" "/app/data/dem"

  echo "Sync ReKis -> /app/data/rekis"
  aws s3 sync "s3://thesis-data-ismaktam/data/rekis" "/app/data/rekis"

  echo "Sync soilgrids -> /app/data/soilgrids"
  aws s3 sync "s3://thesis-data-ismaktam/data/soilgrids" "/app/data/soilgrids"
fi

# Pull cached features + tuned hparams so grk_kfold_cv reuses them.
if [ "${SYNC_GRK_RESULTS:-1}" = "1" ]; then
  echo "Sync GRK results -> /app/outputs/grk"
  aws s3 sync "s3://thesis-data-ismaktam/results/grk" "/app/outputs/grk"
fi

# ── Logging setup ─────────────────────────────────────────────────────────────
# Log file: /app/outputs/logs/YYYYMMDD_HHMMSS_<instance>.log
# S3 path:  s3://thesis-data-ismaktam/logs/YYYYMMDD_HHMMSS_<instance>.log
#
# Background syncer uploads a partial log every 5 minutes so long-running
# jobs can be monitored without waiting for completion.
# trap EXIT uploads the final log on success or failure.
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR="/app/outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
INSTANCE_ID="${VAST_CONTAINERLABEL:-$(hostname)}"
LOG_NAME="${TIMESTAMP}_${INSTANCE_ID}.log"
LOG_FILE="${LOG_DIR}/${LOG_NAME}"
S3_LOG="s3://thesis-data-ismaktam/logs/${LOG_NAME}"

echo "[entrypoint] Log file : $LOG_FILE"
echo "[entrypoint] S3 target: $S3_LOG"

# Background syncer — partial upload every 5 minutes
_sync_log() {
    while true; do
        sleep 300
        aws s3 cp "$LOG_FILE" "$S3_LOG" --quiet 2>/dev/null || true
    done
}
_sync_log &
_SYNC_PID=$!

# Final upload on any exit (success, error, OOM kill via SIGTERM)
_on_exit() {
    kill "$_SYNC_PID" 2>/dev/null || true
    echo "[entrypoint] Uploading final log -> $S3_LOG"
    aws s3 cp "$LOG_FILE" "$S3_LOG" || echo "[entrypoint] WARNING: final log upload failed"
    if [ "${AUTO_DESTROY:-0}" = "1" ] && [ -n "${VAST_CONTAINERLABEL:-}" ]; then
        # VAST_CONTAINERLABEL is "C.<id>"; CLI wants the bare integer.
        INSTANCE_ID="${VAST_CONTAINERLABEL#C.}"
        echo "[entrypoint] Auto-destroying instance $INSTANCE_ID..."
        vastai destroy instance "$INSTANCE_ID" || true
    fi
}
trap _on_exit EXIT

# ── Run command, tee stdout+stderr to log file ────────────────────────────────
echo "Starting command..."
if [ "$#" -eq 0 ]; then
    python -m thesis.scripts.run_task 2>&1 | tee "$LOG_FILE"
else
    exec "$@" 2>&1 | tee "$LOG_FILE"
fi
