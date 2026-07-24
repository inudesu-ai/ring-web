#!/usr/bin/env bash
set -Eeuo pipefail
cd "${HOME}/ring-web"
if [[ -f .ring-live.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .ring-live.env
  set +a
fi
source .venv/bin/activate
MODEL="${RING_MODEL:-ml/results/temporal-cnn-episodic-1m/models/gesture-temporal-cnn-sim-1m.npz}"
PUBLISH_URL="${RING_PUBLISH_URL:-http://127.0.0.1:3000/v1/gesture}"
TELEMETRY_HZ="${RING_TELEMETRY_HZ:-10}"
RING_NAME="${RING_NAME:-ring}"
TARGET_ARGS=(--name "$RING_NAME")
if [[ -n "${RING_ADDRESS:-}" ]]; then
  TARGET_ARGS=(--address "$RING_ADDRESS")
fi
CPUID_ARGS=()
if [[ -n "${RING_CPUID:-}" ]]; then
  CPUID_ARGS=(--cpuid "$RING_CPUID")
fi
printf "[%s] ring live supervisor start target=%s model=%s publish=%s min_rssi=%s\n" "$(date -Is)" "${TARGET_ARGS[*]}" "$MODEL" "$PUBLISH_URL" "${RING_NAME_MIN_RSSI_DBM:--95}" >&2
while true; do
  printf "[%s] starting realtime_infer target=%s cpuid=%s\n" "$(date -Is)" "${TARGET_ARGS[*]}" "${RING_CPUID:-<none>}" >&2
  set +e
  python3 ml/realtime_infer.py \
    "${TARGET_ARGS[@]}" \
    "${CPUID_ARGS[@]}" \
    --model "$MODEL" \
    --robot-commands \
    --publish-url "$PUBLISH_URL" \
    --telemetry-hz "$TELEMETRY_HZ"
  rc=$?
  set -e
  printf "[%s] realtime_infer exited rc=%s; retrying in 2s\n" "$(date -Is)" "$rc" >&2
  sleep 2
done
