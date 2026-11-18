#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  ping)
    cat <<'JSON'
{
  "protocol_version": "1.0",
  "supported_tasks": [],
  "self_trained_rl": false,
  "preferred_speed_mps": null
}
JSON
    exit 0
    ;;
  T2|T3|T4|T5)
    echo "Task $1 is not implemented yet" >&2
    exit 1
    ;;
  *)
    echo "Usage: /entrypoint.sh {ping|T2|T3|T4|T5}" >&2
    exit 2
    ;;
esac
