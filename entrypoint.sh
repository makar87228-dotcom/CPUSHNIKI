#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  ping)
    cat <<'JSON'
{
  "protocol_version": "1.0",
  "supported_tasks": ["T2"],
  "self_trained_rl": false,
  "preferred_speed_mps": null
}
JSON
    exit 0
    ;;
  T2)
    set +u
    source /opt/ros/humble/setup.bash
    source /opt/unitree_go/share/unitree_go/local_setup.bash
    set -u
    exec python3 /home/appuser/ws/safety_filter.py
    ;;
  T3|T4|T5)
    echo "Task $1 is not implemented yet" >&2
    exit 1
    ;;
  *)
    echo "Usage: /entrypoint.sh {ping|T2|T3|T4|T5}" >&2
    exit 2
    ;;
esac
