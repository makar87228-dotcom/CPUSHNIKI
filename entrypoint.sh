#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  ping)
    cat <<'JSON'
{
  "protocol_version": "1.0",
  "supported_tasks": ["T2", "T3", "T4"],
  "self_trained_rl": true,
  "preferred_speed_mps": 0.55
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
  T3)
    set +u
    source /opt/ros/humble/setup.bash
    source /opt/unitree_go/share/unitree_go/local_setup.bash
    set -u
    python3 /home/appuser/ws/safety_filter.py &
    SAFETY_PID=$!
    python3 /home/appuser/ws/rl_node.py &
    RL_PID=$!
    trap 'kill -TERM "$SAFETY_PID" "$RL_PID" 2>/dev/null || true' INT TERM
    wait -n "$SAFETY_PID" "$RL_PID"
    EXIT_CODE=$?
    kill -TERM "$SAFETY_PID" "$RL_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    exit "$EXIT_CODE"
    ;;
  T4)
    set +u
    source /opt/ros/humble/setup.bash
    set -u
    exec python3 /home/appuser/ws/odom_node.py
    ;;
  T5)
    echo "Task $1 is not implemented yet" >&2
    exit 1
    ;;
  *)
    echo "Usage: /entrypoint.sh {ping|T2|T3|T4|T5}" >&2
    exit 2
    ;;
esac
