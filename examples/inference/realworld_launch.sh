#!/usr/bin/env bash
# Real-world DRP eval on robots-workstation1 in one command: read-only pre-flight checks, then the viser panel on
# robot-control-stack (drp_viser_app.py): FR3 through rcs_fr3's joint impedance controller, Robotiq 2F-85 through
# rcs_robotiq2f85, both front RealSense cameras (D455s, --camera-rig d415 for the previous pair) through rcs_realsense,
# one vlagents policy server per checkpoint in
# ~/drp/ckpt_active (starting on #1). Nothing moves until a button in the panel is pressed.
#
#   realworld_launch.sh                          # the full real-world eval
#   realworld_launch.sh --checkpoint NAME        # start on another checkpoint (the dropdown switches live)
#   realworld_launch.sh --noise front_left       # start with a view noised out (the panel's toggles stay)
#   realworld_launch.sh --blind                  # robot without cameras: noise in both views
#   realworld_launch.sh --camera-rig d415        # the previous D415 pair (default: the D455 pair at 640x480)
#   realworld_launch.sh --camera-crop 0.625      # D455 cropped to the D415's 55 deg field before 320x240
#   realworld_launch.sh --platform sim           # dry run in the RCS MuJoCo sim (no robot, cameras or gripper)
#
# Launcher options; everything else goes to drp_viser_app.py (--kp/--kd/--torque-limit, --port, --config, ...):
#   --preflight-only   run the checks and exit       --no-preflight   start without the checks
#   --no-browser       do not open the panel         --app-help       the panel's own options
#   -h, --help         this text
#
# Stop with Ctrl-C once and wait for "session ended": the panel brings the arm to rest, closes the robot, cameras and
# gripper, then stops the policy servers. Each session (options, versions, everything printed, the policy servers'
# logs) is saved in ~/drp/logs/realworld/<date>-<time>/.

set -uo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"   # examples/inference, also when started via a link
RCS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${RCS_PYTHON:-$RCS_DIR/.venv/bin/python}"
APP="$SCRIPT_DIR/drp_viser_app.py"
PREFLIGHT="$SCRIPT_DIR/realworld_preflight.py"
LOG_ROOT="${DRP_REALWORLD_LOG_DIR:-$HOME/drp/logs/realworld}"

usage() { sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$(readlink -f "${BASH_SOURCE[0]}")"; }

preflight=1
preflight_only=0
browser=1
app_args=()
for arg in "$@"; do
    case "$arg" in
        --preflight-only) preflight_only=1 ;;
        --no-preflight) preflight=0 ;;
        --no-browser) browser=0 ;;
        --app-help) exec "$PY" "$APP" --help ;;
        -h | --help) usage; exit 0 ;;
        *) app_args+=("$arg") ;;
    esac
done

for f in "$PY" "$APP" "$PREFLIGHT"; do
    [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 2; }
done

STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_DIR="$LOG_ROOT/$STAMP"
mkdir -p "$SESSION_DIR" || exit 2
LOG="$SESSION_DIR/session.log"

# appends to the log; ignores INT/HUP so that what the panel prints while it shuts down still reaches the log, and a
# closed terminal does not break the panel's output pipe while it is releasing the robot
log_tee() { (trap '' INT HUP; exec tee -a "$LOG"); }
git_rev() { git -C "$1" log -1 --format='%h %ad %s' --date=short 2> /dev/null || echo "?"; }
git_dirty() { [[ -n "$(git -C "$1" status --porcelain --untracked-files=no 2> /dev/null)" ]] && echo " +uncommitted changes"; }

{
    echo "== real-world DRP session $STAMP on $(hostname), user $USER"
    echo "command:  $0 $*"
    echo "panel:    $PY $APP --log-dir $SESSION_DIR ${app_args[*]:-}"
    echo "rcs:      $(git -C "$RCS_DIR" branch --show-current 2> /dev/null) $(git_rev "$RCS_DIR")$(git_dirty "$RCS_DIR")"
    echo "vlagents: $(git -C "$HOME/vlagents" branch --show-current 2> /dev/null) $(git_rev "$HOME/vlagents")$(git_dirty "$HOME/vlagents")"
    echo "impact:   $(git_rev "$HOME/drp/impact_eefgoal")$(git_dirty "$HOME/drp/impact_eefgoal")"
} | log_tee

if (( preflight || preflight_only )); then
    echo | log_tee
    "$PY" "$PREFLIGHT" "${app_args[@]}" 2>&1 | log_tee
    rc=${PIPESTATUS[0]}
    if (( rc != 0 )); then
        echo -e "\npre-flight did not pass: fix the FAIL lines above (or start with --no-preflight)" | log_tee
        echo "log: $LOG"
        exit "$rc"
    fi
    if (( preflight_only )); then
        echo "log: $LOG"
        exit 0
    fi
fi

# Open the panel in a browser once it serves: it prints "UI: http://localhost:<port>" when ready (viser moves off a
# taken port; 8080 is taken on this machine). setsid: a browser started here must not share our process group, or
# the Ctrl-C meant for the panel would close it too.
watcher=""
if (( browser )) && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open > /dev/null; then
    (
        for _ in $(seq 600); do   # robot, cameras and gripper come up in ~5-60 s
            port="$(grep -m1 -oP 'UI: http://localhost:\K[0-9]+' "$LOG" 2> /dev/null)"
            if [[ -n "$port" ]]; then
                setsid -f xdg-open "http://localhost:$port" > /dev/null 2>&1
                exit 0
            fi
            grep -q 'startup failed' "$LOG" 2> /dev/null && exit 0
            sleep 0.5
        done
    ) &
    watcher=$!
fi

# Ctrl-C reaches the panel directly (same foreground process group; the policy servers run in their own session and
# are stopped by the panel) and it unwinds by itself; the trap keeps bash waiting until it has released everything.
trap ':' INT HUP TERM
echo | log_tee
cd "$RCS_DIR" || exit 2
PYTHONUNBUFFERED=1 "$PY" "$APP" --log-dir "$SESSION_DIR" "${app_args[@]}" 2>&1 | log_tee
rc=${PIPESTATUS[0]}

[[ -n "$watcher" ]] && kill "$watcher" 2> /dev/null
echo "== session ended $(date '+%F %T'), panel exit code $rc" | log_tee
echo "log: $LOG"
exit "$rc"
