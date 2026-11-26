#!/usr/bin/env python3
"""Pre-flight checks for the real-world DRP panel (drp_viser_app.py) on robots-workstation1. Read-only: nothing moves.

realworld_launch.sh runs this before the panel (`realworld_launch.sh --preflight-only` runs only this). It takes the
panel's own options and ignores the ones it does not use. Exit code 0: ready (warnings do not block); 1: a FAIL.

The robot check opens the FCI once, reads one state (robot mode, Desk end effector, external torque) and closes it:
hw.Franka's constructor only sets collision thresholds and the guiding mode; no controller or motion is started.
"""

import argparse
import importlib.util
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
logging.getLogger("OpenGL").setLevel(logging.WARNING)

import franka  # noqa: E402
from drp_viser.envs import CAMERA_RIGS, DEFAULT_CAMERA_RIG  # noqa: E402
from drp_viser.policies import BASE_PORT, CKPT_ROOT, POLICY_PYTHON  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)

FCI_PORT = 1337  # libfranka's command port; Desk (443) may stay open
EE_MASS_RANGE = (0.7, 1.3)  # Robotiq 2F-85 with its coupling ~0.9 kg, configured in Desk
IMPACT_ROOT = Path.home() / "drp/impact_eefgoal"
MODULES = ("viser", "vlagents", "json_numpy", "rpyc", "pinocchio", "mujoco", "scipy", "pyrealsense2", "rcs_fr3",
           "rcs_realsense", "rcs_robotiq2f85")

problems, warnings = [], []


def ok(msg):
    print(f"  ok    {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  WARN  {msg}")


def fail(msg):
    problems.append(msg)
    print(f"  FAIL  {msg}")


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return "?"


def check_setup(args) -> list:
    print("checkpoints / code")
    root = Path(args.ckpt_root).expanduser()
    names = sorted(d.name for d in root.glob("*") if (d / "latest.pt").exists() and (d / "model_config.yaml").exists())
    (ok if names else fail)(f"{len(names)} checkpoints under {root}" if names else f"no checkpoints under {root}")
    if args.checkpoint and args.checkpoint not in names:
        fail(f"--checkpoint {args.checkpoint} is not under {root}")
    if not (root / "ckpt_info.json").exists():
        warn("no ckpt_info.json: the dropdown shows bare names")
    (ok if IMPACT_ROOT.is_dir() else fail)(f"model code {IMPACT_ROOT}" + ("" if IMPACT_ROOT.is_dir() else " missing"))
    (ok if POLICY_PYTHON.exists() else fail)(f"policy server python {POLICY_PYTHON}"
                                             + ("" if POLICY_PYTHON.exists() else " missing"))
    missing = [m for m in MODULES if importlib.util.find_spec(m) is None]
    if missing:
        fail(f"not importable in {sys.executable}: {', '.join(missing)} (use ~/robot-control-stack/.venv)")
    else:
        ok(f"{sys.executable}: {', '.join(MODULES)}")
    # the panel relies on patches in the forks; catch an old checkout early
    try:
        from rcs_realsense.camera import DRP_D415_KWARGS, DRP_D455_KWARGS  # noqa: F401
        from rcs_robotiq2f85.hw import RobotiQ2F85GripperConfig
        from vlagents.policies.drp import DrpAgent

        RobotiQ2F85GripperConfig("x", activate_on_init=False)
        assert hasattr(DrpAgent, "metadata")
        ok("RCS fork (rcs_realsense DRP capture, lazy Robotiq activation) and vlagents fork (drp meta query)")
    except Exception as exc:  # noqa: BLE001
        fail(f"fork patches missing ({type(exc).__name__}: {exc}): RCS branch workstation1-drp-deploy, vlagents drp-agent")
    return names


def check_conflicts(args, n_ckpts: int) -> None:
    print("other programs / ports")
    clean = True
    out = subprocess.run(["pgrep", "-f", r"python[0-9.]* .*drp_viser_app\.py"], capture_output=True, text=True).stdout
    for pid in (int(p) for p in out.split() if int(p) not in (os.getpid(), os.getppid())):
        clean = False
        fail(f"another DRP panel is running (pid {pid}: {cmdline(pid)[:100]}) -> stop it first")
    if args.platform == "hw":
        res = subprocess.run(["ss", "-Htnp", "state", "established", f"( dport = :{FCI_PORT} )", "dst", franka.ROBOT_IP],
                             capture_output=True, text=True)
        for line in res.stdout.splitlines():
            clean = False
            pids = sorted({int(p) for p in re.findall(r"pid=(\d+)", line)})
            who = "; ".join(f"pid {p}: {cmdline(p)[:90]}" for p in pids) or "a process of another user"
            fail(f"an FCI client is connected to {franka.ROBOT_IP} ({who}) -> stop it (one FCI client at a time)")
    if args.platform == "hw" and not args.blind:
        nodes = [f"/dev/{v.name}" for v in sorted(Path("/sys/class/video4linux").glob("video*"))
                 if "RealSense" in (v / "name").read_text(errors="replace")]
        if nodes and shutil.which("fuser"):
            res = subprocess.run(["fuser", *nodes], capture_output=True, text=True)
            for pid in sorted({int(p) for p in re.findall(r"\d+", res.stdout)} - {os.getpid()}):
                clean = False
                fail(f"pid {pid} holds a RealSense camera ({cmdline(pid)[:90]}) -> close it (realsense-viewer, ...)")
    taken = []
    for port in range(args.policy_port, args.policy_port + n_ckpts):
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                taken.append(port)
    if taken:
        warn(f"policy port(s) {taken} already serve something: the panel adopts them only if they serve the "
             f"expected checkpoint (left over from an earlier panel), otherwise that checkpoint fails")
    if clean:
        ok("no other panel, FCI client or camera user")


def check_gpu() -> None:
    print("GPU")
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
        name, free, total = (s.strip() for s in line.split(","))
        msg = f"{name}: {float(free) / 1024:.1f} of {float(total) / 1024:.1f} GiB free"
        (ok if float(free) > 12 * 1024 else warn)(msg + ("" if float(free) > 12 * 1024
                                                          else " (10 policy servers need ~9 GiB)"))
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired) as exc:
        fail(f"no GPU visible ({exc})")


def check_cameras(args) -> None:
    print("cameras")
    if args.platform != "hw" or args.blind:
        ok("skipped (--blind / sim)")
        return
    import pyrealsense2 as rs

    found = {}
    for dev in rs.context().query_devices():
        usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "?"
        found[dev.get_info(rs.camera_info.serial_number)] = (dev.get_info(rs.camera_info.name), usb)
    print(f"  ..    camera rig {args.camera_rig}")
    for slot, serial in CAMERA_RIGS[args.camera_rig]["serials"].items():
        if serial not in found:
            fail(f"{slot} camera {serial} not attached (attached: {sorted(found) or 'none'}) -> check its USB cable")
            continue
        name, usb = found[serial]
        (ok if usb.startswith("3") else warn)(f"{slot}: {name} {serial}, USB {usb}"
                                              + ("" if usb.startswith("3") else " -> not USB3: replug it"))


def check_gripper(args) -> None:
    print("gripper")
    if args.platform != "hw":
        ok("skipped (sim)")
        return
    ports = sorted(Path("/dev/serial/by-id").glob(f"*{franka.ROBOTIQ_SERIAL}*"))
    if not ports:
        fail(f"no serial adapter with serial {franka.ROBOTIQ_SERIAL} under /dev/serial/by-id -> check the RS-485 adapter")
        return
    real = os.path.realpath(ports[0])
    (ok if os.access(real, os.R_OK | os.W_OK) else fail)(
        f"Robotiq RS-485 adapter {ports[0].name} -> {real}"
        + ("" if os.access(real, os.R_OK | os.W_OK) else " not read/writable: sudo usermod -aG dialout $USER"))


def check_robot(args) -> None:
    print("robot")
    if args.platform != "hw":
        ok("skipped (sim)")
        return
    ip = franka.ROBOT_IP
    try:
        with socket.create_connection((ip, 443), timeout=2.0):
            pass
        ok(f"{ip} reachable")
    except OSError as exc:
        fail(f"{ip} not reachable ({exc}) -> robot powered on? cable to the control box?")
        return
    if any("FCI client is connected" in p for p in problems):
        print("  ..    state not read: another FCI client is connected")
        return
    from rcs_fr3._core import hw
    from rcs_fr3.configs import DefaultFR3HardwareEnv

    base = DefaultFR3HardwareEnv()
    base.ip = ip
    cfg = base.config().robot_cfg
    cfg.ignore_realtime = True
    robot = None
    try:
        robot = hw.Franka(cfg, None)  # connect only: collision thresholds and guiding mode, no motion
        st = robot.get_state().robot_state
        mode = st.robot_mode.name
        if mode in ("kIdle", "kMove"):
            ok(f"robot mode {mode[1:]}: FCI active, joints unlocked, user stop released")
        elif mode == "kReflex":
            warn("robot mode Reflex -> press 'Recover (clear Reflex)' in the panel before moving")
        elif mode == "kUserStopped":
            fail("robot mode UserStopped -> release the user stop")
        else:
            fail(f"robot mode {mode[1:]} -> in Desk (https://{ip}) unlock the joints and activate FCI")
        m_ee = st.m_ee + st.m_load
        msg = f"Desk end effector {st.m_ee:.2f} kg (+ payload {st.m_load:.2f} kg), flange->EE z {st.F_T_EE[14]:.3f} m"
        (ok if EE_MASS_RANGE[0] <= m_ee <= EE_MASS_RANGE[1] else fail)(
            msg if EE_MASS_RANGE[0] <= m_ee <= EE_MASS_RANGE[1] else
            msg + " -> set the Robotiq 2F-85 (~0.9 kg) as end effector in Desk: the impedance controller relies on "
                  "libfranka's gravity compensation of it")
        ext = max(abs(t) for t in st.tau_ext_hat_filtered)
        (ok if ext < 3.0 else warn)(f"external torque at rest {ext:.1f} Nm"
                                    + ("" if ext < 3.0 else ": something touches the arm, or the load in Desk is off"))
        ok("q = [" + ", ".join(f"{v:+.3f}" for v in st.q) + "]")
    except Exception as exc:  # noqa: BLE001
        fail(f"could not read the robot state: {type(exc).__name__}: {exc} -> Desk: joints unlocked, FCI active?")
    finally:
        del robot
        time.sleep(0.3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--platform", choices=["hw", "sim"], default="hw")
    p.add_argument("--ckpt-root", default=str(CKPT_ROOT))
    p.add_argument("--checkpoint", default="")
    p.add_argument("--policy-port", type=int, default=BASE_PORT)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--camera-rig", choices=sorted(CAMERA_RIGS), default=DEFAULT_CAMERA_RIG)
    args, _ = p.parse_known_args()
    print(f"pre-flight (read-only) for the DRP panel on {socket.gethostname()}, platform {args.platform}")
    names = check_setup(args)
    check_conflicts(args, len(names))
    check_gpu()
    check_cameras(args)
    check_gripper(args)
    check_robot(args)
    print()
    if problems:
        print(f"NOT READY: {len(problems)} problem(s), {len(warnings)} warning(s)")
        return 1
    print(f"READY ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
