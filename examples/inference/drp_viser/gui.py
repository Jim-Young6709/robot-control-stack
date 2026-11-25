"""The viser panel: same folders, controls and behaviour as the DRP panel of the previous robot PC, wired to the
RCS-based `PanelRunner` (robot, gripper and cameras through the RCS env) and `PolicyPool` (one vlagents server per
checkpoint).

UI callbacks never touch the env: they only change the runner's state, goal and settings. Everything that moves the
robot or the gripper happens on the runner's single worker thread, which also owns the env.
"""

import logging
import threading
import time

import numpy as np

from .depth import DepthMask, colorize
from .runner import DEFAULT_HOME, HOME_POSES, PanelRunner, State

logger = logging.getLogger("drp_viser")

JOINT_NAMES = ["J1 shoulder pan", "J2 shoulder lift", "J3 upper-arm roll", "J4 elbow", "J5 forearm roll",
               "J6 wrist flex", "J7 wrist roll"]
ALT_OFF_LABEL, ALT_ON_LABEL = "Toggle between two goals: OFF", "Toggle between two goals: ON"
LOOP_START_LABEL, LOOP_STOP_LABEL = "Start closed-loop rollout", "Stop closed-loop rollout"
WIDTH, HEIGHT = 320, 240

#: camera-side depth post-processing steps (rcs_realsense DepthProcessing + emitter), in pipeline order
CAMERA_DEPTH_STEPS = [
    ("emitter", "IR emitter",
     "The D415's IR dot projector: gives stereo matching texture on plain surfaces such as the table. OFF: depth only "
     "where the scene itself has texture. A capture setting of the camera."),
    ("disparity", "Disparity domain",
     "Run the spatial and temporal filters on disparity (1/depth) instead of depth, as drp_deploy does, so far "
     "surfaces are not over-smoothed. No effect while both filters are off."),
    ("spatial", "Spatial filter",
     "Edge-preserving smoothing within each frame (librealsense defaults: magnitude 2, alpha 0.5, delta 20, no hole "
     "fill)."),
    ("temporal", "Temporal filter",
     "Smoothing over frames (alpha 0.4, delta 20) with persistence: a pixel valid in 2 of the last 4 frames keeps its "
     "last value when it drops out. Lags fast motion a little; restarts with a fresh history when switched on."),
    ("hole_filling", "Hole filling",
     "Fills every pixel without depth from its farthest valid neighbour (librealsense 'farthest from around'): "
     "stereo shadows beside objects get the background's depth instead of 0. The range gate / table mask then cut "
     "far fills again."),
]


def build_gui(server, runner: PanelRunner, pool, scene, kin, active_ckpt: str) -> threading.Event:
    """Builds the panel and starts its refresh threads. Returns the event that stops them."""
    stop_ui = threading.Event()
    cfg = runner.config

    # ---------------------------------------------------------------------------------------------- status
    with server.gui.add_folder("Status"):
        txt_state = server.gui.add_text("State", initial_value="idle", disabled=True)
        txt_mode = server.gui.add_text("Robot mode", initial_value="?", disabled=True)
        txt_dist = server.gui.add_text("Dist to goal", initial_value="-", disabled=True)
        txt_q = server.gui.add_text("q [rad]", initial_value="-", disabled=True, multiline=True)
        txt_stats = server.gui.add_text("Stats", initial_value="-", disabled=True, multiline=True)
        txt_msg = server.gui.add_text("Last action", initial_value="ready", disabled=True, multiline=True)

    def say(msg: str) -> None:
        logger.info("%s", msg)
        txt_msg.value = msg

    # ------------------------------------------------------------------------------------------ checkpoint
    names = pool.names
    labels = {n: pool.label(n) for n in names}
    by_label = {v: k for k, v in labels.items()}
    with server.gui.add_folder("Checkpoint"):
        dd_ckpt = server.gui.add_dropdown(
            "Model", options=[labels[n] for n in names], initial_value=labels[active_ckpt],
            hint="Switching stops any running rollout. Every checkpoint has its own vlagents policy server, all "
                 "started at launch, so switching is instant once its server is up.")
        txt_ckpt = server.gui.add_text("Loaded", initial_value=pool.describe(active_ckpt), disabled=True)
        txt_ckpt_note = server.gui.add_text(
            "Notes", initial_value=pool.note(active_ckpt), disabled=True, multiline=True,
            hint="Role in this eval and the dashboard's sim scores, from ckpt_info.json in the checkpoint directory.")
    switch_lock = threading.Lock()
    active = {"name": active_ckpt}

    @dd_ckpt.on_update
    def _(_) -> None:
        if by_label.get(dd_ckpt.value) == active["name"]:
            return  # also swallows our own revert below, which re-fires this callback
        with switch_lock:
            name = by_label.get(dd_ckpt.value)
            if name is None or name == active["name"]:
                return
            if runner.state is State.RUNNING:
                runner.stop()
                say("stopped the rollout to switch checkpoint")
            error = runner.set_policy(name)
            if error:
                dd_ckpt.value = labels[active["name"]]
                txt_ckpt.value = f"{error} | still active: {active['name']}"
                say(txt_ckpt.value)
                return
            active["name"] = name
            txt_ckpt.value = pool.describe(name)
            txt_ckpt_note.value = pool.note(name)
            say(f"checkpoint -> {pool.describe(name)}")

    # ---------------------------------------------------------------------------------------------- policy
    with server.gui.add_folder("Policy"):
        btn_activate = server.gui.add_button("Activate policy", color="green",
                                             hint="Start execution toward the goal (bounded run)")
        btn_stop = server.gui.add_button(
            "Stop", color="red",
            hint="End the run. The arm stays where it is: the joint impedance controller holds the measured pose.")
        btn_loop = server.gui.add_button(
            LOOP_START_LABEL, color="green",
            hint="Toggle. Starting it makes the policy re-infer from live camera frames continuously until it "
                 "reaches the goal. Click again to stop early.")
        sld_steps = server.gui.add_slider(
            "Exec steps / chunk", min=1, max=10, step=1, initial_value=cfg.exec_steps,
            hint="Steps of each predicted chunk to execute before re-planning (one step per exec period). The sim "
                 "eval executes 1.")
        sld_rate = server.gui.add_slider(
            "Exec rate [Hz]", min=2.0, max=float(runner.control_hz), step=1.0, initial_value=cfg.exec_rate_hz,
            hint=f"How often the policy is asked for a new waypoint (up to the control rate). The RCS control "
                 f"loop runs at {runner.control_hz:.0f} Hz and tracks the latest waypoint within the velocity / "
                 f"acceleration limits. The cameras deliver 30 or 60 fps, so faster rates re-plan from fresh joint "
                 f"states but partly the same frames. The actual rates are shown in the Cameras tab.")
        sld_scale = server.gui.add_slider(
            "Action scale", min=0.5, max=3.0, step=0.1, initial_value=cfg.action_scale,
            hint="Multiplies the policy's step (each chunk waypoint minus the joints it was given) before shaping. "
                 "1.0 = as predicted. >1 points the arm further along the predicted motion, so it moves faster; the "
                 "jump cap, velocity and acceleration limits below still apply. Outside what the policy was trained "
                 "on, so raise it gradually.")
        sld_jump = server.gui.add_slider(
            "Max waypoint jump [rad]", min=0.05, max=0.60, step=0.05, initial_value=cfg.max_waypoint_jump,
            hint="Clamps how far one target may sit from the measured pose; a wild prediction is scaled down.")
        sld_vel = server.gui.add_slider(
            "Stream max vel [rad/s]", min=0.05, max=1.5, step=0.05, initial_value=cfg.max_joint_speed,
            hint="Velocity limit of the commanded joint target, per joint. Immediate.")
        sld_acc = server.gui.add_slider(
            "Stream max accel [rad/s^2]", min=0.5, max=10.0, step=0.5, initial_value=cfg.max_joint_accel,
            hint="Acceleration limit of the commanded joint target. It also sets how early the target slows down "
                 "before each waypoint (stop distance v^2/2a): with the policy's short steps this, more than the "
                 "velocity limit, decides the speed. Also used to come to rest on Stop.")
        sld_ee_tol_cm = server.gui.add_slider(
            "EE goal tol [cm]", min=0.5, max=5.0, step=0.5, initial_value=cfg.goal_tolerance_m * 100.0,
            hint="Goal-pose checkpoints (goal eef) only: the run ends when the flange is this close to its pose at "
                 "the goal joints. Joint-goal checkpoints use 0.05 rad per joint. The sim eval counts 5 cm.")
        sld_ee_tol_deg = server.gui.add_slider(
            "EE goal tol [deg]", min=1.0, max=20.0, step=1.0, initial_value=cfg.goal_tolerance_deg,
            hint="... and its orientation within this. The sim eval counts 15 deg.")

    @btn_activate.on_click
    def _(_) -> None:
        if runner.state is not State.RUNNING:
            cfg.continuous = False
        say(f"activate: {runner.activate()}")

    @btn_stop.on_click
    def _(_) -> None:
        cfg.continuous = False
        say(f"stop: {runner.stop()}")

    @btn_loop.on_click
    def _(_) -> None:
        if runner.state is State.RUNNING:
            cfg.continuous = False
            say(f"closed-loop stopped: {runner.stop()}")
            return
        seeded = ""
        if runner.goal is None:
            q = runner.snapshot()["q"]
            if q is not None:
                runner.set_goal(q)
                seeded = (" (goal seeded from the current pose - drag the gizmo or recall a saved goal to give it "
                          "somewhere to go)")
        cfg.continuous = True
        say(f"closed-loop rollout: {runner.activate()}{seeded}")

    sld_steps.on_update(lambda _: setattr(cfg, "exec_steps", int(sld_steps.value)))
    sld_rate.on_update(lambda _: setattr(cfg, "exec_rate_hz", float(sld_rate.value)))
    sld_jump.on_update(lambda _: setattr(cfg, "max_waypoint_jump", float(sld_jump.value)))
    sld_vel.on_update(lambda _: setattr(cfg, "max_joint_speed", float(sld_vel.value)))
    sld_acc.on_update(lambda _: setattr(cfg, "max_joint_accel", float(sld_acc.value)))
    sld_scale.on_update(lambda _: setattr(cfg, "action_scale", float(sld_scale.value)))
    sld_ee_tol_cm.on_update(lambda _: setattr(cfg, "goal_tolerance_m", float(sld_ee_tol_cm.value) / 100.0))
    sld_ee_tol_deg.on_update(lambda _: setattr(cfg, "goal_tolerance_deg", float(sld_ee_tol_deg.value)))

    # ------------------------------------------------------------------------------------ EE gizmo + IK goal
    with server.gui.add_folder("End-effector target (drag the gizmo)"):
        chk_ik = server.gui.add_checkbox("Follow the gizmo", initial_value=False,
                                         hint="While ticked, dragging the gizmo re-solves IK and updates the goal live.")
        chk_ik_rot = server.gui.add_checkbox("Match orientation too", initial_value=True,
                                             hint="Untick to solve position only, which is far easier to reach.")
        sld_null = server.gui.add_slider("Null-space pull to home", min=0.0, max=1.0, step=0.05, initial_value=0.3,
                                         hint="How hard to bias the redundant joint toward the home pose.")
        btn_ik_reset = server.gui.add_button("Snap gizmo to current EE",
                                             hint="Move the gizmo back to where the flange actually is")
        txt_ik = server.gui.add_text("IK", initial_value="-", disabled=True)

    q0 = runner.snapshot()["q"]
    T0 = kin.flange_pose(q0 if q0 is not None else HOME_POSES[DEFAULT_HOME])
    gizmo = server.scene.add_transform_controls("/ik_target", scale=0.25, position=tuple(T0[:3, 3]),
                                                wxyz=kin.wxyz(T0[:3, :3]))

    def solve_from_gizmo() -> None:
        if not chk_ik.value:
            return
        q_now = runner.snapshot()["q"]
        q_init = q_now if q_now is not None else HOME_POSES[DEFAULT_HOME]
        target_R = kin.matrix(gizmo.wxyz) if chk_ik_rot.value else None
        q_sol, ok = kin.ik(np.array(gizmo.position, dtype=np.float64), target_R, q_init=q_init,
                           q_nullspace=HOME_POSES[DEFAULT_HOME], nullspace_gain=float(sld_null.value))
        pos_err = float(np.linalg.norm(kin.flange_pose(q_sol)[:3, 3] - np.array(gizmo.position)))
        txt_ik.value = (f"{'solved' if ok else 'NOT reachable'}  pos err {pos_err * 1000:.1f} mm  "
                        f"|q-home| {np.linalg.norm(q_sol - HOME_POSES[DEFAULT_HOME]):.2f}")
        if not ok and pos_err > 0.02:
            return  # refuse to chase an unreachable target
        runner.set_goal(q_sol)

    gizmo.on_update(lambda _: solve_from_gizmo())
    chk_ik_rot.on_update(lambda _: solve_from_gizmo())

    @chk_ik.on_update
    def _(_) -> None:
        if chk_ik.value:
            solve_from_gizmo()
        else:
            txt_ik.value = "not following"

    @btn_ik_reset.on_click
    def _(_) -> None:
        q_now = runner.snapshot()["q"]
        if q_now is None:
            return
        T = kin.flange_pose(q_now)
        gizmo.position = tuple(T[:3, 3])
        gizmo.wxyz = kin.wxyz(T[:3, :3])
        say("gizmo snapped to the current flange pose")

    # ---------------------------------------------------------------------------------------- saved goals
    with server.gui.add_folder("Saved goals"):
        btn_save_a = server.gui.add_button("Save -> Goal A", hint="Stores the active goal. Turns green once the slot "
                                                                  "holds one.")
        btn_save_b = server.gui.add_button("Save -> Goal B", hint="Stores the active goal. Turns green once the slot "
                                                                  "holds one.")
        btn_go_a = server.gui.add_button("Go to Goal A", color="blue")
        btn_go_b = server.gui.add_button("Go to Goal B", color="blue")
        btn_alt = server.gui.add_button(
            ALT_OFF_LABEL, color="teal",
            hint="Ping-pong: on reaching one saved goal the other becomes the target immediately and the rollout "
                 "keeps going. Needs both slots saved.")
        txt_slots = server.gui.add_text("Slots", initial_value="A: - | B: -", disabled=True)

    def save_slot(slot: str) -> None:
        q, source = runner.goal, "current goal"
        if q is None:
            q, source = runner.snapshot()["q"], "current arm pose"
        if q is None:
            say("no goal and no arm pose yet")
            return
        q = runner.save_goal_slot(slot, q)
        say(f"Goal {slot} saved from the {source}: {np.round(q, 3).tolist()}")

    btn_save_a.on_click(lambda _: save_slot("A"))
    btn_save_b.on_click(lambda _: save_slot("B"))
    btn_go_a.on_click(lambda _: say(f"Goal A: {runner.use_goal_slot('A')}"))
    btn_go_b.on_click(lambda _: say(f"Goal B: {runner.use_goal_slot('B')}"))

    @btn_alt.on_click
    def _(_) -> None:
        if cfg.alternate_goals:
            cfg.alternate_goals = False
            say("goal alternation OFF")
            return
        missing = [k for k, v in runner.goal_slots.items() if v is None]
        if missing:
            say(f"alternation needs both slots; {', '.join(missing)} empty")
            return
        cfg.alternate_goals = True
        if runner.active_slot is None:
            say(f"Goal A: {runner.use_goal_slot('A')}")
        say("goal alternation ON - A <-> B on each arrival")

    # ----------------------------------------------------------------------------------------------- home
    with server.gui.add_folder("Home"):
        dd_home = server.gui.add_dropdown("Home pose", options=list(HOME_POSES), initial_value=DEFAULT_HOME,
                                          hint="RCS FR3 home (franka.py's home), the previous panel's DROID reset "
                                               "pose, or the libfranka ready pose")
        sld_home_speed = server.gui.add_slider("Home speed [rad/s]", min=0.05, max=0.5, step=0.01,
                                               initial_value=cfg.home_max_speed,
                                               hint="Peak joint speed of the homing trajectory")
        btn_home = server.gui.add_button(
            "Reset to home pose", color="orange",
            hint="Stops the policy, then a smooth trajectory to the home pose through the joint impedance controller")
        btn_recover = server.gui.add_button(
            "Recover (clear Reflex)", color="grape",
            hint="A latched reflex or a released user stop blocks all motion. This runs the FR3's automatic error "
                 "recovery. It commands no motion.")

    @btn_recover.on_click
    def _(_) -> None:
        before = runner.snapshot()["robot_mode"]
        result = runner.clear_error()
        say(f"recover: {before} -> {runner.snapshot()['robot_mode']} ({result})")

    @btn_home.on_click
    def _(_) -> None:
        cfg.home_max_speed = float(sld_home_speed.value)
        say(f"reset to {dd_home.value}: {runner.reset_home(HOME_POSES[dd_home.value])}")

    # ------------------------------------------------------------------------------------------- gripper
    with server.gui.add_folder("Gripper (Robotiq 2F-85)"):
        btn_grip_activate = server.gui.add_button("Activate gripper",
                                                  hint="Required once per power cycle before it will move")
        closure0 = runner.snapshot()["gripper"]["closure"]
        sld_grip = server.gui.add_slider("Closure  0=open 1=closed", min=0.0, max=1.0, step=0.01,
                                         initial_value=float(round(closure0, 2)) if closure0 is not None else 0.0)
        sld_grip_force = server.gui.add_slider("Force", min=0, max=255, step=5, initial_value=100)
        sld_grip_speed = server.gui.add_slider("Speed", min=0, max=255, step=5, initial_value=255)
        txt_grip = server.gui.add_text("Gripper", initial_value="not activated", disabled=True)

    btn_grip_activate.on_click(lambda _: say(f"gripper: {runner.gripper_activate()}"))

    def gripper_changed(_) -> None:
        # only records the target; the worker sends it with its next step (rate bounded there)
        runner.set_gripper(float(sld_grip.value), speed=int(sld_grip_speed.value), force=int(sld_grip_force.value))

    sld_grip.on_update(gripper_changed)
    sld_grip_force.on_update(gripper_changed)
    sld_grip_speed.on_update(gripper_changed)

    # -------------------------------------------------------------------------------- camera views (left)
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cam_panel = server.gui.add_panel()
    with cam_panel.add_tab("Cameras"):
        txt_rates = server.gui.add_text(
            "Rates", initial_value="-", disabled=True, multiline=True,
            hint="Measured: each camera stream's delivery rate (from the device timestamps of new frames), the policy "
                 "rate (inference calls per second while running), the control loop rate, and this view's refresh.")
        views = {}
        for slot in runner.camera_slots:
            views[f"{slot}_rgb"] = server.gui.add_image(blank, label=slot, format="jpeg", jpeg_quality=70,
                                                        visible=False)
            views[f"{slot}_depth"] = server.gui.add_image(blank, label=f"{slot} depth", format="jpeg",
                                                          jpeg_quality=70, visible=False)
        txt_frames = server.gui.add_text("Frames", initial_value="-", disabled=True, multiline=True)
        noise_buttons = {
            slot: server.gui.add_button(
                f"Noise out {slot}: OFF", color="gray",
                hint=f"Replace {slot} (RGB and depth) with random noise before the policy sees it. Ablate one at a "
                     f"time to tell a policy that leans on a single view from one that ignores vision entirely; "
                     f"turn both on for the full blind test.")
            for slot in runner.camera_slots
        }
    with cam_panel.add_tab("Depth processing"):
        txt_chain = server.gui.add_text("Pipeline", initial_value="-", disabled=True, multiline=True,
                                        hint="The depth the policy gets, in order. Every switch applies from the next "
                                             "frame, also during a rollout; the camera views show the result.")
        server.gui.add_markdown("**Camera** (rcs_realsense: 848x480 depth aligned onto the 320x240 colour, then "
                                "per frame)")
        depth_buttons = {
            key: server.gui.add_button(f"{label}: ?", color="gray", hint=hint, disabled=not runner.has_realsense)
            for key, label, hint in CAMERA_DEPTH_STEPS
        }
        server.gui.add_markdown("**Panel** (before the policy; noise ablation is in the Cameras tab)")
        btn_gate = server.gui.add_button(
            "Range gate 0.01-3 m: ON", color="green",
            hint="The renderer's clipping range: depth outside 0.01-3 m becomes 0 ('nothing hit'), here and in the "
                 "policy server. OFF passes the camera's whole range to the policy: an out-of-distribution ablation, "
                 "the checkpoints were trained on clipped renders.")
        btn_mask = server.gui.add_button(
            "Table mask: OFF", color="gray",
            hint="Zero depth below the table, more than 1.6 m above it, or further outside its outline than the "
                 "margin below (the renders contain only table, objects and arm). Needs 'Fit table plane' first; "
                 "applies to the cameras that have a fitted plane.")
        btn_fit_plane = server.gui.add_button(
            "Fit table plane",
            hint="RANSAC on each camera's current depth, while idle. Also reports the camera's height above the table "
                 "and its pitch against the renderer's (0.9-1.4 m, ~27 deg). Refit after moving a camera.")
        num_margin = server.gui.add_number(
            "Outline margin [m]", initial_value=runner.depth_mask.footprint_margin_m, min=0.0, max=1.0, step=0.05,
            hint="How far outside the table's visible outline depth is still kept. From the rear cameras the arm's "
                 "tall links stand outside the visible table.")
        txt_plane = server.gui.add_text("Table plane", initial_value="not fitted", disabled=True, multiline=True)
        btn_depth_defaults = server.gui.add_button(
            "Restore defaults", color="blue",
            hint="Emitter, disparity domain, spatial and temporal filter, hole filling and range gate ON (the "
                 "drp_deploy chain); table mask OFF.")
    cam_panel.dock_left()
    cam_panel.set_width(360)

    def toggle_noise(slot: str):
        def handler(_) -> None:
            slots = cfg.noise_slots
            slots.discard(slot) if slot in slots else slots.add(slot)
            on = sorted(slots)
            say(f"noise ablation: {', '.join(on) if on else 'none'}"
                + (" (fully blind)" if len(on) == len(runner.camera_slots) else ""))
        return handler

    for _slot, _btn in noise_buttons.items():
        _btn.on_click(toggle_noise(_slot))

    @btn_fit_plane.on_click
    def _(_) -> None:
        snap = runner.camera_snapshot()
        if runner.state is State.RUNNING or snap is None:
            say("fit the plane while idle, once the camera preview is live")
            return
        lines = []
        for slot in runner.camera_slots:
            depth, K = snap["raw"].get(f"{slot}_depth"), snap["intrinsics"].get(slot)
            if depth is None or K is None:
                lines.append(f"{slot}: no depth")
                continue
            plane = runner.depth_mask.fit(slot, depth, K)
            lines.append(f"{slot}: {plane.describe() if plane else 'no table plane'}")
        txt_plane.value = "\n".join(lines)
        say("table plane fitted" + ("" if runner.depth_mask.planes else ": none found"))

    def toggle_camera_step(key: str, label: str):
        def handler(_) -> None:
            state = runner.depth_processing()
            if not state:
                say("no RealSense cameras in this session (sim / --blind)")
                return
            on = not bool(state.get(key))
            if key == "emitter":
                runner.set_emitter(on)
            else:
                runner.set_depth_processing(**{key: on})
            say(f"depth: {label} {'ON' if on else 'OFF'} | camera chain: {runner.depth_processing().get('chain')}")
        return handler

    for _key, _label, _ in CAMERA_DEPTH_STEPS:
        depth_buttons[_key].on_click(toggle_camera_step(_key, _label))

    @btn_gate.on_click
    def _(_) -> None:
        runner.depth_mask.use_range_gate = not runner.depth_mask.use_range_gate
        say(f"depth range gate 0.01-3 m {'ON' if runner.depth_mask.use_range_gate else 'OFF (ablation)'}")

    @btn_mask.on_click
    def _(_) -> None:
        runner.depth_mask.use_plane = not runner.depth_mask.use_plane
        say(f"table mask {'ON' if runner.depth_mask.use_plane else 'OFF'}"
            + ("" if runner.depth_mask.planes else " (no plane fitted yet)"))

    @btn_depth_defaults.on_click
    def _(_) -> None:
        if runner.has_realsense:
            runner.set_emitter(True)
            runner.set_depth_processing(disparity=True, spatial=True, temporal=True, hole_filling=True)
        runner.depth_mask.use_range_gate = True
        runner.depth_mask.use_plane = False
        say("depth processing restored to the defaults")

    def refresh_depth_controls() -> None:
        """Labels follow the actual state (also after Restore defaults or a change elsewhere)."""
        state = runner.depth_processing()
        for key, label, _ in CAMERA_DEPTH_STEPS:
            if not state:
                want, colour = f"{label}: n/a (no RealSense)", "gray"
            else:
                on = bool(state.get(key))
                want, colour = f"{label}: {'ON' if on else 'OFF'}", ("green" if on else "gray")
            handle = depth_buttons[key]
            if handle.label != want:
                handle.label, handle.color = want, colour
        mask = runner.depth_mask
        for handle, on, label in ((btn_gate, mask.use_range_gate, "Range gate 0.01-3 m"),
                                  (btn_mask, mask.use_plane, "Table mask")):
            want = f"{label}: {'ON' if on else 'OFF'}"
            if handle.label != want:
                handle.label, handle.color = want, ("green" if on else "gray")
        camera = (f"camera: align -> {state.get('chain', '?')} | emitter {'on' if state.get('emitter') else 'off'}"
                  if state else "camera: none (sim / --blind: noise frames)")
        panel = ("range gate 0.01-3 m" if mask.use_range_gate else "NO range gate")
        panel += (" -> table mask" + ("" if mask.planes else " (no plane fitted: inactive)")) if mask.use_plane else ""
        noised = sorted(cfg.noise_slots)
        panel += f" -> noise on {', '.join(noised)}" if noised else ""
        server_side = ("gate 0.01-3 m" if mask.use_range_gate else "no gate") + " -> metres x depth_scale (2 mm steps)"
        text = f"{camera}\npanel: {panel}\npolicy server: {server_side}"
        if txt_chain.value != text:
            txt_chain.value = text

    @num_margin.on_update
    def _(_) -> None:
        runner.depth_mask.footprint_margin_m = float(num_margin.value)
        say(f"table mask outline margin {runner.depth_mask.footprint_margin_m:.2f} m")

    # ------------------------------------------------------------------------------------------ 3D scene
    with server.gui.add_folder("Gripper mount (match your adapter plate)"):
        sld_mount_yaw = server.gui.add_slider("Yaw about flange Z [deg]", min=-180.0, max=180.0, step=5.0,
                                              initial_value=float(np.degrees(scene.coupler_yaw_rad)),
                                              hint="Only moves the rendered gripper.")
        sld_mount_z = server.gui.add_slider("Z offset [mm]", min=0.0, max=60.0, step=1.0,
                                            initial_value=float(scene.coupler_offset_m * 1000.0),
                                            hint="Thickness of the coupling plate (render only).")
    sld_mount_yaw.on_update(lambda _: setattr(scene, "coupler_yaw_rad", float(np.radians(sld_mount_yaw.value))))
    sld_mount_z.on_update(lambda _: setattr(scene, "coupler_offset_m", float(sld_mount_z.value) / 1000.0))

    # ---------------------------------------------------------------------------------------- refreshers
    last_progress = [""]

    def refresh_worker() -> None:
        while not stop_ui.is_set():
            try:
                snap = runner.snapshot()
                txt_state.value = snap["state"] + (f" ({snap['message']})" if snap["message"] else "")
                filled = [k for k, v in runner.goal_slots.items() if v is not None]
                for slot, handle in (("A", btn_save_a), ("B", btn_save_b)):
                    want = "green" if slot in filled else "gray"
                    if handle.color != want:
                        handle.color = want
                txt_slots.value = (f"A: {'saved' if 'A' in filled else '-'} | B: {'saved' if 'B' in filled else '-'}"
                                   f" | active {runner.active_slot or '-'} | laps {runner.laps}")
                alt_label = ALT_ON_LABEL if cfg.alternate_goals else ALT_OFF_LABEL
                if btn_alt.label != alt_label:
                    btn_alt.label = alt_label
                    btn_alt.color = "green" if cfg.alternate_goals else "teal"
                running = snap["state"] == State.RUNNING.value
                loop_label = LOOP_STOP_LABEL if running else LOOP_START_LABEL
                if btn_loop.label != loop_label:
                    btn_loop.label = loop_label
                    btn_loop.color = "yellow" if running else "green"
                txt_mode.value = f"{snap['robot_mode']} (motion_ready={snap['motion_ready']})"
                txt_dist.value = "-" if snap["dist"] is None else f"{snap['dist']:.4f} rad"
                if snap["q"] is not None:
                    txt_q.value = "\n".join(f"{n:<18} {v:+.4f}" for n, v in zip(JOINT_NAMES, snap["q"]))
                st = snap["stats"]
                rr = runner.rates()
                txt_stats.value = (f"chunks {st.chunks}  waypoints {st.waypoints}\n"
                                   f"inference {st.last_inference_ms:.1f} ms | policy {rr.get('policy', 0.0):.1f} Hz, "
                                   f"loop {rr.get('loop', 0.0):.1f} Hz\n"
                                   f"chunk max step {st.last_chunk_max_step:.4f} rad, lead x{cfg.action_scale:.1f} "
                                   f"{st.last_lead:.3f} rad\n"
                                   f"speed cmd {st.cmd_speed:.2f} / measured {st.meas_speed:.2f} rad/s\n"
                                   f"clamped waypoints {st.clamped_waypoints}")
                g = snap["gripper"]
                txt_grip.value = g["text"]
                if snap["q"] is not None:
                    scene.update(snap["q"], gripper_closure=g["closure"])
                scene.set_goal(snap["goal"])
                # policy servers come up in the background: keep the checkpoint line and the progress current
                desc = pool.describe(active["name"])
                if txt_ckpt.value != desc and not txt_ckpt.value.startswith("REFUSED"):
                    txt_ckpt.value = desc
                progress = pool.progress()
                if progress != last_progress[0]:
                    last_progress[0] = progress
                    say(progress)
            except Exception:
                logger.exception("refresh failed")
            time.sleep(0.2)

    def camera_worker() -> None:
        """Shows what the policy is fed: the frames after the depth mask and noise ablation, for the inputs the active
        checkpoint reads (depth drawn in metres, cut as the policy cuts it; black = 0 = nothing was hit)."""
        last_error = ""
        while not stop_ui.is_set():
            try:
                snap = runner.camera_snapshot()
                if snap is not None:
                    reads = pool.reads(active["name"])  # e.g. {"front_left_rgb", "front_left_depth", ...}
                    frames = snap["policy"]
                    for key, handle in views.items():
                        slot, kind = key.rsplit("_", 1)
                        src = frames.get(slot if kind == "rgb" else f"{slot}_depth")
                        if src is None or (reads is not None and key not in reads):
                            if handle.visible:
                                handle.visible = False
                            continue
                        handle.image = src if kind == "rgb" else colorize(src)
                        if not handle.visible:
                            handle.visible = True
                    noised = cfg.noise_slots
                    for slot, handle in noise_buttons.items():
                        want = f"Noise out {slot}: {'ON' if slot in noised else 'OFF'}"
                        if handle.label != want:
                            handle.label = want
                            handle.color = "red" if slot in noised else "gray"
                    refresh_depth_controls()
                    r = runner.rates()
                    cams = r.get("cameras")
                    lines_r = ([f"{slot}: RGB {v.get('rgb', 0.0):4.1f} Hz | depth {v.get('depth', 0.0):4.1f} Hz"
                                for slot, v in sorted(cams.items())] if cams is not None
                               else ["cameras: none (sim / --blind: noise frames on demand)"])
                    lines_r.append(f"policy {r.get('policy', 0.0):4.1f} Hz (set {cfg.exec_rate_hz:.0f}) | control "
                                   f"loop {r.get('loop', 0.0):4.1f} Hz (set {runner.control_hz:.0f}) | view 5 Hz")
                    rates_text = "\n".join(lines_r)
                    if txt_rates.value != rates_text:
                        txt_rates.value = rates_text
                    age = time.time() - snap["t"]
                    source = f"NOISE on {','.join(sorted(noised))}" if noised else snap["source"]
                    means = {s: float(np.asarray(frames[s]).mean()) for s in runner.camera_slots if s in frames}
                    lines = [f"source {source}, age {age:.2f}s | inputs {pool.inputs(active['name'])} | mean px "
                             + ", ".join(f"{k} {v:.0f}" for k, v in means.items())
                             + ("  <-- BLACK?" if means and min(means.values()) < 2.0 else "")]
                    for slot in runner.camera_slots:
                        d = frames.get(f"{slot}_depth")
                        if d is not None:
                            valid = d > 0
                            lines.append(f"{slot}_depth {d.shape} {d[valid].min() if valid.any() else 0:.2f}.."
                                         f"{d.max():.2f} m, {100 * (1 - valid.mean()):.0f}% zero")
                    txt_frames.value = "\n".join(lines)
                last_error = ""
            except Exception as exc:
                txt_frames.value = f"camera view failed: {type(exc).__name__}: {exc}"
                if txt_frames.value != last_error:
                    logger.exception("camera view update failed")
                    last_error = txt_frames.value
            time.sleep(0.2)

    threading.Thread(target=refresh_worker, daemon=True, name="gui-refresh").start()
    threading.Thread(target=camera_worker, daemon=True, name="gui-cameras").start()
    logger.info("GUI built: status, checkpoint, policy, goal, home, gripper, camera views")
    return stop_ui
