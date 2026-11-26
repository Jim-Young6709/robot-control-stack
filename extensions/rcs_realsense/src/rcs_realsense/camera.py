import contextlib
import copy
import logging
import threading
import time
import typing
from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs
from rcs.camera.hw import CalibrationStrategy, DummyCalibrationStrategy, HardwareCamera
from rcs.camera.interface import BaseCameraSet, CameraFrame, DataFrame, Frame, IMUFrame

from rcs import common


@dataclass
class RealSenseDeviceInfo:
    product_line: str
    serial: str


@dataclass
class RealSenseDevicePipeline:
    pipeline: rs.pipeline
    pipeline_profile: rs.pipeline_profile
    camera: RealSenseDeviceInfo
    depth_scale: float | None = None
    color_intrinsics: np.ndarray[tuple[typing.Literal[3], typing.Literal[4]], np.dtype[np.float64]] | None = None
    depth_intrinsics: np.ndarray[tuple[typing.Literal[3], typing.Literal[4]], np.dtype[np.float64]] | None = None
    depth_to_color: common.Pose | None = None
    # created once per device (rs.align caches its mapping per stream profile; the temporal filter keeps history)
    align: rs.align | None = None
    depth_processing: "DepthProcessing | None" = None


#: Depth post-processing chains, applied to the delivered depth frame (after alignment when aligning), in order.
#: "drp_deploy": the chain of the DRP authors' RealSense node (JasonJZLiu/drp_deploy, src/cameras/cameras/realsense.py):
#:     disparity domain -> spatial filter -> temporal filter -> back to depth -> hole filling, all librealsense
#:     defaults (spatial: magnitude 2, alpha 0.5, delta 20, no hole fill; temporal: alpha 0.4, delta 20, persistence 3;
#:     hole filling: farthest from around).
#: "spatial_fill8": a spatial filter with an 8 px hole fill only (franka_control drp_control/cameras.py).
DEPTH_FILTER_CHAINS = ("drp_deploy", "spatial_fill8")


class DepthProcessing:
    """Depth post-processing of one device, switchable step by step while streaming.

    Order (as in drp_deploy): [to disparity] -> spatial -> temporal -> [back to depth] -> hole filling. The disparity
    pair only wraps the spatial / temporal filters (they then smooth 1/depth, which keeps far surfaces from being
    over-smoothed). Flags are plain attributes read once per frame by the polling thread, so they can be set from
    any thread. Re-enabling the temporal filter starts it with a fresh history.
    """

    STEPS = ("disparity", "spatial", "temporal", "hole_filling")

    def __init__(self, chain: str | None = None):
        self.to_disparity = rs.disparity_transform(True)
        self.to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()  # magnitude 2, alpha 0.5, delta 20, no hole fill
        self.temporal = rs.temporal_filter()  # alpha 0.4, delta 20, persistence "valid in 2 of last 4"
        self.hole_filling = rs.hole_filling_filter()  # "farthest from around"
        self.enabled = dict.fromkeys(self.STEPS, False)
        self.chain = chain
        if chain == "drp_deploy":
            self.enabled.update(disparity=True, spatial=True, temporal=True, hole_filling=True)
        elif chain == "spatial_fill8":
            self.spatial.set_option(rs.option.holes_fill, 3.0)
            self.enabled.update(spatial=True)
        elif chain is not None:
            msg = f"depth_filters must be one of {DEPTH_FILTER_CHAINS} or None, got {chain!r}"
            raise ValueError(msg)

    def set(self, **flags: bool) -> None:
        unknown = set(flags) - set(self.STEPS)
        if unknown:
            msg = f"unknown depth processing step(s) {sorted(unknown)}, have {self.STEPS}"
            raise ValueError(msg)
        if flags.get("temporal") and not self.enabled["temporal"]:
            self.temporal = rs.temporal_filter()  # no stale history from before it was switched off
        self.enabled.update({k: bool(v) for k, v in flags.items()})

    def process(self, frame: rs.frame) -> rs.frame:
        on = dict(self.enabled)  # one consistent set of flags per frame
        smooth = on["spatial"] or on["temporal"]
        if smooth and on["disparity"]:
            frame = self.to_disparity.process(frame)
        if on["spatial"]:
            frame = self.spatial.process(frame)
        if on["temporal"]:
            frame = self.temporal.process(frame)
        if smooth and on["disparity"]:
            frame = self.to_depth.process(frame)
        if on["hole_filling"]:
            frame = self.hole_filling.process(frame)
        return frame

    def describe(self) -> str:
        on = self.enabled
        steps = []
        smooth = [n for n in ("spatial", "temporal") if on[n]]
        if smooth:
            steps.append(("disparity: " if on["disparity"] else "") + " + ".join(smooth))
        if on["hole_filling"]:
            steps.append("hole filling")
        return " -> ".join(steps) or "none"


#: Capture for the DRP checkpoints with the front D415s at 320x240: colour rgb8 320x240, depth z16 848x480 aligned onto
#: colour (848x480 rather than 640x480: on a D415 640x480 depth is a crop that leaves a ~20 px stereo shadow band of
#: zeros in the colour view), the drp_deploy depth post-processing after alignment, emitter on at the sensor's default
#: laser power, true colour intrinsics, 15 warm-up frames, one hardware reset retry.
#: Depth is delivered as uint16 millimetres (exactly the z16 values at the D4xx default 1 mm depth unit) with shape
#: (H, W, 1), matching CameraSetWrapper's declared observation space. Convert to metres with
#: `depth.astype(np.float32) * np.float32(1.0 / BaseCameraSet.DEPTH_SCALE)` (bit identical to drp_control).
DRP_D415_KWARGS: dict[str, typing.Any] = {
    "color_format": "rgb8",
    "depth_resolution": (848, 480),
    "align_depth_to_color": True,
    "depth_filters": "drp_deploy",
    "enable_ir_emitter": True,
    "laser_power": None,
    "true_principal_point": True,
    "round_depth": True,
    "depth_channel_dim": True,
    "warmup_frames": 15,
    "frame_timeout_ms": 5000,
    "hardware_reset_retry": True,
    "restart_attempts": 3,
}


#: The same DRP capture for D455s at 640x480 (both colour and depth): at that mode the D455 sees 79 x 64 deg (inside the
#: renders' 48.8-83.7 deg hfov range). 640x480 depth aligns onto the 640x480 colour with only a narrow invalid band at
#: the left edge, while 16:9 depth modes (848x480) leave the top rows of the 4:3 colour view without depth. The
#: client downscales the frames to the policy's 320x240.
DRP_D455_KWARGS: dict[str, typing.Any] = dict(DRP_D415_KWARGS, depth_resolution=(640, 480))

class RealSenseCameraSet(HardwareCamera):
    TIMESTAMP_FACTOR = 1e-3
    CALIBRATION_FRAME_SIZE = 30
    COLOR_FORMATS: typing.ClassVar[dict[str, typing.Any]] = {"bgr8": rs.format.bgr8, "rgb8": rs.format.rgb8}
    # hardware reset: the device drops off the bus, then re-enumerates
    RESET_SETTLE_S = 3.0
    RESET_REENUMERATE_TIMEOUT_S = 15.0

    def __init__(
        self,
        cameras: dict[str, common.BaseCameraConfig],
        calibration_strategy: dict[str, CalibrationStrategy] | None = None,
        enable_ir_emitter: bool = False,
        enable_ir: bool = False,
        laser_power: int | None = 330,
        enable_imu: bool = False,
        align_depth_to_color: bool = False,
        color_format: str = "bgr8",
        depth_resolution: tuple[int, int] | None = None,
        depth_filters: str | None = None,
        true_principal_point: bool = False,
        round_depth: bool = False,
        depth_channel_dim: bool = False,
        warmup_frames: int = 0,
        frame_timeout_ms: int = 5000,
        hardware_reset_retry: bool = False,
        restart_attempts: int = 0,
    ) -> None:
        """RealSense D400 camera set.

        All new arguments default to the previous behaviour.

        Args:
            laser_power: laser power set whenever the emitter option is written; None = the sensor's own default
                (its option range default, i.e. the power-up state).
            color_format: "bgr8" (capture BGR, deliver an RGB view, previous behaviour) or "rgb8" (capture RGB,
                deliver it contiguous as is).
            depth_resolution: (width, height) of the depth stream; None = the colour resolution. A different depth
                resolution requires align_depth_to_color=True (CameraSetWrapper declares depth at colour size).
            depth_filters: None = no post-processing, or the name of a chain in DEPTH_FILTER_CHAINS ("drp_deploy",
                "spatial_fill8"), applied to the depth frame, after alignment when aligning.
            true_principal_point: intrinsics carry the stream's ppx/ppy instead of ((w-1)/2, (h-1)/2).
            round_depth: depth to uint16 at 1/BaseCameraSet.DEPTH_SCALE m: z16 values are passed through unchanged
                when the depth unit already equals that (D4xx default 1 mm), otherwise rounded to nearest. False
                keeps the previous float conversion with truncation.
            depth_channel_dim: deliver depth as (H, W, 1) instead of (H, W).
            warmup_frames: frames waited out right after each pipeline start (auto exposure), each with
                frame_timeout_ms.
            frame_timeout_ms: timeout of every wait_for_frames.
            hardware_reset_retry: if a device fails to start or sends no frames during warm-up, hardware-reset it
                and try once more (not when the device is busy, i.e. held by another process).
            restart_attempts: when a running device stops delivering frames (USB glitch, device reset), stop and
                restart its pipeline up to this many times (2, 4, 5 s apart) before giving up, instead of ending the
                polling thread at the first timeout. 0 keeps the previous behaviour.
        """
        self.enable_ir_emitter = enable_ir_emitter
        self.enable_ir = enable_ir
        self.laser_power = laser_power
        self.enable_imu = enable_imu
        self.cameras = cameras
        self.align_depth_to_color = align_depth_to_color
        if color_format not in self.COLOR_FORMATS:
            msg = f"color_format must be one of {sorted(self.COLOR_FORMATS)}, got {color_format!r}"
            raise ValueError(msg)
        self.color_format = color_format
        if depth_filters is not None and depth_filters not in DEPTH_FILTER_CHAINS:
            msg = f"depth_filters must be one of {DEPTH_FILTER_CHAINS} or None, got {depth_filters!r}"
            raise ValueError(msg)
        self.depth_filters = depth_filters
        self.true_principal_point = true_principal_point
        self.round_depth = round_depth
        self.depth_channel_dim = depth_channel_dim
        self.warmup_frames = int(warmup_frames)
        self.frame_timeout_ms = int(frame_timeout_ms)
        self.hardware_reset_retry = hardware_reset_retry
        self.restart_attempts = int(restart_attempts)
        if calibration_strategy is None:
            calibration_strategy = {camera_name: DummyCalibrationStrategy() for camera_name in cameras}
        self.calibration_strategy = calibration_strategy
        self._logger = logging.getLogger(__name__)
        assert (
            len({camera.resolution_width for camera in self.cameras.values()}) == 1
            and len({camera.resolution_height for camera in self.cameras.values()}) == 1
            and len({camera.frame_rate for camera in self.cameras.values()}) == 1
        ), "All cameras must have the same resolution and frame rate."
        sample_camera_config = next(iter(self.cameras.values()))
        self.resolution_width = sample_camera_config.resolution_width
        self.resolution_height = sample_camera_config.resolution_height
        self.frame_rate = sample_camera_config.frame_rate
        if depth_resolution is None:
            self.depth_width, self.depth_height = self.resolution_width, self.resolution_height
        else:
            self.depth_width, self.depth_height = (int(v) for v in depth_resolution)
        if (self.depth_width, self.depth_height) != (
            self.resolution_width,
            self.resolution_height,
        ) and not self.align_depth_to_color:
            msg = (
                "depth_resolution different from the colour resolution requires align_depth_to_color=True "
                "(CameraSetWrapper declares depth frames at the colour resolution)"
            )
            raise ValueError(msg)
        # Upper bound for open() (start, warm-up and, with hardware_reset_retry, one reset per device); a
        # HardwareCameraSet that knows this attribute waits that long for the first frame instead of 10 s.
        per_device = 3.0 + self.warmup_frames / self.frame_rate
        if self.hardware_reset_retry:
            per_device += self.frame_timeout_ms / 1000 + self.RESET_SETTLE_S + self.RESET_REENUMERATE_TIMEOUT_S + 4.0
        self.startup_timeout_s = max(10.0, len(self.cameras) * per_device + 2.0)
        self._frame_buffer_lock: dict[str, threading.Lock] = {}
        self._frame_buffer: dict[str, list] = {}

        self.D400_config = self._build_config()
        self._available_devices: dict[str, RealSenseDeviceInfo] = {}
        self._enabled_devices: dict[str, RealSenseDevicePipeline] = {}  # serial numbers of te enabled devices
        self._camera_names = list(self.cameras.keys())

    def _build_config(self) -> rs.config:
        config = rs.config()
        config.enable_stream(
            rs.stream.depth,
            self.depth_width,
            self.depth_height,
            rs.format.z16,
            self.frame_rate,
        )
        config.enable_stream(
            rs.stream.color,
            self.resolution_width,
            self.resolution_height,
            self.COLOR_FORMATS[self.color_format],
            self.frame_rate,
        )
        if self.enable_ir:
            # the infrared imagers are the depth imagers: same resolution as depth
            config.enable_stream(
                rs.stream.infrared,
                1,
                self.depth_width,
                self.depth_height,
                rs.format.y8,
                self.frame_rate,
            )
        if self.enable_imu:
            # TODO(juelg): does not work work at the moment: "Couldnt resolve requests"
            # https://www.intelrealsense.com/how-to-getting-imu-data-from-d435i-and-t265/
            # Accelerometer available FPS: {63, 250}Hz
            config.enable_stream(
                rs.stream.accel,
                rs.format.motion_xyz32f,
                250,
            )
            # Gyroscope available FPS: {200,400}Hz
            config.enable_stream(
                rs.stream.gyro,
                rs.format.motion_xyz32f,
                200,
            )
        return config

    @property
    def camera_names(self) -> list[str]:
        """Returns the names of the cameras in this set."""
        return self._camera_names

    def config(self, camera_name) -> common.BaseCameraConfig:
        return self.cameras[camera_name]

    def open(self):
        # close() disables all streams of the config, rebuild it so that the set can be opened again
        self.D400_config = self._build_config()
        self._available_devices = {}
        self.update_available_devices()
        self._enabled_devices = {}  # serial numbers of te enabled devices
        self.enable_devices({key: value.identifier for key, value in self.cameras.items()}, self.enable_ir_emitter)

    def update_available_devices(self):
        self._available_devices = self.enumerate_connected_devices(rs.context())

    def enable_devices(self, devices_to_enable: dict[str, str], enable_ir_emitter: bool = False):
        """
        Enable the Intel RealSense Devices which are connected to the PC

        Parameters:
        -----------
        devices_to_enable : dict
                            Dictionary with readable name and serial number of the devices to be enabled
        enable_ir_emitter : bool
                            Enable/Disable the IR-Emitter of the device

        """
        for device_name, device_serial in devices_to_enable.items():
            assert (
                device_serial in self._available_devices
            ), f"Device {device_name} not found. Check if it is connected."
            self.enable_device(device_name, self._available_devices[device_serial], enable_ir_emitter)

    def _set_emitter(self, sensor: rs.sensor, enable_ir_emitter: bool):
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1 if enable_ir_emitter else 0)
            laser_power = (
                self.laser_power
                if self.laser_power is not None
                else sensor.get_option_range(rs.option.laser_power).default
            )
            sensor.set_option(rs.option.laser_power, laser_power)

    def _start_pipeline(
        self, camera_name: str, device_info: RealSenseDeviceInfo, enable_ir_emitter: bool
    ) -> tuple[rs.pipeline, rs.pipeline_profile]:
        """Starts the device's pipeline, sets the emitter and waits out `warmup_frames`.

        With hardware_reset_retry a device that fails to start or sends no frames is hardware-reset and started
        once more (a busy device, held by another process, is not reset).
        """
        attempts = 2 if self.hardware_reset_retry else 1
        for attempt in range(1, attempts + 1):
            pipeline = rs.pipeline()
            self.D400_config.enable_device(device_info.serial)
            try:
                pipeline_profile = pipeline.start(self.D400_config)
                # Set the acquisition parameters
                self._set_emitter(pipeline_profile.get_device().first_depth_sensor(), enable_ir_emitter)
                for _ in range(self.warmup_frames):
                    pipeline.wait_for_frames(self.frame_timeout_ms)
            except Exception as exc:
                with contextlib.suppress(Exception):
                    pipeline.stop()
                if attempt == attempts or "busy" in str(exc).lower():
                    raise
                self._logger.warning(
                    "camera %s (%s) did not deliver frames (%s): hardware reset, retrying",
                    camera_name,
                    device_info.serial,
                    exc,
                )
                self._hardware_reset(device_info.serial)
                continue
            return pipeline, pipeline_profile
        msg = "unreachable"
        raise AssertionError(msg)

    def _hardware_reset(self, serial: str):
        for dev in rs.context().query_devices():
            if dev.supports(rs.camera_info.serial_number) and dev.get_info(rs.camera_info.serial_number) == serial:
                dev.hardware_reset()
                break
        time.sleep(self.RESET_SETTLE_S)
        deadline = time.monotonic() + self.RESET_REENUMERATE_TIMEOUT_S
        while time.monotonic() < deadline:
            serials = {
                d.get_info(rs.camera_info.serial_number)
                for d in rs.context().query_devices()
                if d.supports(rs.camera_info.serial_number)
            }
            if serial in serials:
                time.sleep(1.0)
                return
            time.sleep(0.5)
        self._logger.warning(
            "camera %s did not come back within %.0f s after the hardware reset", serial, self.RESET_REENUMERATE_TIMEOUT_S
        )

    def _intrinsics_matrix(
        self, intrinsics: rs.intrinsics
    ) -> np.ndarray[tuple[typing.Literal[3], typing.Literal[4]], np.dtype[np.float64]]:
        if self.true_principal_point:
            cx, cy = intrinsics.ppx, intrinsics.ppy
        else:
            cx, cy = (intrinsics.width - 1) / 2, (intrinsics.height - 1) / 2
        return np.array(  # type: ignore[return-value]
            [
                [intrinsics.fx, 0, cx, 0],
                [0, intrinsics.fy, cy, 0],
                [0, 0, 1, 0],
            ]
        )

    def enable_device(self, camera_name: str, device_info: RealSenseDeviceInfo, enable_ir_emitter: bool = False):
        """
        Enable an Intel RealSense Device

        Parameters:
        -----------
        device_info     : Tuple of strings (serial_number, product_line)
                            Serial number and product line of the realsense device
        enable_ir_emitter : bool
                            Enable/Disable the IR-Emitter of the device

        """
        if device_info.product_line != "D400":
            msg = f"unknown product line {device_info.product_line}"
            raise RuntimeError(msg)
        pipeline, pipeline_profile = self._start_pipeline(camera_name, device_info, enable_ir_emitter)

        sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_vp = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_vp = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intrinsics = self._intrinsics_matrix(color_vp.get_intrinsics())
        depth_intrinsics = self._intrinsics_matrix(depth_vp.get_intrinsics())
        depth_to_color = depth_vp.get_extrinsics_to(color_vp)
        depth_scale = sensor.get_depth_scale()
        if self.round_depth and abs(depth_scale * BaseCameraSet.DEPTH_SCALE - 1.0) > 1e-6:
            self._logger.warning(
                "camera %s: depth unit %.6g m is not 1/%d m, depth is rounded to that grid (not bit exact)",
                camera_name,
                depth_scale,
                BaseCameraSet.DEPTH_SCALE,
            )


        self._enabled_devices[camera_name] = RealSenseDevicePipeline(
            pipeline,
            pipeline_profile,
            device_info,
            depth_scale=depth_scale,
            color_intrinsics=color_intrinsics,
            depth_intrinsics=depth_intrinsics,
            depth_to_color=common.Pose(
                translation=depth_to_color.translation, rotation=np.array(depth_to_color.rotation).reshape(3, 3)  # type: ignore
            ),
            align=rs.align(rs.stream.color) if self.align_depth_to_color else None,
            depth_processing=DepthProcessing(self.depth_filters),
        )

        self._frame_buffer[camera_name] = []
        self._frame_buffer_lock[camera_name] = threading.Lock()
        rs_color = color_vp.get_intrinsics()
        self._logger.info(
            "Enabled %s (%s, %s): colour %dx%d %s fx %.1f fy %.1f ppx %.1f ppy %.1f, depth %dx%d%s%s",
            camera_name,
            device_info.serial,
            device_info.product_line,
            rs_color.width,
            rs_color.height,
            self.color_format,
            rs_color.fx,
            rs_color.fy,
            rs_color.ppx,
            rs_color.ppy,
            self.depth_width,
            self.depth_height,
            " aligned to colour" if self.align_depth_to_color else "",
            f", depth filters {self.depth_filters}" if self.depth_filters else "",
        )

    @staticmethod
    def enumerate_connected_devices(context: rs.context) -> dict[str, RealSenseDeviceInfo]:
        """
        Enumerate the connected Intel RealSense devices

        Parameters:
        -----------
        context 	   : rs.context()
                        The context created for using the realsense library

        Return:
        -----------
        connect_device : array
                        Array of (serial, product-line) tuples of devices which are connected to the PC

        """
        connect_device: dict[str, RealSenseDeviceInfo] = {}

        d: rs.device
        for d in context.devices:
            if d.get_info(rs.camera_info.name).lower() != "platform camera":
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                device_info = RealSenseDeviceInfo(serial=serial, product_line=product_line)
                connect_device[serial] = device_info
        return connect_device

    def _depth_to_rcs(self, raw: np.ndarray, depth_scale: float) -> np.ndarray:
        """z16 frame (already a private copy) -> uint16 in units of 1/BaseCameraSet.DEPTH_SCALE m."""
        if not self.round_depth:
            data = (raw.astype(np.float64) * depth_scale * BaseCameraSet.DEPTH_SCALE).astype(np.uint16)
        elif abs(depth_scale * BaseCameraSet.DEPTH_SCALE - 1.0) < 1e-6:
            # the device's depth unit already is 1/DEPTH_SCALE m (D4xx default 1 mm): the z16 values, unchanged
            data = raw
        else:
            data = np.clip(
                np.rint(raw.astype(np.float64) * depth_scale * BaseCameraSet.DEPTH_SCALE), 0, np.iinfo(np.uint16).max
            ).astype(np.uint16)
        return data[:, :, None] if self.depth_channel_dim else data

    def poll_frame(self, camera_name: str) -> Frame:
        assert camera_name in self.camera_names, f"Camera {camera_name} not found in the enabled devices"
        device = self._enabled_devices[camera_name]

        try:
            frameset = device.pipeline.wait_for_frames(self.frame_timeout_ms)
        except RuntimeError as exc:
            if not self.restart_attempts:
                raise
            frameset = self._restart_device(camera_name, exc)
        streams = device.pipeline_profile.get_streams()

        if self.align_depth_to_color:
            # replaces the frameset with a composite frameset containing the aligned depth
            align = device.align if device.align is not None else rs.align(rs.stream.color)
            frameset = align.process(frameset)

        color: DataFrame | None = None
        ir: DataFrame | None = None
        depth: DataFrame | None = None
        accel: DataFrame | None = None
        gyro: DataFrame | None = None

        def to_numpy(frame: rs.frame) -> np.ndarray:
            return np.asanyarray(frame.get_data()).copy()

        def to_ts(frame: rs.frame) -> float:
            # convert to seconds
            return frame.get_timestamp() * RealSenseCameraSet.TIMESTAMP_FACTOR

        color_extrinsics = self.calibration_strategy[camera_name].get_extrinsics()

        if self.align_depth_to_color:
            # if aligned, depth acts as if it was shot from the color sensor
            depth_extrinsics = color_extrinsics
            active_depth_intrinsics = device.color_intrinsics
        else:
            depth_to_color = device.depth_to_color
            assert depth_to_color is not None, "Depth to color extrinsics not found"
            depth_extrinsics = (
                color_extrinsics @ depth_to_color.inverse().pose_matrix() if color_extrinsics is not None else None
            )
            active_depth_intrinsics = device.depth_intrinsics

        timestamps = []
        for stream in streams:
            if rs.stream.infrared == stream.stream_type():
                frame = frameset.get_infrared_frame(stream.stream_index())
                ir = DataFrame(data=to_numpy(frame), timestamp=to_ts(frame))
            elif rs.stream.color == stream.stream_type():
                frame = frameset.get_color_frame()
                color_data = to_numpy(frame)
                color = DataFrame(
                    # always RGB: rgb8 as captured, bgr8 as a reversed view (previous behaviour)
                    data=color_data if self.color_format == "rgb8" else color_data[:, :, ::-1],
                    timestamp=to_ts(frame),
                    intrinsics=device.color_intrinsics,
                    extrinsics=color_extrinsics,
                )
            elif rs.stream.depth == stream.stream_type():
                frame = frameset.get_depth_frame()
                if device.depth_processing is not None:
                    # on the frame that is delivered, i.e. after alignment (order of drp_deploy and drp_control)
                    frame = device.depth_processing.process(frame)
                assert device.depth_scale is not None, "Depth scale not found"
                depth = DataFrame(
                    data=self._depth_to_rcs(to_numpy(frame), device.depth_scale),
                    timestamp=to_ts(frame),
                    intrinsics=active_depth_intrinsics,
                    extrinsics=depth_extrinsics,  # type: ignore
                )
            elif rs.stream.accel == stream.stream_type():
                frame = frameset.first(stream.stream_index())
                md = frame.as_motion_frame().get_motion_data()
                accel = DataFrame(data=np.array([md.x, md.y, md.z]), timestamp=to_ts(frame))
            elif rs.stream.gyro == stream.stream_type():
                frame = frameset.first(stream.stream_index())
                md = frame.as_motion_frame().get_motion_data()
                gyro = DataFrame(data=np.array([md.x, md.y, md.z]), timestamp=to_ts(frame))
            else:
                msg = f"Unknown stream type {stream.stream_type()}"
                self._logger.warning(msg)
                continue
            timestamps.append(to_ts(frame))

        assert color is not None, "Color frame not found"
        cf = CameraFrame(
            color=color,
            ir=ir,
            depth=depth,
        )
        imu = IMUFrame(accel=accel, gyro=gyro)
        f = Frame(camera=cf, imu=imu, avg_timestamp=float(np.mean(timestamps)) if len(timestamps) > 0 else None)
        with self._frame_buffer_lock[camera_name]:
            if len(self._frame_buffer[camera_name]) >= self.CALIBRATION_FRAME_SIZE:
                self._frame_buffer[camera_name].pop(0)
            self._frame_buffer[camera_name].append(copy.deepcopy(f))
        return f

    def disable_streams(self):
        self.D400_config.disable_all_streams()

    def close(self):
        for device in self._enabled_devices.values():
            device.pipeline.stop()
        self.disable_streams()

    def _restart_device(self, camera_name: str, error: Exception):
        """Restarts a device that stopped delivering frames (as drp_deploy's RealSense node does); returns the first
        frameset of the new pipeline. Raises after `restart_attempts` failed restarts."""
        device = self._enabled_devices[camera_name]
        last: Exception = error
        for attempt in range(1, self.restart_attempts + 1):
            self._logger.warning(
                "camera %s (%s) stopped delivering frames (%s): restarting its pipeline, attempt %d/%d",
                camera_name, device.camera.serial, last, attempt, self.restart_attempts,
            )
            with contextlib.suppress(Exception):
                device.pipeline.stop()
            time.sleep(min(2.0 * attempt, 5.0))
            try:
                pipeline, profile = self._start_pipeline(camera_name, device.camera, self.enable_ir_emitter)
                device.pipeline, device.pipeline_profile = pipeline, profile
                if self.align_depth_to_color:
                    device.align = rs.align(rs.stream.color)
                if device.depth_processing is not None and device.depth_processing.enabled["temporal"]:
                    device.depth_processing.temporal = rs.temporal_filter()  # no history across the gap
                frameset = pipeline.wait_for_frames(self.frame_timeout_ms)
                self._logger.info("camera %s (%s) is delivering frames again", camera_name, device.camera.serial)
                return frameset
            except Exception as exc:  # noqa: BLE001 - busy, not re-enumerated yet, ...
                last = exc
        msg = f"camera {camera_name} ({device.camera.serial}) did not recover after {self.restart_attempts} restarts: {last}"
        raise RuntimeError(msg)

    def set_depth_processing(self, **flags: bool) -> None:
        """Switches depth post-processing steps (DepthProcessing.STEPS) of every device while streaming."""
        for device in self._enabled_devices.values():
            if device.depth_processing is not None:
                device.depth_processing.set(**flags)

    def depth_processing(self) -> dict[str, typing.Any]:
        """Current post-processing flags (of the first device; all devices are switched together) and emitter."""
        for device in self._enabled_devices.values():
            if device.depth_processing is not None:
                out: dict[str, typing.Any] = dict(device.depth_processing.enabled)
                out["chain"] = device.depth_processing.describe()
                sensor = device.pipeline_profile.get_device().first_depth_sensor()
                if sensor.supports(rs.option.emitter_enabled):
                    out["emitter"] = bool(sensor.get_option(rs.option.emitter_enabled))
                return out
        return {}

    def enable_emitter(self, enable_ir_emitter=True):
        """
        Enable/Disable the emitter of the intel realsense device

        """
        self.enable_ir_emitter = bool(enable_ir_emitter)  # also used when a device is restarted
        for device in self._enabled_devices.values():
            # Get the active profile and enable the emitter for all the connected devices
            sensor = device.pipeline_profile.get_device().first_depth_sensor()
            if not sensor.supports(rs.option.emitter_enabled):
                continue
            sensor.set_option(rs.option.emitter_enabled, 1 if enable_ir_emitter else 0)
            if enable_ir_emitter:
                laser_power = (
                    self.laser_power
                    if self.laser_power is not None
                    else sensor.get_option_range(rs.option.laser_power).default
                )
                sensor.set_option(rs.option.laser_power, laser_power)

    def calibrate(self) -> bool:
        for camera_name in self.cameras:
            color_intrinsics = self._enabled_devices[camera_name].color_intrinsics
            assert color_intrinsics is not None, f"Color intrinsics for camera {camera_name} not found"
            if not self.calibration_strategy[camera_name].calibrate(
                intrinsics=color_intrinsics,
                samples=self._frame_buffer[camera_name],
                lock=self._frame_buffer_lock[camera_name],
            ):
                self._logger.warning(f"Calibration of camera {camera_name} failed.")
                return False
        self._logger.info("Calibration successful.")
        return True
