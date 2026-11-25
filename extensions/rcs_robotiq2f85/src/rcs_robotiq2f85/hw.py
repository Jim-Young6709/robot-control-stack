import logging
import typing

from rcs._core.common import Gripper, GripperConfig, GripperState
from rcs.common_typing import GripperConfigKwargs
from robotiq2f import Robotiq2F85, Robotiq2FStatus

import rcs

logger = logging.getLogger(__name__)


class RobotiQ2F85GripperConfig(GripperConfig):

    def __init__(
        self,
        serial_number: str,
        speed: float = 100,
        force: float = 50,
        async_control: bool = True,
        activate_on_init: bool = True,
        **kwargs: typing.Unpack[GripperConfigKwargs],
    ) -> None:
        """
        Args:
            serial_number: Get the serial number with `udevadm info -a -n /dev/ttyUSB0 | grep serial`, make sure you have read/write permissions to the port.
            speed: Speed in mm/s. Must be between 20 and 150 mm/s.
            force: Force in N. Must be between 20 and 235 N.
            async_control: If True, gripper commands return immediately without waiting for the movement to complete
                (a new command interrupts any ongoing movement), and the driver polls the gripper status in a
                background thread so that state reads are served from cache instead of blocking on Modbus.
            activate_on_init: If True, the gripper is reset (deactivated and activated, the fingers move) when the
                driver is created. If False, only the serial port is opened and `RobotiQ2F85Gripper.activate()` has
                to be called before the gripper moves; motion commands before that are ignored, because a "go to"
                request carries the activation bit and would start the activation implicitly.
        """
        super().__init__(**kwargs)
        self.serial_number = serial_number
        self.speed = speed
        self.force = force
        self.async_control = async_control
        self.activate_on_init = activate_on_init
        self.gripper_type = rcs.common.GripperType("Robotiq2F85")


class RobotiQ2F85GripperState(GripperState):
    def __init__(self, state: Robotiq2FStatus) -> None:
        super().__init__()
        self.state = state


class RobotiQ2F85Gripper(Gripper):
    def __init__(self, cfg: RobotiQ2F85GripperConfig):
        super().__init__()
        self._cfg: RobotiQ2F85GripperConfig = cfg
        self.gripper = Robotiq2F85(serial_number=cfg.serial_number, async_control=cfg.async_control)
        self._activated = False
        if getattr(cfg, "activate_on_init", True):
            self.activate()
        else:
            # a gripper activated earlier in this power cycle stays activated: usable right away, nothing moves
            self._activated = bool(self.gripper.read_status_sync().is_activated)

    def activate(self) -> None:
        """Deactivates and activates the gripper (blocking, the fingers move). Also clears gripper faults."""
        # commands from other threads are dropped while the activation runs
        self._activated = False
        self.gripper.reset()
        self._activated = True

    def is_activated(self) -> bool:
        return self._activated and self.gripper.read_status().is_activated

    def get_normalized_width(self) -> float:
        return self.gripper.opening / self.gripper.MAX_OPENING

    def grasp(self) -> None:
        """
        Close the gripper to grasp an object.
        """
        self.set_normalized_width(0.0, force=self._cfg.force)

    def open(self) -> None:
        """
        Open the gripper to its maximum width.
        """
        self.set_normalized_width(1.0)

    def reset(self) -> None:
        # with lazy activation the gripper only moves on explicit commands, so an env reset does not open it
        if getattr(self._cfg, "activate_on_init", True):
            self.open()

    def set_normalized_width(self, width: float, force: float = 0) -> None:
        """
        Set the gripper width to a normalized value between 0 and 1.
        """
        if not (0 <= width <= 1):
            msg = f"Width must be between 0 and 1, got {width}."
            raise ValueError(msg)
        if not self._activated:
            logger.warning("Robotiq %s is not activated yet, ignoring the command", self._cfg.serial_number)
            return
        abs_width = width * self.gripper.MAX_OPENING
        self.gripper.go_to(
            opening=float(abs_width),
            speed=self._cfg.speed,
            force=force if force != 0 else self._cfg.force,
        )

    def shut(self) -> None:
        """
        Close the gripper.
        """
        self.set_normalized_width(0.0)

    def close(self) -> None:
        self.gripper.client.serial.close()

    def get_config(self) -> GripperConfig:
        return self._cfg

    def set_config(self, cfg: RobotiQ2F85GripperConfig) -> None:
        self._cfg = cfg

    def get_state(self) -> GripperState:
        return RobotiQ2F85GripperState(state=self.gripper.read_status())
