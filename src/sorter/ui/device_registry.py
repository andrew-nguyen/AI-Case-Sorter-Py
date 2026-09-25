"""Which sorter tab holds which serial port and which camera.

Every tab drives its own machine, and a machine's board and camera must be
opened by exactly one tab: two ``SerialBroker`` s on one port interleave
commands, and a second ``cv2.VideoCapture`` on one V4L2 device often opens and
then delivers garbage. The operating system refuses neither reliably, and when
it does refuse, its error cannot name the tab holding the device. This
registry is the one place that knows, so every connect path asks it first and
every settings list can say "in use by <tab name>".

It lives in the UI layer on purpose: the holder is a tab, and naming it is UI
vocabulary the hardware modules should not carry. It is in-process only — a
second copy of the app is not a case this app supports.

``EMULATED_PORT`` is exempt: it is not a device, and any number of tabs may
each run their own emulator.
"""

from __future__ import annotations

from collections.abc import Callable

from ..hardware.serial_emulator import EMULATED_PORT

IN_USE_TEXT = "in use by {name}"


def in_use_label(name: str) -> str:
    return IN_USE_TEXT.format(name=name)


class DeviceRegistry:
    """Serial ports and camera indices, each held by at most one sorter id."""

    def __init__(self, name_of: Callable[[int], str]) -> None:
        # Resolved at the moment a message is built, so a rename is reflected
        # without re-claiming anything.
        self._name_of = name_of
        self._serial: dict[str, int] = {}
        self._camera: dict[int, int] = {}

    # ----- serial ---------------------------------------------------------

    def serial_holder(self, port: str, sorter_id: int) -> str | None:
        """The name of the *other* tab holding ``port``, or None if it is free."""
        port = (port or "").strip()
        if not port or port == EMULATED_PORT:
            return None
        holder = self._serial.get(port)
        if holder is None or holder == sorter_id:
            return None
        return self._name_of(holder)

    def claim_serial(self, port: str, sorter_id: int) -> str | None:
        """Take ``port`` for ``sorter_id``; returns the blocking tab's name on refusal.

        A tab holds one board at a time, so a successful claim drops whatever
        port it held before.
        """
        port = (port or "").strip()
        blocker = self.serial_holder(port, sorter_id)
        if blocker is not None:
            return blocker
        self.release_serial(sorter_id)
        if port and port != EMULATED_PORT:
            self._serial[port] = sorter_id
        return None

    def release_serial(self, sorter_id: int) -> None:
        for port in [p for p, holder in self._serial.items() if holder == sorter_id]:
            del self._serial[port]

    # ----- camera ---------------------------------------------------------

    def camera_holder(self, index: int, sorter_id: int) -> str | None:
        holder = self._camera.get(int(index))
        if holder is None or holder == sorter_id:
            return None
        return self._name_of(holder)

    def claim_camera(self, index: int, sorter_id: int) -> str | None:
        blocker = self.camera_holder(index, sorter_id)
        if blocker is not None:
            return blocker
        self.release_camera(sorter_id)
        self._camera[int(index)] = sorter_id
        return None

    def release_camera(self, sorter_id: int) -> None:
        for index in [i for i, holder in self._camera.items() if holder == sorter_id]:
            del self._camera[index]

    def cameras_held_by_others(self, sorter_id: int) -> dict[int, str]:
        """Index -> holder name, for every camera another tab has open."""
        return {index: self._name_of(holder) for index, holder in self._camera.items() if holder != sorter_id}

    # ----- lifecycle ------------------------------------------------------

    def release_all(self, sorter_id: int) -> None:
        self.release_serial(sorter_id)
        self.release_camera(sorter_id)
