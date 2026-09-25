"""Which sorter tab holds which serial port and camera (``ui/device_registry.py``).

Pure bookkeeping, no Qt: the behaviour a user sees is the settings pages
saying "in use by <name>" and refusing, which ``test_tabs.py`` covers through
the real window.
"""

from __future__ import annotations

from sorter.hardware.serial_emulator import EMULATED_PORT
from sorter.ui.device_registry import DeviceRegistry, in_use_label

NAMES = {1: "Sorter 1", 2: "Line B"}


def registry() -> DeviceRegistry:
    return DeviceRegistry(name_of=lambda sid: NAMES[sid])


def test_a_port_held_by_one_tab_is_refused_to_another_by_name() -> None:
    devices = registry()

    assert devices.claim_serial("COM3", 1) is None
    assert devices.claim_serial("COM3", 2) == "Sorter 1"
    assert devices.serial_holder("COM3", 2) == "Sorter 1"
    # The holder itself is never blocked by its own claim.
    assert devices.serial_holder("COM3", 1) is None


def test_claiming_a_new_port_gives_the_old_one_back() -> None:
    devices = registry()
    devices.claim_serial("COM3", 1)

    assert devices.claim_serial("COM4", 1) is None

    assert devices.claim_serial("COM3", 2) is None


def test_the_emulator_is_never_held() -> None:
    devices = registry()

    assert devices.claim_serial(EMULATED_PORT, 1) is None
    assert devices.claim_serial(EMULATED_PORT, 2) is None


def test_a_camera_index_is_exclusive_and_listed_for_the_other_tabs() -> None:
    devices = registry()
    devices.claim_camera(0, 1)

    assert devices.claim_camera(0, 2) == "Sorter 1"
    assert devices.cameras_held_by_others(2) == {0: "Sorter 1"}
    assert devices.cameras_held_by_others(1) == {}


def test_release_all_frees_both_devices() -> None:
    devices = registry()
    devices.claim_serial("COM3", 1)
    devices.claim_camera(0, 1)

    devices.release_all(1)

    assert devices.claim_serial("COM3", 2) is None
    assert devices.claim_camera(0, 2) is None


def test_the_holder_name_is_read_when_asked_so_a_rename_shows() -> None:
    names = dict(NAMES)
    devices = DeviceRegistry(name_of=lambda sid: names[sid])
    devices.claim_serial("COM3", 1)

    names[1] = "Bench"

    assert devices.claim_serial("COM3", 2) == "Bench"
    assert in_use_label("Bench") == "in use by Bench"
