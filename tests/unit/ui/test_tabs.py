"""Sorter tabs: one machine per tab, all of them live at once.

What these pin is what an operator with several machines relies on: a tab
keeps its own board, camera, model and bins; nothing one tab does leaks into
another; tabs survive a restart; and closing one never costs the library a
model or an image.
"""

from __future__ import annotations

import itertools
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("PySide6")

from sorter import paths
from sorter.data.config import Config
from sorter.data.models import Model
from sorter.data.repository import CartridgeRepo, HeadstampRepo, ModelRepo
from sorter.data.sorters import front_sorter_id, list_sorters
from sorter.hardware.serial_emulator import EMULATED_PORT, EmulatorBroker
from sorter.ml import classifier, local_inference
from sorter.ui import settings_camera, settings_serial
from sorter.ui.app import RUNNING_TAB_TOOLTIP
from sorter.ui.dashboard_page import DASHBOARD_TITLE
from sorter.ui.dialog_headstamps import MODEL_DEFAULT_LABEL, HeadstampManagerDialog
from sorter.ui.models_page import ACTIVE_MARK

from .conftest import drain_until, seed_model, tab


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CASESORTER_DATA_DIR", str(tmp_path / "data"))


class Recorder:
    def __init__(self, answer: Any = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.answer = answer

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.answer


def second_model(config: Any, headstamps: dict[str, int], name: str = "Other model") -> int:
    cartridge = CartridgeRepo(config.db).list()[0]
    model = ModelRepo(config.db).create(Model(name=name, cartridge_id=cartridge.id))
    for headstamp, slot in headstamps.items():
        HeadstampRepo(config.db).add(model.id, headstamp, slot)
    return model.id


def activate(window: Any, sorter: Any, model_id: int | None) -> None:
    """What the Models page's Activate does, for a tab that may not be in front."""
    sorter.config.set_active_model_id(model_id)
    sorter.bus.post("mode/changed", {"active_model_id": model_id})
    window.drain_all()


def give_checkpoint(config: Any, model_id: int) -> None:
    trained = paths.model_trained_dir(model_id)
    trained.mkdir(parents=True, exist_ok=True)
    checkpoint = trained / f"{model_id}.pth"
    checkpoint.write_bytes(b"not-really-a-checkpoint")
    repo = ModelRepo(config.db)
    model = repo.get(model_id)
    assert model is not None
    model.model_path = str(checkpoint)
    repo.update(model)


def stub_camera(sorter: Any) -> None:
    frame = np.zeros((480, 640, 3), np.uint8)
    sorter.camera = types.SimpleNamespace(capture_frame=lambda: frame, latest_frame=lambda: None, stop=lambda: None)


# ----- the strip ------------------------------------------------------------------


def test_one_sorter_looks_the_way_the_app_always_did(window) -> None:
    """AC2: an upgraded install is one tab, titled as before, nothing to close."""
    assert [window.tab_bar.tabText(i) for i in range(window.tab_bar.count())] == [DASHBOARD_TITLE, "Sorter 1"]
    assert window.tab_bar.currentIndex() == 1
    assert window.serial_dock.windowTitle() == "Serial Monitor"
    assert window.history_dock.windowTitle() == "Classification History"
    assert not window._close_buttons[tab(window).sorter_id].isVisibleTo(window.tab_bar)
    assert not window.tab_bar.isMovable()


def test_plus_opens_an_unconnected_sorter_on_its_sort_page(window) -> None:
    window.go_to_activity("Models")

    window.new_sorter_button.click()

    added = tab(window, 1)
    assert added.name == "Sorter 2"
    assert window.current_tab is added
    assert window._current_page_name() == "Sort"
    assert window.tab_stacks["Sort"].currentWidget() is added.sort_page
    assert added.broker is None
    # With two, both can be closed, and the docks say whose traffic they show.
    assert all(b.isVisibleTo(window.tab_bar) for b in window._close_buttons.values())
    assert window.serial_dock.windowTitle() == "Serial Monitor — Sorter 2"


def test_double_click_renames_and_a_taken_name_is_refused(window) -> None:
    window.new_sorter()
    window.notify = Recorder()
    window.ask_text = Recorder("Line B")

    window.tab_bar.tabBarDoubleClicked.emit(2)

    assert tab(window, 1).name == "Line B"
    assert window.tab_bar.tabText(2) == "Line B"
    assert [r.name for r in list_sorters(window.db)] == ["Sorter 1", "Line B"]

    window.ask_text = Recorder("sorter 1")
    window.tab_bar.tabBarDoubleClicked.emit(2)

    assert tab(window, 1).name == "Line B"
    assert window.notify.calls, "a clashing name must say why it was refused"


def test_the_dashboard_tab_is_not_renamed(window) -> None:
    window.ask_text = Recorder("Anything")

    window.tab_bar.tabBarDoubleClicked.emit(0)

    assert window.ask_text.calls == []


def test_closing_a_connected_sorter_asks_and_declining_keeps_it(window) -> None:
    window.new_sorter()
    doomed = tab(window, 1)
    doomed.connect_serial(EMULATED_PORT)
    window.confirm_close_tab = Recorder(False)

    assert window.close_tab(doomed) is False

    assert window.confirm_close_tab.calls == [(doomed,)]
    assert doomed in window.tabs
    assert isinstance(doomed.broker, EmulatorBroker)


def test_closing_a_sorter_stops_it_and_keeps_every_model_and_image(window, config) -> None:
    model_id = seed_model(config, {"WIN": 1})
    images = paths.model_images_dir(model_id)
    images.mkdir(parents=True, exist_ok=True)
    (images / "WIN__1.jpg").write_bytes(b"jpg")
    window.new_sorter()
    doomed = tab(window, 1)
    activate(window, doomed, model_id)
    doomed.connect_serial(EMULATED_PORT)
    broker = doomed.broker
    stopped: list[bool] = []
    broker.stop = lambda: stopped.append(True)
    window.confirm_close_tab = Recorder(True)

    assert window.close_tab(doomed) is True

    assert stopped == [True]
    assert [t.name for t in window.tabs] == ["Sorter 1"]
    assert window.current_tab is tab(window)
    assert [r.name for r in list_sorters(window.db)] == ["Sorter 1"]
    assert ModelRepo(config.db).get(model_id) is not None
    assert (images / "WIN__1.jpg").exists()
    # Back to one: the survivor can't be closed, and the docks drop the name.
    assert not window._close_buttons[tab(window).sorter_id].isVisibleTo(window.tab_bar)
    assert window.serial_dock.windowTitle() == "Serial Monitor"


def test_the_last_sorter_cannot_be_closed(window) -> None:
    window.confirm_close_tab = Recorder(True)

    assert window.close_tab(tab(window)) is False

    assert len(window.tabs) == 1


# ----- persistence -----------------------------------------------------------------


def test_tabs_come_back_after_a_restart(config, window_factory) -> None:
    """AC2: the roster, names, order, devices, models, bins and front tab return."""
    model_id = seed_model(config, {"WIN": 1, "FC": 2})
    other_id = second_model(config, {"PPU": 1})
    window = window_factory(config)
    window.new_sorter()
    line_b = tab(window, 1)
    window.ask_text = Recorder("Line B")
    window.rename_tab(line_b)
    activate(window, line_b, other_id)
    line_b.config.set_headstamp_slot("PPU", 4)
    line_b.connect_serial(EMULATED_PORT)
    tab(window).config.set_headstamp_slot("WIN", 3)
    window.close()

    front = front_sorter_id(config.db)
    assert front is not None
    assert front == line_b.sorter_id
    reopened = window_factory(Config(config.db, sorter_id=front).load())

    assert [t.name for t in reopened.tabs] == ["Sorter 1", "Line B"]
    assert reopened.current_tab is tab(reopened, 1)
    assert tab(reopened).config.active_model_id == model_id
    assert tab(reopened, 1).config.active_model_id == other_id
    assert tab(reopened).config.slot_for_headstamp("WIN") == 3
    assert tab(reopened, 1).config.slot_for_headstamp("PPU") == 4
    assert tab(reopened, 1).config.serial["port"] == EMULATED_PORT
    assert tab(reopened, 1).slot_grid.cards[4].names_label.text() == "PPU"


# ----- one machine per tab ----------------------------------------------------------


def test_a_port_another_tab_holds_is_labelled_and_refused(window, monkeypatch) -> None:
    """AC5: the port is still listed, so the operator sees where their board went."""
    window.new_sorter()
    first, second = tab(window), tab(window, 1)
    assert window.devices.claim_serial("COM3", first.sorter_id) is None
    monkeypatch.setattr(settings_serial.serial_broker, "list_serial_ports", lambda: ["COM3"])

    second.serial_section.refresh_ports()

    combo = second.serial_section.port_combo
    index = combo.findData("COM3")
    assert combo.itemText(index) == "COM3 (in use by Sorter 1)"
    assert not combo.model().item(index).isEnabled()

    second.connect_serial("COM3")

    assert second.broker is None
    assert "in use by Sorter 1" in second.last_status


def test_a_camera_another_tab_holds_is_refused_and_never_probed(window, monkeypatch) -> None:
    window.new_sorter()
    first, second = tab(window), tab(window, 1)
    window.devices.claim_camera(0, first.sorter_id)
    second.config.camera["device_index"] = 0

    second.start_camera()

    assert second.camera_state == ("Camera: in use by Sorter 1", False)

    probed: list[Any] = []

    def listing(**kwargs: Any) -> list[dict[str, Any]]:
        probed.append(kwargs.get("skip"))
        return [{"index": 1, "name": "Bench cam", "resolutions": [(640, 480)]}]

    monkeypatch.setattr(settings_camera, "camera_names", lambda: {})
    monkeypatch.setattr(settings_camera, "list_cameras_with_metadata", listing)
    second.camera_section.detect_devices()
    assert drain_until(window, lambda: bool(probed))

    assert probed == [{0}]


def test_two_tabs_on_one_model_keep_their_own_bins(window, config) -> None:
    """AC5: a slot change on one tab never moves a case on another."""
    model_id = seed_model(config, {"WIN": 1})
    window.new_sorter()
    first, second = tab(window), tab(window, 1)
    activate(window, second, model_id)

    first.config.set_headstamp_slot("WIN", 3)
    first.bus.post("run/assignment_changed", None)
    window.drain_all()

    assert first.config.slot_for_headstamp("WIN") == 3
    assert second.config.slot_for_headstamp("WIN") == 1
    assert first.slot_grid.cards[3].names_label.text() == "WIN"
    assert second.slot_grid.cards[1].names_label.text() == "WIN"
    assert second.slot_grid.cards[3].names_label.text() != "WIN"


def test_run_options_belong_to_the_tab(window) -> None:
    window.new_sorter()
    first, second = tab(window), tab(window, 1)

    first.floor_spin.setValue(81)
    first.package_check.setChecked(True)

    assert first.config.run_confidence_floor == 81
    assert second.config.run_confidence_floor != 81
    assert second.config.run_package_mode is False
    assert not second.package_check.isChecked()


def test_the_models_page_names_every_tab_a_model_is_active_on(window, config) -> None:
    model_id = seed_model(config, {"WIN": 1}, name="Range brass")
    window.go_to_activity("Models")
    marks = {
        window.models_page.tree.topLevelItem(i).text(0): window.models_page.tree.topLevelItem(i).text(1)
        for i in range(window.models_page.tree.topLevelItemCount())
    }
    assert marks["Range brass"] == ACTIVE_MARK

    window.new_sorter()
    activate(window, tab(window, 1), model_id)

    marks = {
        window.models_page.tree.topLevelItem(i).text(0): window.models_page.tree.topLevelItem(i).text(1)
        for i in range(window.models_page.tree.topLevelItemCount())
    }
    assert marks["Range brass"] == "● Sorter 1, Sorter 2"


# ----- the headstamp editor's "Slots for" -------------------------------------------


def slot_cells(dialog: Any) -> dict[str, str]:
    return {
        dialog.tree.topLevelItem(i).text(0): dialog.tree.topLevelItem(i).text(1)
        for i in range(dialog.tree.topLevelItemCount())
    }


def select(dialog: Any, name: str) -> None:
    for index in range(dialog.tree.topLevelItemCount()):
        if dialog.tree.topLevelItem(index).text(0) == name:
            dialog.tree.setCurrentItem(dialog.tree.topLevelItem(index))
            return
    pytest.fail(f"no row for {name!r}")


def open_editor(window: Any, model_id: int) -> Any:
    front = window.current_tab
    dialog = HeadstampManagerDialog(window, front.config, model_id, bus=front.bus, slot_targets=window.slot_targets)
    dialog.notify = Recorder()
    dialog.confirm = Recorder(True)
    return dialog


def targets(dialog: Any) -> list[str]:
    return [dialog.target_combo.itemText(i) for i in range(dialog.target_combo.count())]


def test_slots_for_lists_only_the_tabs_running_the_model(window, config) -> None:
    model_a = seed_model(config, {"WIN": 1})
    model_b = second_model(config, {"WIN": 2})
    window.new_sorter()
    first, second = tab(window), tab(window, 1)
    activate(window, second, model_b)
    window.show_tab(first)

    dialog = open_editor(window, model_a)

    assert targets(dialog) == [MODEL_DEFAULT_LABEL, "Sorter 1"]
    assert dialog.target_combo.currentText() == "Sorter 1"

    select(dialog, "WIN")
    dialog.slot_spin.setValue(5)

    assert first.config.slot_for_headstamp("WIN") == 5
    assert second.config.slot_for_headstamp("WIN") == 2, "model B's same-named headstamp moved"
    assert HeadstampRepo(config.db).list_for_model(model_a)[0].slot == 1, "the model default moved"

    # Activating A on the second tab while the editor is open offers it too.
    activate(window, second, model_a)

    assert targets(dialog) == [MODEL_DEFAULT_LABEL, "Sorter 1", "Sorter 2"]
    dialog.reject()


def test_slots_for_opens_on_the_model_default_when_the_front_tab_runs_something_else(window, config) -> None:
    model_a = seed_model(config, {"WIN": 1})
    model_b = second_model(config, {"PPU": 1})
    window.new_sorter()
    activate(window, tab(window, 1), model_b)

    dialog = open_editor(window, model_a)

    assert targets(dialog) == [MODEL_DEFAULT_LABEL, "Sorter 1"]
    assert dialog.target_combo.currentText() == MODEL_DEFAULT_LABEL
    dialog.reject()


# ----- everything runs at once ------------------------------------------------------


def test_two_sorters_run_at_once_whichever_is_in_front(config, window_factory, monkeypatch) -> None:
    """AC3: both counters climb while only one tab is shown, and switching
    tabs neither stops nor restarts either run."""
    model_id = seed_model(config, {"WIN": 1, "FC": 2})
    give_checkpoint(config, model_id)
    monkeypatch.setattr(local_inference, "is_installed", lambda: True)
    labels = itertools.cycle(("WIN", "FC"))
    monkeypatch.setattr(classifier, "classify_active", lambda *_a, **_k: (next(labels), 95.0))
    window = window_factory(config)
    window.notify = Recorder()
    window.new_sorter()
    first, second = tab(window), tab(window, 1)
    activate(window, second, model_id)
    for sorter in (first, second):
        stub_camera(sorter)
        sorter.connect_serial(EMULATED_PORT)

    first.start_run()
    second.start_run()
    assert drain_until(window, lambda: first.is_running and second.is_running)
    started = (first.run_controller, second.run_controller)

    window.show_tab(first)
    assert drain_until(window, lambda: first._master_count >= 3 and second._master_count >= 3, timeout_s=30)
    window.show_tab(second)
    before = first._master_count
    assert drain_until(window, lambda: first._master_count > before, timeout_s=30), "the background tab stalled"

    assert (first.run_controller, second.run_controller) == started
    running_index = window.tab_index(first)
    assert not window.tab_bar.tabIcon(running_index).isNull()
    assert window.tab_bar.tabToolTip(running_index) == RUNNING_TAB_TOOLTIP

    first.stop_run()
    second.stop_run()
    assert drain_until(window, lambda: not first.is_running and not second.is_running, timeout_s=10)
    assert window.tab_bar.tabIcon(running_index).isNull()
    assert window.notify.calls == []


def test_switching_template_on_one_tab_leaves_the_other_on_its_own(window, config) -> None:
    """AC5: template definitions are shared per model, the choice is per tab."""
    model_id = seed_model(config, {"WIN": 1})
    window.new_sorter()
    first, second = tab(window), tab(window, 1)
    activate(window, second, model_id)
    default = second.config.active_slot_template()

    blank = first.config.create_slot_template("Match prep", copy_current=False)

    assert first.config.active_slot_template().id == blank.id
    assert first.config.slot_for_headstamp("WIN") in (None, 0)
    assert second.config.active_slot_template().id == default.id
    assert second.config.slot_for_headstamp("WIN") == 1
    assert {t.name for t in second.config.list_slot_templates()} == {"Default", "Match prep"}


# ----- upgrade ------------------------------------------------------------------------


def test_a_pre_tabs_install_opens_as_one_unchanged_sorter(tmp_path: Path, window_factory) -> None:
    """AC2: the port, camera, active model and bins a single-machine install
    had are what its one tab shows, with nothing to close and no tab names in
    the panel titles."""
    from sorter.data.db import Database
    from sorter.data.repository import SettingsRepo
    from sorter.data.sorters import ensure_default_sorter

    db = Database(tmp_path / "legacy.db")
    db.ensure_initialized()
    model = ModelRepo(db).list()[0]
    HeadstampRepo(db).add(model.id, "WIN", slot=3)
    settings = SettingsRepo(db)
    settings.set("serial", {"port": EMULATED_PORT, "baud": 19200})
    settings.set("camera", {"device_index": 2})
    settings.set("default_model_id", model.id)

    ensure_default_sorter(db)
    front = front_sorter_id(db)
    assert front is not None
    window = window_factory(Config(db, sorter_id=front).load())

    only = tab(window)
    assert [t.name for t in window.tabs] == ["Sorter 1"]
    assert only.config.serial["port"] == EMULATED_PORT
    assert only.config.serial["baud"] == 19200
    assert only.config.camera["device_index"] == 2
    assert only.config.active_model_id == model.id
    assert only.slot_grid.cards[3].names_label.text() == "WIN"
    assert not window._close_buttons[only.sorter_id].isVisibleTo(window.tab_bar)
    assert window.serial_dock.windowTitle() == "Serial Monitor"
    window.close()
    db.close()
