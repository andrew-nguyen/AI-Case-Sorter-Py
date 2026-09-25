"""One sorter tab: the machine it drives, and the pages that drive it.

The window (``app.py``) is the shell every tab shares — sidebar, docks, status
bar, menus, the model library, Community, themes, sign-in and the one database
connection. A ``SorterTab`` is the per-machine half: its own ``Config``
(serial, camera, image processing, active model, run options, live slot
layouts), its own ``EventBus``, camera, serial broker and ``RunController``,
and its own instances of every per-machine page — Sort, Train, AI Config and
the Camera / Serial / Image Processing settings sections.

Those pages are built with ``build_*(tab)`` exactly as they were built with
``build_*(window)`` before tabs existed, so this class exposes the attribute
surface they already read (``config``, ``bus``, ``db``, ``camera``,
``broker``, ``run_controller``, ``notify``, ``run_worker``, ``set_status``,
``ensure_torch``, ``frame_to_image``, ``palette_colors``, ``go_to_activity``,
…) and delegates the shared parts of it to the window.

Every tab keeps running whatever tab is in front: its bus is drained by the
window's one timer, and its Sort page exists (hidden) to receive its own
counts. What only the front tab gets is the window's shared surfaces — the
status bar, the docks, the sidebar's mode inks — which the window repaints
from ``current_tab`` whenever a tab reports a change through ``_changed``.

Scope and rationale: docs/ui-modernization.md, "Sorter tabs".
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from ..control.events import EventBus
from ..control.run_controller import RunController
from ..hardware import serial_broker
from ..hardware.camera import Camera
from ..hardware.serial_emulator import EMULATED_PORT, EmulatorBroker
from ..hardware.serial_log import SerialTrafficLog
from ..ml import classifier
from .ai_page import build_ai_page
from .device_registry import in_use_label
from .dialog_headstamp_assign import HeadstampAssignDialog
from .dialog_slot_assign import CATCH_ALL_HINT, SlotAssignDialog
from .dialog_template import EditTemplateDialog, NewTemplateDialog
from .message_log import ERROR, INFO
from .serial_monitor import MAX_LINES as SERIAL_BUFFER_LINES
from .settings_camera import build_camera_section
from .settings_imageproc import build_imageproc_section
from .settings_serial import build_serial_section
from .slot_grid import SlotGrid
from .train_page import build_train_page

# The Sort column's primary panel: the crop the classifier actually saw, plus
# the one result it produced (Seth, 2026-08-13 — the Windows app's layout).
# History belongs to the Monitor dock, not to a strip under the dashboard.
CROP_EMPTY_TEXT = "No case captured yet"
RESULT_EMPTY_TEXT = "—"
RESULT_EMPTY_CONFIDENCE = "—"
CAPTURE_CAPTION = "Last capture"
HEADSTAMP_CAPTION = "Headstamp"
CONFIDENCE_CAPTION = "Confidence"

# The live feed is a monitor, not the working surface: off unless asked for,
# and the preview timer does no camera read while it is (see refresh_preview).
# One app-wide preference, not a per-machine one.
SHOW_CAMERA_TEXT = "Show live camera"
SETTING_SHOW_CAMERA = "ui.sort_show_camera"

# The Start/Stop toggle: one button, two faces. The key is what
# `action_buttons` exposes it under.
RUN_TOGGLE_KEY = "Start/Stop"
RUN_START_TEXT = "Start"
RUN_STOP_TEXT = "Stop"
TEMPLATE_BUTTON_WIDTH = 36

# The camera preview when there is nothing to show.
PREVIEW_INITIAL_TEXT = "No frame"
CAMERA_DEAD_TEXT = "No camera feed"
CAMERA_DEAD_LINK = "open Camera settings"
CAMERA_FAILED_STATUS = "Camera failed to start — pick a device in Settings → Camera."

# Run options, grouped into one popover rather than spread over the page.
STORE_IMAGES_LABELS = {
    "none": "None",
    "above": "Above confidence floor",
    "below": "Below confidence floor",
    "all": "All images",
}
STORE_IMAGES_BY_LABEL = {label: mode for mode, label in STORE_IMAGES_LABELS.items()}
STORE_IMAGES_WARNING_TITLE = "Store images enabled"
STORE_IMAGES_WARNING_TEXT = (
    "Classified run images will be saved under the active model's run_images "
    "folder. This can use significant disk space over time."
)

# Community model settings (issue #29, A24/A25). The fetch fires on entering
# Sort with a community model active and is fail-open throughout: anything that
# goes wrong leaves the local floor, the local opt-in and no prompts.
MODEL_UPDATE_BUTTON = "Model update: v{version}"
NOTES_BUTTON = "Moderator notes ({count})"
FEEDBACK_BLOCKED_STATUS = "Feedback paused by the model's moderator."
NOTES_GATE_TITLE = "Moderator note"
NOTES_GATE_TEXT = (
    "A moderator has left a note about this model's feedback images. "
    "Read and acknowledge it before starting a run — open it with the "
    "“Moderator notes” button."
)

EMPTY_STATE_TITLE = "Nothing connected yet"
EMPTY_STATE_HINT = "Connect a board and a camera to start sorting."

# A run is stopped, not paused, when the link drops (issue #35): a reconnect
# mid-cycle leaves the wheel's position and the drop pipeline unknown, and
# resuming from there is how a case lands in the wrong bin.
SERIAL_LOST_TITLE = "Serial disconnected"
SERIAL_LOST_TEXT = (
    "The board stopped responding, so the run was stopped.\n\n"
    "Check the cable and the board's power, then reconnect from "
    "Settings → Serial."
)

# What the dashboard and the snapshot call AI Config mode's "model".
AI_CONFIG_MODEL_LABEL = "AI Config"

# How many classification records a tab keeps for the history dock to replay
# when the tab comes to the front. Each carries a 480×480 crop (~0.7 MB), and
# the dock rarely shows more tiles than this.
HISTORY_BUFFER = 48

# The per-machine activities and settings sections, keyed the way the window's
# sidebar and Settings list name them.
TAB_ACTIVITIES = ("Sort", "Train", "AI Config")
TAB_SETTINGS_SECTIONS = ("Camera", "Serial", "Image Processing")


def frame_to_image(frame: np.ndarray) -> QImage:
    """Wrap a BGR numpy frame as a QImage.

    ``QImage`` borrows the buffer it is handed, so the copy is what cuts the
    result loose from a frame the grab thread is about to overwrite.
    """
    buffer = np.ascontiguousarray(frame)
    height, width = buffer.shape[:2]
    image = QImage(buffer.data, width, height, buffer.strides[0], QImage.Format.Format_BGR888)
    return image.copy()


class _PreviewLabel(QLabel):
    """The camera preview surface — plain text only, whole-widget click.

    This was a rich-text label with an ``<a>`` link; PySide6 6.11's
    QTextDocument path crashed the Windows CI runner (offscreen) with a
    deterministic access violation the first time an event pump touched it.
    Plain text plus a click signal gives the same affordance without ever
    instantiating a text document.
    """

    clicked = Signal()

    def mousePressEvent(self, event: Any) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)


class _CropPanel(QLabel):
    """The last cropped headstamp, filling whatever space the column gives it.

    Keeps the source pixmap aside and re-scales on resize: the scaled copy must
    never become the label's size hint, or each repaint grows the layout the
    next one is scaled to (same discipline as ``_PreviewLabel``'s host).
    """

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(1, 1)

    def set_source(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        self._rescale()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None or self._source.isNull():
            return
        self.setPixmap(
            self._source.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class SorterTab(QObject):
    """The per-machine state and pages of one sorter tab.

    A ``QObject`` parented to the window so ``QTimer.singleShot(0, self, …)``
    has an owner that dies with the tab: a modal queued by a tab that is then
    closed is dropped instead of firing into a torn-down page.
    """

    def __init__(self, window: Any, config: Any, name: str) -> None:
        super().__init__(window)
        self.window = window
        self.config = config
        self.sorter_id = int(config.sorter_id)
        self.name = name
        self.bus = EventBus()
        # The same traffic, to a file while this tab's Settings → Serial has it
        # switched on. One per tab: the setting, the checkbox and the bus are.
        self.serial_log = SerialTrafficLog(enabled=bool(config.serial.get("log_traffic", False)))
        self.serial_log.attach(self.bus)
        self.broker: Any | None = None
        self.run_controller: RunController | None = None
        # Constructing a Camera does not open the device; start_camera does.
        self.camera = Camera(
            device_index=int(config.camera.get("device_index", 0)),
            width=int(config.camera.get("width", 640)),
            height=int(config.camera.get("height", 480)),
        )
        self._is_running = False
        self._master_count = 0
        self._templates: list[Any] = []
        self.headstamp_assign_dialog: HeadstampAssignDialog | None = None
        # The current case only — (display label, confidence, above the floor).
        self._current_result: tuple[str, float, bool] | None = None
        self.last_crop: np.ndarray | None = None
        # The store-images disk-usage notice shows once per session, not per run.
        self._store_warning_shown = False
        # Community model settings, from the last Sort-page fetch (A24/A25).
        # `_community_settings` is kept so a serial reconnect — which builds a
        # fresh RunController, and with it a fresh FeedbackService — doesn't
        # silently drop the server's policy.
        self._settings_fetch_busy = False
        self._community_settings: tuple[int, Any] | None = None
        self._model_update: tuple[str, int, int, Any] | None = None
        self._camera_state = ("Camera: disconnected", False)
        self._serial_state = ("Serial: disconnected", False)
        # The last status line this tab produced, shown again when it comes
        # to the front.
        self.last_status = ""
        # Retained for the shared docks, which follow the front tab and
        # replay these on a switch (serial_monitor / history_view `retarget`).
        self.serial_lines: deque[tuple[str, float, str]] = deque(maxlen=SERIAL_BUFFER_LINES)
        self.history_records: deque[dict[str, Any]] = deque(maxlen=HISTORY_BUFFER)
        self.history_case_number = 0
        self._muted_labels: list[QLabel] = []
        # Modal seams (CLAUDE.md §5): instance attributes, so a test replaces
        # them and nothing blocks offscreen.
        self.open_notes_dialog: Callable[[], None] = self._open_notes_dialog
        self.open_model_update_dialog: Callable[[], None] = self._open_model_update_dialog

        self._subscribe()
        self.sort_page = self._build_sort_page()
        self.train_page = build_train_page(self)
        self.ai_page = build_ai_page(self)
        self.camera_section = build_camera_section(self)
        self.serial_section = build_serial_section(self)
        self.imageproc_section = build_imageproc_section(self)
        # What the window's per-tab stacks hold, by activity / section name.
        self.pages: dict[str, QWidget] = {
            "Sort": self.sort_page,
            "Train": self.train_page,
            "AI Config": self.ai_page,
            "Camera": self.camera_section,
            "Serial": self.serial_section,
            "Image Processing": self.imageproc_section,
        }
        self._update_sort_empty_state()
        self._paint_preview_placeholder()

    def _subscribe(self) -> None:
        bus = self.bus
        bus.subscribe("status", self.set_status)
        bus.subscribe("status/error", lambda msg: self.set_status(msg, level=ERROR))
        bus.subscribe("status/progress", lambda msg: self.set_status(msg, progress=True))
        # Run state comes from the controller's own events, never from the
        # button handlers — a run can also end on its own (error, package halt).
        bus.subscribe("run/started", lambda _p: self._on_run_started())
        bus.subscribe("run/stopped", lambda _p: self._on_run_stopped())
        bus.subscribe("run/status", self.set_status)
        # Manual feed / test cycles report on their own topic; without this
        # their progress is invisible and the previous status looks stuck.
        bus.subscribe("test/status", self.set_status)
        bus.subscribe("run/error", lambda msg: self.set_status(f"Run error: {msg}", level=ERROR))
        bus.subscribe("test/error", lambda msg: self.set_status(f"Test error: {msg}", level=ERROR))
        bus.subscribe("run/result", self._on_run_result)
        # Both fire right after classify_active — the moment the inference
        # device is guaranteed to have been picked.
        bus.subscribe("run/classified", lambda _p: self.refresh_device_indicator())
        bus.subscribe("test/classified", lambda _p: self.refresh_device_indicator())
        bus.subscribe("run/history", self._on_run_history)
        bus.subscribe("run/assignment_changed", lambda _p: self._refresh_sort_grid())
        bus.subscribe("run/package_full", self._on_package_full)
        bus.subscribe("run/package_halt", self._on_package_halt)
        bus.subscribe("run/out_of_brass", self._on_out_of_brass)
        bus.subscribe("serial/disconnected", self._on_serial_disconnected)
        # Subscribed before any dock retargets onto this bus, so the buffer
        # always holds what the monitor would have shown.
        bus.subscribe("serial/rx", lambda line: self._buffer_serial("rx", line))
        bus.subscribe("serial/tx", lambda line: self._buffer_serial("tx", line))
        bus.subscribe("serial/note", lambda line: self._buffer_serial("note", line))
        # Headstamps, templates and the Train activity are all scoped to the
        # active model, so a mode switch re-reads every one of them.
        bus.subscribe("mode/changed", lambda _p: self._on_mode_changed())

    # ----- delegated to the window -------------------------------------------

    @property
    def db(self) -> Any:
        return self.window.db

    @property
    def palette_colors(self) -> dict[str, str]:
        return self.window.palette_colors

    @property
    def ensure_torch(self) -> Any:
        return self.window.ensure_torch

    @property
    def devices(self) -> Any:
        return self.window.devices

    @property
    def is_front(self) -> bool:
        return self.window.current_tab is self

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def camera_state(self) -> tuple[str, bool]:
        """(indicator text, connected) — what the status bar shows for the front tab."""
        return self._camera_state

    @property
    def serial_state(self) -> tuple[str, bool]:
        return self._serial_state

    def notify(self, title: str, text: str) -> None:
        """A modal on the window, titled with this tab's name when there are several."""
        if len(self.window.tabs) > 1:
            title = f"{self.name} — {title}"
        self.window.notify(title, text)

    def run_worker(
        self,
        fn: Callable[[], Any],
        *,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.window.run_worker(fn, on_done=on_done, on_error=on_error)

    def set_status(self, message: Any, *, level: str = INFO, progress: bool | None = None) -> None:
        self.last_status = str(message)
        self.window.show_tab_status(self, self.last_status, level=level, progress=progress)

    frame_to_image = staticmethod(frame_to_image)

    def go_to_activity(self, name: str) -> None:
        """An in-page "take me there" button: this tab, then the activity."""
        self.window.show_tab(self)
        self.window.go_to_activity(name)

    def _open_settings_section(self, name: str) -> None:
        self.window.show_tab(self)
        self.window._open_settings_section(name)

    def open_serial_monitor(self) -> None:
        self.window.show_tab(self)
        self.window.open_serial_monitor()

    def refresh_device_indicator(self) -> None:
        self.window.refresh_device_indicator()

    def beep(self) -> None:
        """Non-blocking batch-complete tone. Best-effort — never fails a handler."""
        try:
            QApplication.beep()
        except Exception:
            pass

    def _load_setting(self, key: str) -> Any:
        return self.window._load_setting(key)

    def _save_setting(self, key: str, value: Any) -> None:
        self.window._save_setting(key, value)

    def _muted_label(self, text: str, parent: QWidget | None = None) -> QLabel:
        """A label in the muted role — registered so a theme switch recolors it."""
        label = QLabel(text, parent)
        label.setStyleSheet(f"color: {self.palette_colors['text_muted']};")
        self._muted_labels.append(label)
        return label

    def _changed(self) -> None:
        """Anything the status bar, the tab title or the dashboard shows moved."""
        self.window.on_tab_changed(self)

    def rename(self, name: str) -> None:
        self.name = name
        self._changed()

    def apply_palette(self) -> None:
        """Re-ink what a stylesheet can't reach, after a theme switch."""
        muted = f"color: {self.palette_colors['text_muted']};"
        for label in self._muted_labels:
            label.setStyleSheet(muted)
        self._paint_current_result()

    # ----- snapshot ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Everything the "All sorters" dashboard shows for this tab, read live."""
        model = self.active_model()
        port = str(getattr(self.broker, "port", "") or self.config.serial.get("port") or "")
        return {
            "id": self.sorter_id,
            "name": self.name,
            "port": port,
            "serial": self._serial_state[0],
            "serial_connected": self._serial_state[1],
            "camera": self._camera_state[0],
            "camera_connected": self._camera_state[1],
            "model": model.name if model is not None else AI_CONFIG_MODEL_LABEL,
            "running": self._is_running,
            "connected": self.broker is not None,
            "count": self._master_count,
            "result": self._current_result,
            "crop": self.last_crop,
        }

    # ----- sort page ---------------------------------------------------------

    def _build_sort_page(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(12, 12, 12, 12)
        column.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal, page)
        splitter.addWidget(self._build_preview_column(splitter))
        splitter.addWidget(self._build_grid_column(splitter))
        # JL: the slot cards are the working surface and get the majority; the
        # camera is a monitor, not the centerpiece.
        splitter.setSizes([360, 640])

        # Index 0 is the working dashboard, 1 the first-run guided panel — see
        # _update_sort_empty_state.
        self.sort_stack = QStackedWidget(page)
        self.sort_stack.addWidget(splitter)
        self.sort_stack.addWidget(self._build_empty_state_panel(page))
        column.addWidget(self.sort_stack, 1)
        # At the foot, mirroring the Train page's Training strip (JL): the
        # working surface first, the launchers under it.
        column.addLayout(self._build_action_row(page))
        return page

    def _build_grid_column(self, parent: QWidget) -> QWidget:
        """The slot grid, with the run counter/reset and the template picker above it.

        JL (follow-up to the run-options move): the counter and reset button
        felt orphaned floating on the action row — they belong with what
        they count, not with Start/Stop/Manual feed. Same argument for the
        template picker: it names the layout these cards *are*.
        """
        holder = QWidget(parent)
        column = QVBoxLayout(holder)
        # Left margin clears the splitter handle — "Slots" was pressed right
        # up against the divider line (JL live-testing).
        column.setContentsMargins(10, 0, 0, 0)
        column.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(self._muted_label("Slots", holder))
        # The inverse of clicking a card: every headstamp, its slot set on the row (#129).
        self.assign_by_headstamp_button = QPushButton("Assign by headstamp…", holder)
        self.assign_by_headstamp_button.clicked.connect(self.open_headstamp_assign)
        header.addWidget(self.assign_by_headstamp_button)
        header.addStretch(1)
        header.addWidget(self._muted_label("Sorted this run", holder))
        self.master_count_label = QLabel("0", holder)
        self.master_count_label.setObjectName("masterCount")
        header.addWidget(self.master_count_label)
        reset = QPushButton("Reset counts", holder)
        reset.clicked.connect(self.reset_counts)
        header.addWidget(reset)
        self._add_template_group(holder, header)
        column.addLayout(header)

        self.slot_grid = SlotGrid(self.config, holder)
        self.slot_grid.slot_clicked.connect(lambda slot: self.open_slot_editor(slot))
        self.slot_grid.slot_reset.connect(lambda slot: self.reset_slot_count(slot))
        column.addWidget(self.slot_grid, 1)
        return holder

    def _build_empty_state_panel(self, page: QWidget) -> QWidget:
        """First-run guidance in place of a grid nothing has configured yet."""
        panel = QWidget(page)
        column = QVBoxLayout(panel)
        column.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.setSpacing(10)

        title = QLabel(EMPTY_STATE_TITLE, panel)
        title.setObjectName("emptyStateTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(title)
        hint = self._muted_label(EMPTY_STATE_HINT, panel)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(hint)

        self.empty_state_board_button = QPushButton("Connect a board — Settings → Serial", panel)
        self.empty_state_board_button.clicked.connect(lambda: self._open_settings_section("Serial"))
        column.addWidget(self.empty_state_board_button)

        self.empty_state_camera_button = QPushButton("Connect a camera — Settings → Camera", panel)
        self.empty_state_camera_button.clicked.connect(lambda: self._open_settings_section("Camera"))
        column.addWidget(self.empty_state_camera_button)
        return panel

    def _build_action_row(self, page: QWidget) -> QHBoxLayout:
        """The launchers, right-aligned — the Train page's Training strip, mirrored (JL)."""
        actions = QHBoxLayout()
        self.action_buttons: dict[str, QPushButton] = {}
        actions.addStretch(1)

        # Visible whenever the active model has notes at all, acknowledged or
        # not — it is the history view as well as the ack flow.
        self.notes_button = QPushButton(NOTES_BUTTON.format(count=0), page)
        self.notes_button.clicked.connect(lambda: self.open_notes_dialog())
        self.notes_button.hide()
        actions.addWidget(self.notes_button)

        feed_button = QPushButton("Manual feed", page)
        feed_button.clicked.connect(self.manual_feed)
        actions.addWidget(feed_button)
        self.action_buttons["Manual feed"] = feed_button

        # No dedicated row for this (JL) — was its own bar under the
        # template row. The run counter/reset live with the grid instead
        # (see `_build_grid_column`), not here.
        self.run_options_button = self._build_run_options_button(page)
        actions.addWidget(self.run_options_button)

        # One button, two faces (JL): a run is on or it isn't, and the button
        # that ends it is the one that started it. `_update_run_buttons` owns
        # the label/role swap. Last on the strip: the green primary sits at
        # the far right, matching the Training strip (JL).
        self.run_button = QPushButton(RUN_START_TEXT, page)
        self.run_button.setObjectName("action")
        self.run_button.clicked.connect(self.toggle_run)
        actions.addWidget(self.run_button)
        self.action_buttons[RUN_TOGGLE_KEY] = self.run_button

        self._update_run_buttons()
        return actions

    def _add_template_group(self, page: QWidget, bar: QHBoxLayout) -> None:
        """The template picker, right-aligned on the slot grid's header row.

        JL: which layout a run uses belongs over the cards that layout *is*,
        after the counters — not down on the launcher strip. The picker and
        its two edits are a group, so New/Edit shrink to glyphs with tooltips.
        """
        self.template_hint = self._muted_label("", page)
        bar.addWidget(self.template_hint)
        bar.addWidget(self._muted_label("Template", page))
        self.template_combo = QComboBox(page)
        self.template_combo.setMinimumWidth(200)
        self.template_combo.setMaximumWidth(240)
        # `activated` is user-only, so repopulating the combo can't look like a switch.
        self.template_combo.activated.connect(self._on_template_selected)
        bar.addWidget(self.template_combo)
        self.template_new_button = QPushButton("+", page)
        self.template_new_button.setToolTip("New template…")
        self.template_new_button.clicked.connect(self.new_template)
        self.template_edit_button = QPushButton("✎", page)
        self.template_edit_button.setToolTip("Edit template…")
        self.template_edit_button.clicked.connect(self.edit_template)
        for button in (self.template_new_button, self.template_edit_button):
            button.setMaximumWidth(TEMPLATE_BUTTON_WIDTH)
            bar.addWidget(button)
        self._refresh_templates()

    def _build_run_options_button(self, page: QWidget) -> QToolButton:
        """Everything that configures how a run behaves, in one popover.

        Grouped compactly rather than always-visible (JL, increment 14:
        package mode/batch moved here from the template bar, which now
        carries only template things). Every option is this tab's own.
        """
        button = QToolButton(page)
        button.setText("⚙ Run options")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        menu = QMenu(button)
        form_holder = QWidget(menu)
        form = QFormLayout(form_holder)
        form.setContentsMargins(10, 8, 10, 8)

        self.store_images_combo = QComboBox(form_holder)
        self.store_images_combo.addItems(list(STORE_IMAGES_LABELS.values()))
        self.store_images_combo.setCurrentText(STORE_IMAGES_LABELS.get(self.config.run_store_images, "None"))
        self.store_images_combo.currentTextChanged.connect(self._on_store_images_changed)
        form.addRow("Store images", self.store_images_combo)

        self.floor_spin = QSpinBox(form_holder)
        self.floor_spin.setRange(0, 100)
        self.floor_spin.setSuffix("%")
        self.floor_spin.setValue(int(self.config.run_confidence_floor))
        self.floor_spin.valueChanged.connect(self._on_floor_changed)
        form.addRow("Confidence floor", self.floor_spin)

        self.auto_select_check = QCheckBox("Automatically select trays", form_holder)
        self.auto_select_check.setChecked(bool(self.config.run_auto_select_trays))
        self.auto_select_check.toggled.connect(self._on_auto_select_toggled)
        form.addRow(self.auto_select_check)

        self.package_check = QCheckBox("Package mode", form_holder)
        self.package_check.setChecked(bool(self.config.run_package_mode))
        self.package_check.toggled.connect(self._on_package_mode_toggled)
        form.addRow(self.package_check)

        self.batch_caption = self._muted_label("Batch size", form_holder)
        self.batch_spin = QSpinBox(form_holder)
        self.batch_spin.setRange(1, 999999)
        self.batch_spin.setValue(int(self.config.run_package_size))
        self.batch_spin.valueChanged.connect(self._on_batch_size_changed)
        form.addRow(self.batch_caption, self.batch_spin)
        self._apply_package_visibility()

        action = QWidgetAction(menu)
        action.setDefaultWidget(form_holder)
        menu.addAction(action)
        button.setMenu(menu)
        return button

    def _on_store_images_changed(self, label: str) -> None:
        mode = STORE_IMAGES_BY_LABEL.get(label, "none")
        self.config.set_run_store_images(mode)
        if mode != "none" and not self._store_warning_shown:
            self._store_warning_shown = True
            self.notify(STORE_IMAGES_WARNING_TITLE, STORE_IMAGES_WARNING_TEXT)

    def _on_floor_changed(self, value: int) -> None:
        # The current-result line reads config.run_confidence_floor live on
        # every run/history event — no separate wiring for the coloring.
        self.config.set_run_confidence_floor(int(value))

    def _on_auto_select_toggled(self, checked: bool) -> None:
        self.config.set_run_auto_select_trays(bool(checked))

    def _build_preview_column(self, parent: QWidget) -> QWidget:
        """The crop the classifier saw, what it made of it, and — on request — the feed.

        Seth (2026-08-13): the operator watches the *cropped* headstamp and the
        call made on it, the way the Windows app shows them. The live camera is
        a setup aid, so it is off by default and secondary when shown.
        """
        holder = QWidget(parent)
        column = QVBoxLayout(holder)
        # Right margin clears the splitter handle, mirroring the grid
        # column's left margin (JL: the divider line sat too tight).
        column.setContentsMargins(0, 0, 10, 0)
        column.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(self._muted_label(CAPTURE_CAPTION, holder))
        header.addStretch(1)
        self.show_camera_check = QCheckBox(SHOW_CAMERA_TEXT, holder)
        # Restored before the preview exists, so the handler is wired below it.
        self.show_camera_check.setChecked(bool(self._load_setting(SETTING_SHOW_CAMERA)))
        header.addWidget(self.show_camera_check)
        column.addLayout(header)

        self.crop_label = _CropPanel(CROP_EMPTY_TEXT, holder)
        self.crop_label.setObjectName("cropPanel")
        column.addWidget(self.crop_label, 3)
        column.addLayout(self._build_result_row(holder))

        self.preview_label = _PreviewLabel(PREVIEW_INITIAL_TEXT, holder)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # A dead camera is the one thing this panel can usefully say, so it
        # says where to fix it rather than sitting black (JL). The whole
        # label is the click target — see _PreviewLabel for why not a link.
        self.preview_label.clicked.connect(self._on_preview_clicked)
        # Ignored + tiny minimum: the label must never report the pixmap as
        # its size hint, or each scaled frame grows the layout that the next
        # frame is scaled to — the window ratchets larger on every repaint.
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview_label.setMinimumSize(1, 1)
        # A video letterbox is black in every theme; not chrome, so not themed.
        self.preview_label.setStyleSheet("background-color: #000000; color: #808080;")
        self.preview_label.setVisible(self.show_camera_check.isChecked())
        # Smaller stretch than the crop: shown, it is the secondary panel.
        column.addWidget(self.preview_label, 2)
        self.show_camera_check.toggled.connect(self._on_show_camera_toggled)
        return holder

    def _build_result_row(self, holder: QWidget) -> QHBoxLayout:
        """The current case, Windows-style: what it is and how sure we are."""
        row = QHBoxLayout()
        row.addWidget(self._muted_label(HEADSTAMP_CAPTION, holder))
        self.result_label = QLabel(RESULT_EMPTY_TEXT, holder)
        self.result_label.setObjectName("currentHeadstamp")
        row.addWidget(self.result_label)
        row.addStretch(1)
        row.addWidget(self._muted_label(CONFIDENCE_CAPTION, holder))
        self.result_confidence_label = QLabel(RESULT_EMPTY_CONFIDENCE, holder)
        self.result_confidence_label.setObjectName("currentConfidence")
        row.addWidget(self.result_confidence_label)
        self._paint_current_result()
        return row

    def _on_show_camera_toggled(self, checked: bool) -> None:
        self.preview_label.setVisible(bool(checked))
        if checked and not self._camera_state[1]:
            self._paint_preview_placeholder()
        self._save_setting(SETTING_SHOW_CAMERA, bool(checked))

    def enter_sort(self) -> None:
        """This tab's Sort page came to the front."""
        # Assignments can have changed in Settings since the cards were last
        # drawn; they are cheap to re-read and never cached.
        self._refresh_sort_grid()
        self._refresh_notes_button()
        self._fetch_community_settings()

    def _refresh_sort_grid(self) -> None:
        """Assignments changed (bus event, template swap, mode switch, edit)."""
        self.slot_grid.refresh_assignments()
        self._update_sort_empty_state()

    def _update_sort_empty_state(self) -> None:
        """No board, no camera, nothing routed anywhere yet -> the guided panel.

        Re-evaluated on every indicator change (camera/serial connect changes)
        and every assignment change, never cached: any one of the three
        conditions clearing is enough to swap back to the real dashboard.
        """
        connected = self._camera_state[1] or self._serial_state[1]
        fresh_db = not any(int(h.get("slot", 0)) > 0 for h in self.config.headstamps)
        self.sort_stack.setCurrentIndex(1 if (not connected and fresh_db) else 0)

    def open_slot_editor(self, slot: int) -> None:
        """Edit what routes to one slot. The catch-all isn't configurable."""
        if int(slot) == 0:
            self.set_status(CATCH_ALL_HINT)
            return
        dialog = SlotAssignDialog(self.config, int(slot), self.sort_page)
        dialog.changed.connect(self._refresh_sort_grid)
        dialog.exec()
        self._refresh_sort_grid()

    def open_headstamp_assign(self) -> None:
        """Every headstamp in one table; the cards repaint on each edit while it is open.

        ``open()``, not ``exec()``: modal to the window but returning at once,
        so a test can drive the dialog it leaves in ``headstamp_assign_dialog``.
        """
        dialog = HeadstampAssignDialog(self.config, self.sort_page)
        dialog.changed.connect(self._refresh_sort_grid)
        dialog.finished.connect(self._on_headstamp_assign_closed)
        self.headstamp_assign_dialog = dialog
        dialog.open()

    def _on_headstamp_assign_closed(self, _result: int) -> None:
        dialog, self.headstamp_assign_dialog = self.headstamp_assign_dialog, None
        if dialog is not None:
            dialog.deleteLater()
        self._refresh_sort_grid()

    # ----- templates -----------------------------------------------------------

    def _refresh_templates(self) -> None:
        """Repopulate the combo for the active model + current run mode."""
        mode = self.config.slot_template_mode()
        self._templates = self.config.list_slot_templates(mode)
        active = self.config.active_slot_template(mode)
        self.template_combo.clear()
        self.template_combo.addItems([t.name for t in self._templates])
        for index, template in enumerate(self._templates):
            if template.id == active.id:
                self.template_combo.setCurrentIndex(index)
                break
        self.template_hint.setText("Package-mode layout" if mode == "package" else "")

    def _template_busy(self) -> bool:
        """Templates swap the whole layout, so keep them out of a live run."""
        if not self._is_running:
            return False
        self.notify(
            "Run in progress",
            "Stop the run before changing sorting templates — switching one reassigns every slot.",
        )
        return True

    def _on_template_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._templates):
            return
        target = self._templates[index]
        if self._template_busy() or self.config.activate_slot_template(target.id) is None:
            self._refresh_templates()  # snap the combo back to the active one
            return
        self._after_template_change(f"Loaded sorting template “{target.name}”.")

    def new_template(self) -> None:
        if self._template_busy():
            return
        mode = self.config.slot_template_mode()
        dialog = NewTemplateDialog(self.config, mode, self.config.active_slot_template(mode).name, self.sort_page)
        if dialog.exec() and dialog.created is not None:
            self._after_template_change(f"Created sorting template “{dialog.created.name}”.")

    def edit_template(self) -> None:
        if self._template_busy():
            return
        mode = self.config.slot_template_mode()
        dialog = EditTemplateDialog(
            self.config,
            self.config.active_slot_template(mode),
            can_delete=len(self.config.list_slot_templates(mode)) > 1,
            parent=self.sort_page,
        )
        if dialog.exec():
            self._after_template_change("Sorting templates updated.")

    def _after_template_change(self, status: str) -> None:
        """Counters are per-layout: a slot may hold another headstamp now."""
        self._refresh_templates()
        self._clear_counts()
        self._refresh_sort_grid()
        self.set_status(status)

    # ----- run options and counts ------------------------------------------------

    def _apply_package_visibility(self) -> None:
        enabled = self.package_check.isChecked()
        self.batch_caption.setVisible(enabled)
        self.batch_spin.setVisible(enabled)

    def _on_package_mode_toggled(self, enabled: bool) -> None:
        self.config.set_run_package_mode(bool(enabled))
        self._apply_package_visibility()
        # Counts, assignments and templates are all mode-specific.
        self._clear_counts()
        self._refresh_templates()
        self._refresh_sort_grid()

    def _on_batch_size_changed(self, value: int) -> None:
        self.config.set_run_package_size(int(value))
        self._refresh_sort_grid()

    def _clear_counts(self) -> None:
        self.slot_grid.reset_counts()
        self._master_count = 0
        self.master_count_label.setText("0")
        self._changed()

    def reset_counts(self) -> None:
        """Zero the dashboard's counters and the run's package batches."""
        self._clear_counts()
        reset = getattr(self.run_controller, "reset_package_counts", None)
        if reset is not None:
            reset()
        self.set_status("Counters reset.")

    def reset_slot_count(self, slot: int) -> None:
        """Package mode: empty one bin and let it refill while the run continues."""
        reset = getattr(self.run_controller, "reset_package_slot", None)
        if reset is not None:
            reset(int(slot))
        self.slot_grid.reset_slot(int(slot))
        self.set_status(f"Reset counter for slot {slot}.")

    # ----- active model --------------------------------------------------------

    def active_model(self) -> Any | None:
        if self.db is None:
            return None
        from ..data.repository import ModelRepo

        model_id = self.config.active_model_id
        return ModelRepo(self.db).get(model_id) if model_id is not None else None

    def _on_mode_changed(self) -> None:
        """This tab's active model changed: everything scoped to it is re-read."""
        self._clear_counts()
        self._refresh_templates()
        self._refresh_sort_grid()
        self.ai_page.refresh_mode()
        # The server's policy and the version prompt belonged to the old model.
        self._clear_community_settings()
        self._refresh_notes_button()
        # Everything on the Train page is scoped to the active model.
        self.train_page.refresh()
        # The sidebar inks, the library's Active column and the device line.
        self.window.on_tab_mode_changed(self)
        self._changed()

    # ----- indicators ---------------------------------------------------------

    def _set_camera_indicator(self, message: str, *, connected: bool) -> None:
        self._camera_state = (message, connected)
        self._paint_preview_placeholder()
        self._update_sort_empty_state()
        self._changed()

    def _set_serial_indicator(self, message: str, *, connected: bool) -> None:
        self._serial_state = (message, connected)
        self._update_sort_empty_state()
        self._changed()
        self.bus.post("serial/state", {"connected": connected, "message": message})

    def _camera_placeholder_text(self) -> str:
        return f"{CAMERA_DEAD_TEXT} — {CAMERA_DEAD_LINK}"

    def _paint_preview_placeholder(self) -> None:
        """A camera that isn't connected says so where the feed would be."""
        if self._camera_state[1]:
            return  # frames are arriving (or about to); the timer owns the label
        label = getattr(self, "preview_label", None)
        if label is not None:
            label.setText(self._camera_placeholder_text())
            label.setCursor(Qt.CursorShape.PointingHandCursor)

    def _on_preview_clicked(self) -> None:
        if not self._camera_state[1]:
            self._open_settings_section("Camera")

    # ----- camera ---------------------------------------------------------------

    def start_camera(self) -> None:
        # The saved choice, which is what this tab's camera was built from.
        index = int(self.config.camera.get("device_index", 0))
        blocker = self.devices.claim_camera(index, self.sorter_id)
        if blocker is not None:
            self.set_status(f"Camera {index} is {in_use_label(blocker)} — pick another in Settings → Camera.")
            self._set_camera_indicator(f"Camera: {in_use_label(blocker)}", connected=False)
            return
        try:
            if self.camera.start_preview():
                self._set_camera_indicator(
                    f"Camera: connected ({self.camera.width}x{self.camera.height})",
                    connected=True,
                )
            else:
                # The red dot alone left the user with nowhere to go (JL).
                self.bus.post("status/error", CAMERA_FAILED_STATUS)
                self._set_camera_indicator("Camera: failed to start", connected=False)
        except Exception as exc:
            self.bus.post("status/error", f"Camera error: {exc} — pick a device in Settings → Camera.")
            self._set_camera_indicator("Camera: error", connected=False)

    def refresh_preview(self) -> None:
        """One live-feed frame, painted by the window's preview timer (front tab only)."""
        # Hidden is the default; while hidden no frame is fetched or painted
        # (the camera's grab thread runs regardless).
        if not self.show_camera_check.isChecked():
            return
        frame = self.camera.latest_frame()
        if frame is None:
            return
        pixmap = QPixmap.fromImage(self.frame_to_image(frame))
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # ----- serial ---------------------------------------------------------------

    def _buffer_serial(self, kind: str, line: Any) -> None:
        self.serial_lines.append((kind, time.time(), str(line)))

    def _refuse_serial(self, port: str, blocker: str) -> None:
        self.set_status(f"{port} is {in_use_label(blocker)}.")
        self._set_serial_indicator(f"Serial: {port} {in_use_label(blocker)}", connected=False)

    def auto_connect_serial(self) -> None:
        """Try the saved port first, then walk the rest until one handshakes.

        The single-tab startup path, unchanged from before tabs existed. Ports
        are probed on a worker (``try_open`` waits out the handshake), ports
        another tab holds are never probed, and ``_after_connect`` is the
        shared tail with the Settings page's own Connect.
        """
        saved_port = (self.config.serial.get("port") or "").strip()
        if saved_port == EMULATED_PORT:
            broker = EmulatorBroker()
            broker.try_open()
            self._after_connect(broker, EMULATED_PORT)
            return

        available = serial_broker.list_serial_ports()
        held = {p: self.devices.serial_holder(p, self.sorter_id) for p in available}
        candidates: list[str] = []
        # The saved port is always probed, even if the filter below would
        # skip it — the user chose it once, so it is not a guess.
        if saved_port and saved_port in available and held.get(saved_port) is None:
            candidates.append(saved_port)
        for port in available:
            if port not in candidates and held.get(port) is None and serial_broker.is_probe_candidate(port):
                candidates.append(port)
        skipped = [p for p in available if p not in candidates and held.get(p) is None]
        if skipped:
            self.bus.post(
                "serial/note",
                "skipping (Bluetooth/pseudo, connect manually from Settings → Serial): " + ", ".join(skipped),
            )
        for port, holder in held.items():
            if holder is not None:
                self.bus.post("serial/note", f"skipping {port} ({in_use_label(holder)})")

        if not candidates:
            self.set_status("No serial ports detected.")
            self._set_serial_indicator("Serial: no ports", connected=False)
            return

        baud = int(self.config.serial.get("baud", 9600))
        probe_timeout = float(self.config.serial.get("handshake_timeout_s", serial_broker.HANDSHAKE_READ_TIMEOUT_S))

        def _probe() -> tuple[Any, str] | tuple[None, None]:
            for port in candidates:
                self.bus.post("status", f"Auto-connect: probing {port}…")
                self.bus.post("serial/note", f"probing {port} @ {baud}…")
                broker = serial_broker.SerialBroker(
                    port=port,
                    baud=baud,
                    require_serial_ready=True,
                    handshake_timeout_s=probe_timeout,
                )
                # Listen *before* the handshake: whatever the board says to a
                # probe that fails is the only evidence of why it failed.
                self._attach_serial_listeners(broker)
                if broker.try_open():
                    broker.start()
                    return broker, port
                self.bus.post("serial/note", f"{port} did not handshake")
            return None, None

        self.set_status(f"Auto-connecting to serial — {len(candidates)} port(s) to try…")
        self.run_worker(
            _probe,
            on_done=self._finalize_auto_connect,
            on_error=lambda exc: self.set_status(f"Auto-connect error: {exc}", level=ERROR),
        )

    def open_saved_port_blocking(self, port: str) -> Any | None:
        """Worker thread: open one saved port and handshake, touching no widget.

        The several-tab startup path (D8): the window runs this for each tab
        in turn on one worker, then hands every result to
        ``finish_saved_port`` on the main thread, so device claims happen in
        tab order.
        """
        baud = int(self.config.serial.get("baud", 9600))
        timeout = float(self.config.serial.get("handshake_timeout_s", serial_broker.HANDSHAKE_READ_TIMEOUT_S))
        self.bus.post("serial/note", f"opening saved port {port} @ {baud}…")
        broker = serial_broker.SerialBroker(
            port=port,
            baud=baud,
            require_serial_ready=True,
            handshake_timeout_s=timeout,
        )
        self._attach_serial_listeners(broker)
        if broker.try_open():
            broker.start()
            return broker
        self.bus.post("serial/note", f"{port} did not handshake")
        return None

    def finish_saved_port(self, port: str, broker: Any | None) -> None:
        """Main thread: the tail of ``open_saved_port_blocking``."""
        if broker is not None:
            self._after_connect(broker, port)
            return
        blocker = self.devices.serial_holder(port, self.sorter_id)
        if blocker is not None:
            self._refuse_serial(port, blocker)
            return
        self.set_status(f"No board responded on {port}.")
        self._set_serial_indicator(f"Serial: no board on {port}", connected=False)

    def _finalize_auto_connect(self, result: tuple[Any, str] | tuple[None, None]) -> None:
        broker, port = result
        if broker is None or port is None:
            self.set_status("No board responded on any port.")
            self._set_serial_indicator("Serial: no board found", connected=False)
            return
        self._after_connect(broker, port)

    def _attach_serial_listeners(self, broker: Any) -> None:
        """Fan a broker's traffic onto this tab's bus, once — a second attach doubles every line."""
        if getattr(broker, "_bus_listeners_attached", False):
            return
        broker.on_received.append(lambda line: self.bus.post("serial/rx", line))
        broker.on_sent.append(lambda line: self.bus.post("serial/tx", line))
        # Fires on the reader thread (or a failed writer's), so it goes through
        # the bus like everything else rather than touching a widget directly.
        on_disconnect = getattr(broker, "on_disconnect", None)
        if isinstance(on_disconnect, list):
            on_disconnect.append(lambda reason: self.bus.post("serial/disconnected", reason))
        broker._bus_listeners_attached = True

    def _after_connect(self, broker: Any, port: str, *, source: str = "auto") -> None:
        """Shared tail of every connect path: claim, listeners, persistence, controller."""
        blocker = self.devices.claim_serial(port, self.sorter_id)
        if blocker is not None:
            # Another tab took the port while this one's handshake ran.
            try:
                broker.stop()
            except Exception:
                pass
            self._refuse_serial(port, blocker)
            return
        self._attach_serial_listeners(broker)
        self.broker = broker
        baud = int(getattr(broker, "baud", self.config.serial.get("baud", 9600)))
        if port != (self.config.serial.get("port") or "") or baud != int(self.config.serial.get("baud", 9600)):
            self.config.serial["port"] = port
            self.config.serial["baud"] = baud
            self.config.save()
        self._set_serial_indicator(
            f"Serial: connected ({port} @ {getattr(broker, 'baud', '?')}) — {broker.firmware_version}",
            connected=True,
        )
        self.set_status(f"{'Auto-connected' if source == 'auto' else 'Connected'} to {port}.")
        self._rebuild_run_controller()
        # Pushed from the shared connect tail, so auto-connect and the Settings
        # page behave alike.
        if self.config.serial.get("init_on_startup", False):
            settings = dict(self.config.serial.get("init_settings", {}))
            if settings:
                self.run_worker(
                    lambda: broker.update_init_settings(settings),
                    on_done=lambda _r: self.set_status(f"Connected to {port}. Init settings pushed."),
                    on_error=lambda err: self.set_status(f"Init push failed: {err}", level=ERROR),
                )

    def disconnect_serial(self) -> None:
        """Stop the run, close the board and give the port back. Silent: callers report."""
        controller, self.run_controller = self.run_controller, None
        broker, self.broker = self.broker, None
        try:
            if controller is not None:
                controller.stop()
            if broker is not None:
                broker.stop()
        except Exception:
            pass
        self.devices.release_serial(self.sorter_id)
        self._update_run_buttons()

    def connect_serial(self, port: str | None = None) -> None:
        """Open one explicit port, chosen in Settings → Serial or the monitor.

        The emulator is opened inline (nothing blocks); a real port opens on a
        worker because ``try_open`` waits out the board's handshake.
        """
        if self.broker is not None:
            self.disconnect_serial()

        if port is None:
            port = (self.config.serial.get("port") or "").strip()
        if not port:
            self.set_status("No port selected.")
            self._set_serial_indicator("Serial: no port selected", connected=False)
            return
        blocker = self.devices.serial_holder(port, self.sorter_id)
        if blocker is not None:
            self._refuse_serial(port, blocker)
            return

        if port == EMULATED_PORT:
            broker: Any = EmulatorBroker()
            broker.try_open()
            self._after_connect(broker, port, source="manual")
            return

        baud = int(self.config.serial.get("baud", 9600))
        broker = serial_broker.SerialBroker(port=port, baud=baud, require_serial_ready=True)
        # As in the probe: a failed open should still leave a trace in the monitor.
        self._attach_serial_listeners(broker)
        self.set_status(f"Connecting to {port}…")

        def _open() -> bool:
            if not broker.try_open():
                return False
            broker.start()
            return True

        def _done(opened: bool) -> None:
            if not opened:
                self.set_status(f"Failed to open {port}.", level=ERROR)
                self._set_serial_indicator(f"Serial: failed to open {port}", connected=False)
                return
            self._after_connect(broker, port, source="manual")

        self.run_worker(
            _open,
            on_done=_done,
            on_error=lambda exc: self.set_status(f"Connect error: {exc}", level=ERROR),
        )

    def _on_serial_disconnected(self, reason: Any = None) -> None:
        """The link died on its own: say so, drop the board, stop the run.

        Nothing here reconnects. The board's arm position and the wheel's
        pipeline are only knowable while the link is up, so recovery is an
        explicit reconnect (which re-handshakes through ``try_open``), not a
        silent one — see ``SERIAL_LOST_TEXT``.
        """
        detail = str(reason or "link lost")
        port = str(getattr(self.broker, "port", "") or self.config.serial.get("port") or "")
        self.bus.post("serial/note", f"disconnected — {detail}")
        was_running = self._is_running
        self.disconnect_serial()
        self._set_serial_indicator(
            f"Serial: disconnected ({port})" if port else "Serial: disconnected",
            connected=False,
        )
        self.set_status(f"Serial disconnected — {detail}", level=ERROR)
        if was_running:
            self.beep()
            # Same shape as the package halt: a modal, queued out of the drain
            # so it can't re-enter it.
            QTimer.singleShot(0, self, lambda: self.notify(SERIAL_LOST_TITLE, SERIAL_LOST_TEXT))

    # ----- run ------------------------------------------------------------------

    def _rebuild_run_controller(self) -> None:
        if self.broker is None:
            return
        self.run_controller = RunController(
            config=self.config,
            broker=self.broker,
            camera=self.camera,
            bus=self.bus,
            db=self.db,
        )
        # A fresh controller carries a fresh FeedbackService, so the server's
        # policy has to be re-installed on it.
        self._apply_community_settings()
        self._refresh_sort_grid()
        self._update_run_buttons()

    def _ai_credentials_missing(self) -> bool:
        """HTTP classification can't run without an API key and a model name.

        Scoped to the HTTP paths: a local model never touches the HTTP
        client, so an unset key there is no reason to refuse a run. An
        active openai-mode model is checked against **its own** config — the
        same one `classify_active` will use — never the app-level one.
        """
        from ..data.models import is_openai_model

        model = classifier.active_model(self.db, model_id=self.config.active_model_id)
        if is_openai_model(model):
            cfg = model.ai_model_config if model is not None else None
            return cfg is None or not (cfg.api_key and cfg.model)
        if classifier.uses_local_inference(self.db, model_id=self.config.active_model_id):
            return False
        api = self.config.api
        return not (api.get("api_key") and api.get("model"))

    def _ready_to_sort(self) -> RunController | None:
        """Preflight: board, moderator notes, AI config, checkpoint, torch.

        Each check is asked of the layer that owns the answer, so the Qt shell
        never re-derives the rule. Returns the controller when a run may start.
        An unacknowledged moderator note gates Start (issue #29, A25). The
        dashboard's per-row Start goes through here too.
        """
        controller = self.run_controller
        if controller is None or self.broker is None:
            self.set_status("Connect to the board first (Settings → Serial).")
            return None
        from ..community.notes import unacknowledged

        if unacknowledged(self._stored_notes()):
            self.notify(NOTES_GATE_TITLE, NOTES_GATE_TEXT)
            return None
        if self._ai_credentials_missing():
            self.notify(
                "AI not configured",
                "Set the endpoint, API key and model on the AI Config page first.",
            )
            return None
        model_id = self.config.active_model_id
        problem = classifier.checkpoint_problem(self.db, model_id=model_id)
        if problem is not None:
            self.notify("Model not ready", problem)
            return None
        if classifier.uses_local_inference(self.db, model_id=model_id) and not self.ensure_torch(
            self.start_run,
            reason="Sorting needs PyTorch",
            # The model decides block-vs-offer on an outdated torch, so the
            # gate needs to know whose checkpoint is about to be loaded.
            model=classifier.active_model(self.db, model_id=model_id),
        ):
            # The gate re-enters start_run after a successful install.
            return None
        return controller

    @staticmethod
    def _set_button_role(button: QPushButton, object_name: str) -> None:
        """Swap a button's palette role. QSS matches on the objectName, and a
        live widget has to be re-polished for the new rule to take."""
        if button.objectName() == object_name:
            return
        button.setObjectName(object_name)
        style = button.style()
        style.unpolish(button)
        style.polish(button)

    def toggle_run(self) -> None:
        """The one button's handler; the run state decides which half it is."""
        if self._is_running:
            self.stop_run()
        else:
            self.start_run()

    def start_run(self) -> None:
        controller = self._ready_to_sort()
        if controller is None or self._is_running:
            return
        self._refresh_sort_grid()
        controller.start()

    def stop_run(self) -> None:
        if self.run_controller is not None:
            self.run_controller.stop()
            self.set_status("Stopping…")

    def manual_feed(self) -> None:
        controller = self._ready_to_sort()
        if controller is None or self._is_running:
            return
        # One cycle blocks on the board; the bus carries the result back.
        self.run_worker(controller.cycle_once)

    def _on_run_started(self) -> None:
        # Counts survive Stop/Start on purpose: operators stop to clear a jam
        # and restart mid-tray. Only the explicit resets clear.
        self._set_running(True)

    def _on_run_stopped(self) -> None:
        self._set_running(False)
        # Terminal status, or whatever was in flight ("Stopping…",
        # "Classifying…") reads as stuck forever (Seth).
        self.set_status("Run stopped.")

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        self._update_run_buttons()
        self._changed()

    def _update_run_buttons(self) -> None:
        connected = self.broker is not None
        running = self._is_running
        # A run that outlives its board still has to be stoppable.
        self.run_button.setEnabled(connected or running)
        self.run_button.setText(RUN_STOP_TEXT if running else RUN_START_TEXT)
        self._set_button_role(self.run_button, "danger" if running else "action")
        self.action_buttons["Manual feed"].setEnabled(connected and not running)
        for button in (self.run_button, self.action_buttons["Manual feed"]):
            button.setToolTip("" if connected else "Connect to the board first")
        # The Train page's Feed drives the same board.
        train_page = getattr(self, "train_page", None)
        if train_page is not None:
            train_page.refresh_connection()

    def _on_run_result(self, result: Any) -> None:
        # `run/result` carries a slot even for a failed cycle, so `ok` is what
        # decides whether a case actually landed anywhere.
        if not isinstance(result, dict) or not result.get("ok"):
            return
        self.slot_grid.increment(int(result.get("slot") or 0))
        self._master_count += 1
        self.master_count_label.setText(str(self._master_count))
        self._changed()

    def _on_run_history(self, payload: Any) -> None:
        """The current case: its crop and the call made on it, and the dock's buffer."""
        if not isinstance(payload, dict):
            return
        # The number is stamped onto the posted dict itself: this handler is
        # subscribed before the history panel's, so the panel reads the same
        # number off the payload instead of keeping a count of its own.
        self.history_case_number += 1
        payload["number"] = self.history_case_number
        self.history_records.append(payload)
        image = payload.get("image")
        self._show_crop(image)
        confidence = float(payload.get("confidence", 0) or 0)
        floor = float(getattr(self.config, "run_confidence_floor", 0) or 0)
        label = str(payload.get("label") or "(empty)")
        parent = payload.get("parent")
        self._current_result = (
            f"{parent} · {label}" if parent else label,
            confidence,
            floor <= 0 or confidence >= floor,
        )
        self._paint_current_result()
        self._changed()

    def _paint_current_result(self) -> None:
        """The confidence colour is baked in, so a theme switch re-runs this."""
        if self._current_result is None:
            self.result_label.setText(RESULT_EMPTY_TEXT)
            self.result_confidence_label.setText(RESULT_EMPTY_CONFIDENCE)
            self.result_confidence_label.setStyleSheet(f"color: {self.palette_colors['text_muted']};")
            return
        label, confidence, above_floor = self._current_result
        self.result_label.setText(label)
        self.result_confidence_label.setText(f"{confidence:.0f}%")
        color = self.palette_colors["success" if above_floor else "warning"]
        self.result_confidence_label.setStyleSheet(f"color: {color};")

    def _show_crop(self, image: Any) -> None:
        """The headstamp as the classifier saw it — the column's primary panel."""
        if not isinstance(image, np.ndarray) or image.size == 0:
            return
        self.last_crop = image
        self.crop_label.set_source(QPixmap.fromImage(self.frame_to_image(image)))

    def _on_package_full(self, payload: Any) -> None:
        data = payload if isinstance(payload, dict) else {}
        self.beep()
        self.set_status(f"Slot {data.get('slot')} batch full ({data.get('count')}). Reset it to refill.")

    def _on_out_of_brass(self, payload: Any) -> None:
        data = payload if isinstance(payload, dict) else {}
        flushed = data.get("flushed", 0)
        self.beep()
        self.set_status(f"Out of brass — run finished. {flushed} in-flight case(s) were flushed to their slots.")

    def _on_package_halt(self, payload: Any) -> None:
        label = (payload or {}).get("label") if isinstance(payload, dict) else None
        self.beep()
        message = (
            f"Run stopped — every slot for “{label or '?'}” is full. "
            "Empty the bins, reset their counters, then Start again."
        )
        self.set_status(message)
        # Operators rely on the dialog here, not just the status line. Queued
        # single-shot: a modal straight from a bus handler re-enters the drain.
        QTimer.singleShot(0, self, lambda: self.notify("Package complete", message))

    # ----- community model settings (A24/A25) ------------------------------------

    def _fetch_community_settings(self) -> None:
        """One ``FetchModelSettings`` per Sort-page entry, on a worker.

        Gated on the active model being a community one *and* an account
        already existing — reading the token cache must stay behind a
        user action, and a signed-out user has nothing to fetch with.
        Everything downstream fails open: a ``None`` result leaves the local
        floor and opt-in in charge and raises no prompt.
        """
        if self._settings_fetch_busy or self.db is None:
            return
        from ..community.feedback import is_community_model

        community_page = self.window.community_page
        model = self.active_model()
        # `model is None` is folded into the guard so the type checker can
        # narrow it for the rest of this method.
        if model is None or not is_community_model(model) or not community_page.is_signed_in():
            self._clear_community_settings()
            self._refresh_notes_button()
            return
        uid = str(model.community_model_uid)
        model_id = int(model.id)
        name = model.name
        # The row's own version; 0/absent means "unknown", which never nags.
        installed = int(model.model_version or 0)
        api_factory = community_page.api_factory
        find_entry = community_page.find_catalogue_entry
        self._settings_fetch_busy = True

        def work() -> tuple[Any, Any]:
            api = api_factory()
            settings = api.fetch_model_settings(uid)
            info = None
            if settings is not None and installed > 0 and settings.version > installed:
                # Only then, and only to fill the dialog + drive the update:
                # the settings response carries a version number and nothing
                # else about the published model.
                try:
                    info = find_entry(uid, api)
                except Exception:
                    info = None
            return settings, info

        self.run_worker(
            work,
            on_done=lambda payload: self._on_community_settings(model_id, uid, name, installed, payload),
            on_error=lambda _exc: self._on_community_settings_failed(),
        )

    def _on_community_settings(self, model_id: int, uid: str, name: str, installed: int, payload: Any) -> None:
        self._settings_fetch_busy = False
        settings, info = payload if isinstance(payload, tuple) else (None, None)
        if settings is None:
            self._on_community_settings_failed()
            return
        self._community_settings = (model_id, settings)
        self._apply_community_settings()
        if settings.blocked:
            # The contract prefers saying so over going quiet — once per fetch,
            # never per case.
            self.set_status(FEEDBACK_BLOCKED_STATUS)

        from ..community import notes as notes_store

        stored = notes_store.merge(self.db, uid, settings.notes)
        self._refresh_notes_button()

        newer = installed > 0 and settings.version > installed and info is not None
        self._model_update = (name, installed, int(settings.version), info) if newer else None
        self._changed()

        if notes_store.unacknowledged(stored):
            # Queued: this runs inside a bus drain, and a modal here would
            # re-enter it (CLAUDE.md §5).
            QTimer.singleShot(0, self, self.open_notes_dialog)

    def _on_community_settings_failed(self) -> None:
        """Offline / refused / garbage: back to purely local behaviour."""
        self._settings_fetch_busy = False
        self._clear_community_settings()
        self._refresh_notes_button()

    def _clear_community_settings(self) -> None:
        self._community_settings = None
        self._model_update = None
        self._changed()
        if self.run_controller is not None:
            self.run_controller.clear_community_settings()

    def _apply_community_settings(self) -> None:
        """Push the last fetch's policy into the (possibly rebuilt) controller."""
        if self.run_controller is None:
            return
        if self._community_settings is None:
            self.run_controller.clear_community_settings()
            return
        model_id, settings = self._community_settings
        self.run_controller.apply_community_settings(
            model_id,
            confidence_floor=int(settings.confidence_floor),
            feedback_enabled=bool(settings.feedback_enabled),
            blocked=bool(settings.blocked),
            wish_list=settings.wish_list,
        )

    @property
    def model_update_version(self) -> int | None:
        """The newer published version the status-bar button offers, if any."""
        return None if self._model_update is None else self._model_update[2]

    def model_update_dialog(self) -> Any | None:
        """The dialog, wired but not shown — tests drive its buttons directly."""
        if self._model_update is None:
            return None
        from .dialog_model_update import build_model_update_dialog

        name, installed, available, info = self._model_update
        dialog = build_model_update_dialog(
            self.window,
            model_name=name,
            installed_version=installed,
            available_version=available,
            info=info,
            on_update=self.start_model_update,
        )
        # "Not now" (or closing) drops the affordance; the next Sort-page entry
        # re-fetches and raises it again. No permanent dismissal.
        dialog.rejected.connect(self.dismiss_model_update)
        return dialog

    def dismiss_model_update(self) -> None:
        self._model_update = None
        self._changed()

    def _open_model_update_dialog(self) -> None:
        dialog = self.model_update_dialog()
        if dialog is not None:
            dialog.exec()

    def start_model_update(self) -> None:
        """Accepting the prompt: the Community page's own download+import path."""
        if self._model_update is None:
            return
        _name, _installed, _available, info = self._model_update
        self._model_update = None
        self._changed()
        self.window.community_page.start_update(info)

    # ----- moderator notes -------------------------------------------------------

    def _community_uid(self) -> str | None:
        model = self.active_model()
        uid = getattr(model, "community_model_uid", None)
        return str(uid) if uid else None

    def _stored_notes(self) -> list[Any]:
        uid = self._community_uid()
        if uid is None or self.db is None:
            return []
        from ..community import notes as notes_store

        return notes_store.load(self.db, uid)

    def _refresh_notes_button(self) -> None:
        notes = self._stored_notes()
        self.notes_button.setText(NOTES_BUTTON.format(count=len(notes)))
        self.notes_button.setVisible(bool(notes))

    def notes_dialog(self) -> Any | None:
        """The dialog, wired but not shown — tests drive its buttons directly."""
        from .dialog_community_notes import build_community_notes_dialog

        uid = self._community_uid()
        if uid is None:
            return None
        model = self.active_model()
        return build_community_notes_dialog(
            self.window,
            model_name=getattr(model, "name", ""),
            notes=self._stored_notes(),
            on_acknowledge=lambda ids: self._acknowledge_notes(uid, ids),
        )

    def _open_notes_dialog(self) -> None:
        dialog = self.notes_dialog()
        if dialog is not None:
            dialog.exec()

    def _acknowledge_notes(self, uid: str, ids: list[int]) -> None:
        from ..community import notes as notes_store

        notes_store.acknowledge(self.db, uid, ids)
        self._refresh_notes_button()

    # ----- lifecycle -------------------------------------------------------------

    def reload(self) -> None:
        """Re-read this tab's settings and everything scoped to its model.

        After something rewrote the database underneath the tab — the
        Windows-app import — rather than a surgical refresh.
        """
        self.config.load()
        self._on_mode_changed()

    def shutdown(self) -> None:
        """Stop the run, close the board and the camera, and give both back.

        What closing a tab and closing the window both do; every step is
        best-effort, because a device that fails to close must not keep the
        others open.
        """
        try:
            if self.run_controller is not None:
                self.run_controller.stop()
        except Exception:
            pass
        try:
            if self.broker is not None:
                self.broker.stop()
        except Exception:
            pass
        try:
            self.camera.stop()
        except Exception:
            pass
        try:
            self.serial_log.close()
        except Exception:
            pass
        self.devices.release_all(self.sorter_id)
