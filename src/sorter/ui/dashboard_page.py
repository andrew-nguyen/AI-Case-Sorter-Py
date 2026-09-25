"""The "All sorters" dashboard: one row per sorter tab, live.

The fixed first tab of the window's tab strip. Each row shows what an operator
walking past several machines wants at a glance: the sorter's name, its board
port and connection state, its camera and connection state, the active model,
whether it is running, how many cases it has sorted this run, the last
classification with its confidence, and a thumbnail of the last crop.

**Fed, not polled.** A tab calls ``window.on_tab_changed`` whenever its run,
device or result state moves, and that posts ``sorters/updated <id>`` on the
app bus; the dashboard re-reads that one tab's ``snapshot()`` and repaints its
row. Adding, closing or renaming a tab calls ``rebuild()``.

**Start/Stop is the tab's own button.** A row's Start/Stop calls the tab's
``toggle_run``, so the pre-flight checks (board connected, moderator notes
acknowledged, AI Config credentials, a usable checkpoint, the PyTorch gate)
are the same code the Sort page runs. Those are genuine per-machine controls,
which is why this table embeds buttons where the Models and Community tables
deliberately don't (CLAUDE.md §5). The table is therefore rebuilt, never
sorted: an item widget does not survive a sort.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QBrush, QColor, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .history_view import bgr_to_pixmap
from .sorter_tab import RUN_START_TEXT, RUN_STOP_TEXT, SorterTab

DASHBOARD_TITLE = "All sorters"
DASHBOARD_HINT = "Every sorter at a glance. Click a name to open its tab."

COLUMNS = ("Sorter", "Board", "Camera", "Model", "Run", "Cases", "Last result", "Last crop", "")
COL_NAME, COL_BOARD, COL_CAMERA, COL_MODEL, COL_RUN, COL_CASES, COL_RESULT, COL_CROP, COL_ACTION = range(len(COLUMNS))

THUMB_SIZE = 56
NO_RESULT = "—"
RUN_STATE_RUNNING = "Running"
RUN_STATE_IDLE = "Idle"


def describe_board(snapshot: dict[str, Any]) -> str:
    """Port and state in one cell: "COM3 · Connected", or the indicator text alone."""
    port = snapshot.get("port") or ""
    state = snapshot.get("serial") or ""
    if port and port not in state:
        return f"{port} · {state}" if state else port
    return state or NO_RESULT


def describe_result(snapshot: dict[str, Any]) -> str:
    result = snapshot.get("result")
    if not result:
        return NO_RESULT
    label, confidence, _above_floor = result
    return f"{label} ({confidence:.0f}%)"


class DashboardPage(QWidget):
    """The table. ``win`` is the main window; rows follow ``win.tabs`` order."""

    def __init__(self, win: Any) -> None:
        super().__init__()
        self._win = win
        self.run_buttons: dict[int, QPushButton] = {}
        column = QVBoxLayout(self)
        column.setContentsMargins(12, 12, 12, 12)
        column.setSpacing(8)
        hint = QLabel(DASHBOARD_HINT, self)
        hint.setObjectName("mutedLabel")
        column.addWidget(hint)
        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setObjectName("dashboardTable")
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.verticalHeader().setVisible(False)
        self.table.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.table.verticalHeader().setDefaultSectionSize(THUMB_SIZE + 8)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_MODEL, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._on_cell_clicked)
        column.addWidget(self.table, 1)
        win.bus.subscribe("sorters/updated", self._on_sorter_updated)
        self.rebuild()

    # ----- structure ---------------------------------------------------------

    def rebuild(self) -> None:
        """One row per tab, in tab order. Called when tabs are added, closed or renamed."""
        self.table.setRowCount(0)
        self.run_buttons.clear()
        for row, tab in enumerate(self._win.tabs):
            self.table.insertRow(row)
            for col in range(COL_ACTION):
                self.table.setItem(row, col, QTableWidgetItem())
            button = QPushButton(RUN_START_TEXT, self.table)
            button.clicked.connect(lambda _checked=False, t=tab: self._toggle(t))
            self.table.setCellWidget(row, COL_ACTION, button)
            self.run_buttons[tab.sorter_id] = button
            self._paint_row(row, tab)

    def refresh(self) -> None:
        """Re-read every row, e.g. when the dashboard is brought to the front."""
        if self.table.rowCount() != len(self._win.tabs):
            self.rebuild()
            return
        for row, tab in enumerate(self._win.tabs):
            self._paint_row(row, tab)

    def apply_palette(self) -> None:
        """State colours are item brushes, out of the stylesheet's reach."""
        self.refresh()

    # ----- updates -----------------------------------------------------------

    def _on_sorter_updated(self, sorter_id: Any) -> None:
        for row, tab in enumerate(self._win.tabs):
            if tab.sorter_id == sorter_id:
                if row < self.table.rowCount():
                    self._paint_row(row, tab)
                else:
                    self.rebuild()
                return

    def _paint_row(self, row: int, tab: Any) -> None:
        snap = tab.snapshot()
        colors = self._win.palette_colors

        def put(col: int, text: str, *, ok: bool | None = None) -> QTableWidgetItem:
            item = self.table.item(row, col)
            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(row, col, item)
            item.setText(text)
            if ok is None:
                item.setData(Qt.ItemDataRole.ForegroundRole, None)
            else:
                item.setForeground(QBrush(QColor(colors["success" if ok else "text_muted"])))
            return item

        name = put(COL_NAME, snap["name"])
        name.setToolTip(f"Open {snap['name']}")
        font = name.font()
        font.setUnderline(True)
        name.setFont(font)
        put(COL_BOARD, describe_board(snap), ok=bool(snap["serial_connected"]))
        put(COL_CAMERA, snap["camera"] or NO_RESULT, ok=bool(snap["camera_connected"]))
        put(COL_MODEL, snap["model"])
        running = bool(snap["running"])
        put(COL_RUN, RUN_STATE_RUNNING if running else RUN_STATE_IDLE, ok=running)
        put(COL_CASES, str(snap["count"]))
        result = put(COL_RESULT, describe_result(snap))
        if snap["result"]:
            result.setForeground(QBrush(QColor(colors["success" if snap["result"][2] else "warning"])))
        crop = put(COL_CROP, "")
        crop.setIcon(QIcon(bgr_to_pixmap(snap["crop"], THUMB_SIZE)) if snap["crop"] is not None else QIcon())
        button = self.run_buttons.get(tab.sorter_id)
        if button is not None:
            button.setText(RUN_STOP_TEXT if running else RUN_START_TEXT)
            # Same two faces as the Sort page's run button, and the same
            # rule that a run outliving its board must stay stoppable.
            SorterTab._set_button_role(button, "danger" if running else "action")
            button.setEnabled(bool(snap["connected"]) or running)

    # ----- actions -----------------------------------------------------------

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if col == COL_NAME and 0 <= row < len(self._win.tabs):
            self._win.show_tab(self._win.tabs[row])

    def _toggle(self, tab: Any) -> None:
        if tab in self._win.tabs:
            tab.toggle_run()


def build_dashboard_page(win: Any) -> DashboardPage:
    return DashboardPage(win)
