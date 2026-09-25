"""The PySide6 shell: tab strip, activity sidebar, stacked pages, docks, status bar.

This is the app's only UI; ``python -m sorter`` lands here. Every colour comes
from ``ui/palettes.py``, which ``ui/theme.py`` renders as QSS.

**Sorter tabs.** Each machine the app drives is a ``SorterTab``
(``ui/sorter_tab.py``): its own config scope, event bus, camera, serial
broker, run controller, and its own Sort, Train and AI Config pages and
Camera / Serial / Image Processing settings sections. The window is what they
share: one sidebar, one outer page stack, the Models and Community pages,
the Theme and Import from Windows settings sections, the docks, the status
bar, sign-in and the database. A per-tab surface is an inner
``QStackedWidget`` holding every tab's page; bringing a tab to the front
(``show_tab``) points each inner stack at that tab's page, so switching tabs
never rebuilds or reparents anything and every background tab keeps running.
The fixed first tab, "All sorters", is the dashboard (``ui/dashboard_page.py``).

The panels are Qt Advanced Docking System dock widgets (see ``_build_dock``
and ``DOCK_HOMES``): serial monitor at the bottom, classification history,
user guide, themes and messages on the right, all but the monitor closed until
asked for. The serial monitor and the history panel follow the front tab
(``retarget``). The sidebar+pages are the manager's *central* widget, which
is what makes them a fixed anchor the panels arrange around.

Every bus (the window's own ``bus`` for models, community, dashboard updates
and workers, plus one per tab) is drained by a single 50 ms ``QTimer``
(``drain_all``): workers post, the main thread dispatches.

Scope and rationale: docs/ui-modernization.md ("Sorter tabs").
"""

from __future__ import annotations

import base64
import html
import itertools
import logging
import os
import sys
import threading
import traceback
from collections.abc import Callable
from typing import Any

import PySide6QtAds as ads
from PySide6.QtCore import (
    QByteArray,
    QEvent,
    QSize,
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTabBar,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..control.events import EventBus
from ..data.config import Config
from ..data.sorters import (
    create_sorter,
    delete_sorter,
    ensure_default_sorter,
    rename_sorter,
    set_front_sorter_id,
)
from ..hardware.serial_emulator import EMULATED_PORT
from ..ml import classifier, local_inference
from ..paths import app_data_dir
from .community_page import build_community_page
from .dashboard_page import DASHBOARD_TITLE, build_dashboard_page
from .device_registry import DeviceRegistry
from .dialog_winforms_import import (
    SECTION_NAME as WINFORMS_IMPORT_SECTION,
)
from .dialog_winforms_import import (
    build_winforms_import_section,
    maybe_offer_first_run,
)
from .help_viewer import build_help_window, topic_for
from .history_view import build_history_view
from .icons import AI_CONFIG, COMMUNITY, MODELS, SETTINGS, SORT, TRAIN, app_icon
from .icons import icon as vector_icon
from .message_log import ERROR, INFO, MessageLog
from .messages_view import build_messages_view
from .models_page import build_models_page
from .palettes import (
    SETTING_CUSTOM_THEMES,
    SETTING_THEME,
    THEMES,
    load_custom_themes,
    resolve_theme,
    theme_names,
)
from .serial_monitor import build_serial_monitor
from .sorter_tab import MODEL_UPDATE_BUTTON, TAB_ACTIVITIES, TAB_SETTINGS_SECTIONS, SorterTab
from .theme import build_stylesheet, unavailable_ink
from .torch_gate import TorchGate

log = logging.getLogger(__name__)

PREVIEW_FPS = 20
SIDEBAR_WIDTH = 84
# Defensive: every SETTINGS_SECTIONS entry has a builder, so nothing renders
# this unless a section is added without one.
PLACEHOLDER_TEXT = "This section has no settings yet."

# Sidebar: (icon name, page name), in three groups separated by hairlines. The
# icons are drawn from ui/icons.py and inked by the live palette
# (_paint_sidebar_icons) — emoji read as artwork and themed badly.
AI_CONFIG_ACTIVITY = "AI Config"
# The always-live surfaces.
ACTIVITIES = (
    (SORT, "Sort"),
    (MODELS, "Models"),
    (COMMUNITY, "Community"),
)
# Below a separator: the two ways a classifier is taught, of which exactly one
# is live at a time (Seth on AI Config: "it takes the place of the training
# screen; it is analogous to training for an LLM"). Both are always visible —
# the muted one explains itself when clicked (JL).
MODE_ACTIVITIES = (
    (TRAIN, "Train"),
    (AI_CONFIG, AI_CONFIG_ACTIVITY),
)
# Below a second separator, and in the flow rather than pinned under a stretch:
# pinned to the bottom it fell off-screen at a modest window height (JL, on a
# Mac), which is the one entry that must be reachable from anywhere.
SETTINGS_ACTIVITY = (SETTINGS, "Settings")
# One line each, always set, saying only whether this entry is the live one.
ACTIVITY_TOOLTIP_LIVE = "Classification uses this now"
TRAIN_TOOLTIP_MUTED = "Activates when a local model is active — see Models"
AI_CONFIG_TOOLTIP_MUTED = "Activates when 'Use AI Config' or an OpenAI model is selected on Models"
SIDEBAR_ICON_SIZE = 26
# "Import from Windows" is last: it is a one-off errand, not a knob, and its
# name is also a GUIDE.md heading (help_viewer slugifies section names
# straight to an anchor, which tests/unit/ui/test_help.py pins).
SETTINGS_SECTIONS = ("Camera", "Serial", "Image Processing", "Theme", WINFORMS_IMPORT_SECTION)
# On every dock's tab: QtAds's drop overlays show where a panel *can* go once
# a drag starts, but nothing hints that it can be dragged at all (JL).
DOCK_DRAG_HINT = "Drag this tab to move the panel; View \u2192 Re-dock panels brings it home."

# The tab strip. "+" appends a sorter; a running sorter's tab carries a dot in
# the palette's action role (painted, because no stylesheet reaches one tab).
NEW_SORTER_TOOLTIP = "Add a sorter (one tab per machine)"
CLOSE_SORTER_TOOLTIP = "Close this sorter"
RENAME_TITLE = "Rename sorter"
RENAME_LABEL = "Sorter name:"
CLOSE_TITLE = "Close sorter"
CLOSE_TEXT = (
    "“{name}” is {state}. Closing it stops the run and disconnects its board and "
    "camera. Its models and images stay in the library."
)
RUNNING_TAB_TOOLTIP = "Running"
TAB_MARKER_SIZE = 10

# Persisted window/session state (JL, increment 14): dock layout + the model
# table's column widths, the same _load_setting/_save_setting pattern used
# for the theme choice.
SETTING_WINDOW_STATE = "ui.window_state"
SETTING_MODELS_COLUMNS = "ui.models_columns"

MIN_WINDOW_SIZE = (960, 660)

# Each panel's home: where it is built, and where View → Re-dock returns it.
DOCK_HOMES = (
    ("serial_dock", ads.BottomDockWidgetArea),
    ("history_dock", ads.RightDockWidgetArea),
    ("help_dock", ads.RightDockWidgetArea),
    ("themes_dock", ads.RightDockWidgetArea),
    ("messages_dock", ads.RightDockWidgetArea),
)

MESSAGES_HINT = "Click to see every recent message in full (View → Messages)."


def _configure_dock_manager() -> None:
    """QtAds config flags. Static/global, so they must precede the first manager.

    ``DisableStylesheet`` is the load-bearing one: QtAds otherwise installs its
    own ~10 KB sheet on the manager, which — being nearer the dock widgets than
    the window's — wins over ours and freezes the panels at one hard-coded
    palette. With it off, theme.py's ``ads--*`` rules are the only thing
    painting them.
    """
    flags = ads.CDockManager.eConfigFlag
    for flag in (
        flags.DisableStylesheet,
        flags.HideSingleCentralWidgetTitleBar,  # the pages area is chrome, not a panel
        flags.OpaqueSplitterResize,
        flags.FocusHighlighting,
        flags.DockAreaHasUndockButton,
        flags.DockAreaHasCloseButton,
    ):
        ads.CDockManager.setConfigFlag(flag, True)


class QtMainWindow(QMainWindow):
    def __init__(self, config: Any, *, auto_connect: bool = True) -> None:
        """Build the shell and one tab per saved sorter.

        ``config`` is the front sorter's ``Config`` (``__main__`` builds it);
        every other tab loads its own. Nothing here forwards to the front tab:
        a caller that means a sorter says which (``current_tab``, ``tabs``).
        """
        super().__init__()
        self.db = config.db
        # The app bus: models/changed, community/*, sorters/updated, status,
        # and every run_worker result. Each tab has its own bus for the rest.
        self.bus = EventBus()
        self._worker_tokens = itertools.count()
        self._muted_labels: list[QLabel] = []
        self.tabs: list[SorterTab] = []
        self._close_buttons: dict[int, QToolButton] = {}
        # False until the tab strip exists; a tab's construction reports its
        # state before there is anywhere to show it.
        self._shell_ready = False
        # The activity a sorter tab shows; kept while the dashboard is in front.
        self._activity = "Sort"
        self.devices = DeviceRegistry(name_of=self._sorter_name)
        # Before _build_ui: set_status records into it from the first page on.
        self.status_log = MessageLog()

        # Modal seams (CLAUDE.md §5): instance attributes, so a test replaces
        # them and nothing blocks offscreen.
        self.ask_text: Callable[[str, str, str], str | None] = self._ask_text
        self.confirm_close_tab: Callable[[SorterTab], bool] = self._confirm_close_tab

        self.setMinimumSize(*MIN_WINDOW_SIZE)

        load_custom_themes(self._load_setting(SETTING_CUSTOM_THEMES))
        self.theme_name = resolve_theme(self._load_setting(SETTING_THEME))
        self.palette_colors = THEMES[self.theme_name]

        # The one sanctioned front door for anything needing local inference.
        self.ensure_torch = TorchGate(self)

        # Before the shell: every tab builds its own pages, which read the
        # palette, the torch gate and the device registry above.
        roster = ensure_default_sorter(self.db)
        for record in roster:
            cfg = config if record.id == config.sorter_id else Config(self.db, sorter_id=record.id).load()
            self.tabs.append(SorterTab(self, cfg, record.name))
        self.current_tab = next((t for t in self.tabs if t.sorter_id == config.sorter_id), self.tabs[0])

        self.setWindowTitle(f"AI Case Sorter OSS - v{__version__} · GPL-3.0")
        # The headstamp mark, in one fixed neutral (see icons.APP_ICON_COLOR):
        # a taskbar owns its own background, so this one must not follow the
        # live palette. Set application-wide as well, so dialogs inherit it.
        window_icon = app_icon()
        self.setWindowIcon(window_icon)
        QGuiApplication.setWindowIcon(window_icon)
        self._build_ui()
        self._apply_theme(self.theme_name)
        self._restore_window_state()

        # community_page posts "status" on the app bus.
        self.bus.subscribe("status", self.set_status)
        self.bus.subscribe("status/error", lambda msg: self.set_status(msg, level=ERROR))
        self.bus.subscribe("status/progress", lambda msg: self.set_status(msg, progress=True))
        self._bus_timer = QTimer(self)
        self._bus_timer.timeout.connect(self.drain_all)
        self._bus_timer.start(50)

        # One live feed at a time: the front tab's. Background cameras keep
        # grabbing; nobody is looking at their frames.
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(lambda: self.current_tab.refresh_preview())
        self._preview_timer.start(int(1000 / PREVIEW_FPS))

        self.auth: Any | None = None
        self._update_info: Any | None = None
        self._pending_update: Any | None = None
        if auto_connect:
            # A returning signed-in user finds Community present at launch.
            # Reads the token cache only; never constructed in tests.
            try:
                from ..community.auth import AuthManager

                self.auth = AuthManager()
            except Exception:
                self.auth = None
            self.community_page.refresh_auth_state()
            # The shell opens on Sort, so nothing would otherwise "enter" it.
            self.current_tab.enter_sort()
            for tab in self.tabs:
                tab.start_camera()
            self._auto_connect_serial()
            self._warm_device_indicator()
            QTimer.singleShot(2500, self, self._startup_update_check)
            # After the shell is up, so the dialog has a parent to centre on.
            # Silent unless a Windows install is actually there (issue #98).
            QTimer.singleShot(0, self, self._offer_winforms_import)
        self._apply_auth_visibility()

    # ----- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        # Built here rather than with the rest of the status bar (below): the
        # sidebar's own construction enters the Sort page, which paints it.
        # Same quiet role as the app-update button: it appears only when the
        # front tab's Sort-page fetch found a newer published version.
        self.model_update_button = QPushButton(self)
        self.model_update_button.setObjectName("update")
        self.model_update_button.clicked.connect(lambda: self.current_tab.open_model_update_dialog())
        self.model_update_button.hide()

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(self._build_tab_strip(central))
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.pages = QStackedWidget(central)
        self._pages_by_name: dict[str, QWidget] = {}
        # Per-tab surfaces: one inner stack each, holding every tab's page.
        self.tab_stacks: dict[str, QStackedWidget] = {
            name: QStackedWidget() for name in (*TAB_ACTIVITIES, *TAB_SETTINGS_SECTIONS)
        }
        for tab in self.tabs:
            self._add_tab_pages(tab)
        self._show_tab_pages(self.current_tab)
        for name in TAB_ACTIVITIES:
            self._add_page(name, self.tab_stacks[name])
        self._add_page("Models", self._build_models_page())
        self._add_page("Community", self._build_community_page())
        self._add_page("Settings", self._build_settings_page())
        self.dashboard_page = build_dashboard_page(self)
        self._add_page(DASHBOARD_TITLE, self.dashboard_page)
        body.addWidget(self._build_sidebar())
        body.addWidget(self.pages, 1)
        layout.addLayout(body, 1)

        # QtAds owns the docking; the manager installs itself as the window's
        # central widget, and the tabs+sidebar+pages become *its* central
        # area: a fixed, unclosable, undraggable anchor the panels arrange
        # around.
        _configure_dock_manager()
        self.dock_manager = ads.CDockManager(self)
        self.central_dock = ads.CDockWidget(self.dock_manager, "Workspace")
        self.central_dock.setWidget(central)
        self.dock_manager.setCentralWidget(self.central_dock)

        # Docks are built before the menus: View hosts their toggle actions.
        self._build_serial_dock()
        self._build_history_dock()
        self._build_help_dock()
        self._build_themes_dock()
        self._build_messages_dock()
        self._build_menus()

        # Where local classification runs (e.g. "Inference: MPS · Apple M4").
        # Hidden until the first classify() has picked a device; absent in AI
        # Config mode, where classification is an HTTP call.
        self.device_label = self._muted_label("", self)
        self.device_label.hide()
        self.statusBar().addPermanentWidget(self.device_label)
        self.camera_label = QLabel(self)
        self.serial_label = QLabel(self)
        # Added left to right, so serial ends up rightmost.
        self.statusBar().addPermanentWidget(self.camera_label)
        self.statusBar().addPermanentWidget(self.serial_label)
        # Rightmost pair: update affordance (hidden until there is something to
        # do; the Help menu is the always-reachable route), then sign-in.
        self.update_button = QPushButton(self)
        self.update_button.setObjectName("update")
        self.update_button.clicked.connect(lambda: self.open_update_dialog())
        self.update_button.hide()
        self.statusBar().addPermanentWidget(self.update_button)
        self.statusBar().addPermanentWidget(self.model_update_button)
        # Community identity, the only surface for it now (JL): the
        # Community page used to carry its own "Signed in as ... [Sign out]"
        # row, which duplicated this button. Hidden until signed in;
        # text/tooltip filled by _apply_auth_visibility.
        self.identity_label = self._muted_label("", self)
        self.identity_label.hide()
        self.statusBar().addPermanentWidget(self.identity_label)
        self.signin_button = QPushButton("Sign in", self)
        self.signin_button.clicked.connect(self._on_signin_clicked)
        self.statusBar().addPermanentWidget(self.signin_button)
        # The message area is painted by the bar itself, not a child widget,
        # so a click on it reaches only the bar (see eventFilter).
        self._status_bar = self.statusBar()
        self._status_bar.installEventFilter(self)
        self._shell_ready = True
        self._retitle_docks()
        self._paint_indicators()
        self._paint_model_update_button()
        for tab in self.tabs:
            self._paint_tab_marker(tab)
        self._apply_mode_visibility()
        self.set_status("Idle.")

    # ----- tab strip ------------------------------------------------------------

    def _build_tab_strip(self, parent: QWidget) -> QHBoxLayout:
        """ "All sorters" first, then one tab per sorter, then "+".

        Not movable: the dashboard is fixed first and the roster order is the
        order tabs were created in. Close buttons are ours rather than
        ``tabsClosable``'s, so the last sorter's can be hidden and brought
        back without Qt deleting it; ``#tabCloseButton`` gives them the same
        icon as a dock tab's.
        """
        row = QHBoxLayout()
        row.setContentsMargins(4, 4, 4, 0)
        row.setSpacing(4)
        self.tab_bar = QTabBar(parent)
        self.tab_bar.setObjectName("sorterTabs")
        self.tab_bar.setMovable(False)
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDrawBase(False)
        self.tab_bar.setIconSize(QSize(TAB_MARKER_SIZE, TAB_MARKER_SIZE))
        self.tab_bar.addTab(DASHBOARD_TITLE)
        for tab in self.tabs:
            self._append_tab_bar_entry(tab)
        self._update_close_buttons()
        self.tab_bar.setCurrentIndex(self.tabs.index(self.current_tab) + 1)
        # Connected last: seeding the current index is not the user switching.
        self.tab_bar.currentChanged.connect(self._on_tab_bar_changed)
        self.tab_bar.tabBarDoubleClicked.connect(self._on_tab_double_clicked)
        row.addWidget(self.tab_bar)
        self.new_sorter_button = QToolButton(parent)
        self.new_sorter_button.setObjectName("newSorterButton")
        self.new_sorter_button.setText("+")
        self.new_sorter_button.setToolTip(NEW_SORTER_TOOLTIP)
        self.new_sorter_button.clicked.connect(self.new_sorter)
        row.addWidget(self.new_sorter_button)
        row.addStretch(1)
        return row

    def _append_tab_bar_entry(self, tab: SorterTab) -> None:
        index = self.tab_bar.addTab(tab.name)
        button = QToolButton(self.tab_bar)
        button.setObjectName("tabCloseButton")
        button.setToolTip(CLOSE_SORTER_TOOLTIP)
        button.clicked.connect(lambda _checked=False, t=tab: self.close_tab(t))
        self.tab_bar.setTabButton(index, QTabBar.ButtonPosition.RightSide, button)
        self._close_buttons[tab.sorter_id] = button

    def _update_close_buttons(self) -> None:
        """The last sorter can't be closed, so its tab offers no close button."""
        closable = len(self.tabs) > 1
        for button in self._close_buttons.values():
            button.setVisible(closable)

    def _sorter_name(self, sorter_id: int) -> str:
        return next((t.name for t in self.tabs if t.sorter_id == sorter_id), f"Sorter {sorter_id}")

    def tab_index(self, tab: SorterTab) -> int:
        """The tab strip index of a sorter (0 is "All sorters")."""
        return self.tabs.index(tab) + 1

    def _on_tab_bar_changed(self, index: int) -> None:
        if index <= 0:
            self._show_dashboard()
        elif index - 1 < len(self.tabs):
            self.show_tab(self.tabs[index - 1])

    def _on_tab_double_clicked(self, index: int) -> None:
        if index >= 1:
            self.rename_tab(self.tabs[index - 1])

    def dashboard_showing(self) -> bool:
        return self.pages.currentWidget() is self.dashboard_page

    def _show_dashboard(self) -> None:
        if self.tab_bar.currentIndex() != 0:
            blocked = self.tab_bar.blockSignals(True)
            self.tab_bar.setCurrentIndex(0)
            self.tab_bar.blockSignals(blocked)
        self.pages.setCurrentWidget(self.dashboard_page)
        self.dashboard_page.refresh()
        # No sidebar entry is "where you are" on the dashboard.
        self._sidebar_group.setExclusive(False)
        for button in self.sidebar_buttons.values():
            button.setChecked(False)
        self._sidebar_group.setExclusive(True)

    def show_tab(self, tab: SorterTab) -> None:
        """Bring a sorter tab to the front, on the activity last shown."""
        index = self.tab_index(tab)
        if self.tab_bar.currentIndex() != index:
            blocked = self.tab_bar.blockSignals(True)
            self.tab_bar.setCurrentIndex(index)
            self.tab_bar.blockSignals(blocked)
        self._make_front(tab)
        button = self.sidebar_buttons.get(self._activity)
        if button is not None:
            button.setChecked(True)
        self.show_page(self._activity)

    def _make_front(self, tab: SorterTab) -> None:
        """Point every shared surface at ``tab``: stacks, docks, status bar."""
        if tab is self.current_tab:
            return
        self.current_tab = tab
        self._show_tab_pages(tab)
        set_front_sorter_id(self.db, tab.sorter_id)
        self.serial_monitor.retarget(tab)
        self.history_view.retarget(tab)
        self._retitle_docks()
        self._paint_indicators()
        self._paint_model_update_button()
        self._apply_mode_visibility()
        self.refresh_device_indicator()
        # Replayed, not re-recorded: the Messages panel already has this line.
        self.statusBar().showMessage(self._tab_status_text(tab, tab.last_status))

    def _add_tab_pages(self, tab: SorterTab) -> None:
        for name, page in tab.pages.items():
            self.tab_stacks[name].addWidget(page)

    def _show_tab_pages(self, tab: SorterTab) -> None:
        for name, stack in self.tab_stacks.items():
            stack.setCurrentWidget(tab.pages[name])

    def new_sorter(self) -> SorterTab:
        """ "+": a new, unconnected sorter, opened on its Sort page."""
        record = create_sorter(self.db)
        tab = SorterTab(self, Config(self.db, sorter_id=record.id).load(), record.name)
        self.tabs.append(tab)
        self._add_tab_pages(tab)
        self._append_tab_bar_entry(tab)
        self._update_close_buttons()
        self._retitle_docks()
        self.dashboard_page.rebuild()
        self._activity = "Sort"
        self.show_tab(tab)
        return tab

    def rename_tab(self, tab: SorterTab) -> None:
        """Double-click on a title. Names are unique; the data layer says why not."""
        name = self.ask_text(RENAME_TITLE, RENAME_LABEL, tab.name)
        if name is None or not name.strip() or name.strip() == tab.name:
            return
        try:
            record = rename_sorter(self.db, tab.sorter_id, name)
        except ValueError as exc:
            self.notify(RENAME_TITLE, str(exc))
            return
        tab.rename(record.name)
        self._paint_tab_marker(tab)
        self._retitle_docks()
        self.dashboard_page.rebuild()

    def close_tab(self, tab: SorterTab) -> bool:
        """Close one sorter: stop it, give its devices back, forget its settings.

        Never touches the model library or any image folder: those are
        shared, and a model active here may be active on another tab.
        """
        if tab not in self.tabs or len(self.tabs) <= 1:
            return False
        if (tab.broker is not None or tab.is_running) and not self.confirm_close_tab(tab):
            return False
        tab.shutdown()
        try:
            delete_sorter(self.db, tab.sorter_id)
        except ValueError as exc:
            self.notify(CLOSE_TITLE, str(exc))
            return False
        index = self.tabs.index(tab)
        was_front = tab is self.current_tab
        self.tabs.remove(tab)
        blocked = self.tab_bar.blockSignals(True)
        self.tab_bar.removeTab(index + 1)
        self.tab_bar.blockSignals(blocked)
        self._close_buttons.pop(tab.sorter_id, None)
        self._update_close_buttons()
        if was_front:
            successor = self.tabs[min(index, len(self.tabs) - 1)]
            if self.dashboard_showing():
                self._make_front(successor)
            else:
                self.show_tab(successor)
        for name, page in tab.pages.items():
            self.tab_stacks[name].removeWidget(page)
            page.deleteLater()
        tab.deleteLater()
        self._retitle_docks()
        self.dashboard_page.rebuild()
        return True

    def _confirm_close_tab(self, tab: SorterTab) -> bool:
        state = "running" if tab.is_running else "connected"
        answer = QMessageBox.question(
            self,
            CLOSE_TITLE,
            CLOSE_TEXT.format(name=tab.name, state=state),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _ask_text(self, title: str, label: str, current: str) -> str | None:
        text, ok = QInputDialog.getText(self, title, label, text=current)
        return text if ok else None

    def _paint_tab_marker(self, tab: SorterTab) -> None:
        index = self.tab_index(tab)
        self.tab_bar.setTabText(index, tab.name)
        if tab.is_running:
            self.tab_bar.setTabIcon(index, self._running_marker())
            self.tab_bar.setTabToolTip(index, RUNNING_TAB_TOOLTIP)
        else:
            self.tab_bar.setTabIcon(index, QIcon())
            self.tab_bar.setTabToolTip(index, "")

    def _running_marker(self) -> QIcon:
        """A dot in the action role, painted, since an icon is out of QSS's reach."""
        size = TAB_MARKER_SIZE
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self.palette_colors["action"]))
        painter.drawEllipse(1, 1, size - 2, size - 2)
        painter.end()
        return QIcon(pixmap)

    def _retitle_docks(self) -> None:
        """The two docks that follow the front tab say which one, once there are several."""
        suffix = f" — {self.current_tab.name}" if len(self.tabs) > 1 else ""
        for dock, title in ((self.serial_dock, "Serial Monitor"), (self.history_dock, "Classification History")):
            dock.setWindowTitle(title + suffix)
            # The View menu names the panel, not the sorter.
            dock.toggleViewAction().setText(title)

    # ----- what a tab asks of the window --------------------------------------------

    def on_tab_changed(self, tab: SorterTab) -> None:
        """A tab's run, device or result state moved (``SorterTab._changed``)."""
        if not self._shell_ready or tab not in self.tabs:
            return
        self._paint_tab_marker(tab)
        if tab is self.current_tab:
            self._paint_indicators()
            self._paint_model_update_button()
        self.bus.post("sorters/updated", tab.sorter_id)

    def on_tab_mode_changed(self, tab: SorterTab) -> None:
        """A tab's active model changed: the shared surfaces that show it follow."""
        if not self._shell_ready:
            return
        if tab is self.current_tab:
            self._apply_mode_visibility()
            self.refresh_device_indicator()
        # The library's Active column names every tab a model is active on.
        self.models_page.refresh()
        self.bus.post("sorters/updated", tab.sorter_id)

    def show_tab_status(self, tab: SorterTab, message: str, *, level: str = INFO, progress: bool | None = None) -> None:
        """A tab's status line reaches the status bar only while it is in front; the Messages panel keeps every one."""
        if not self._shell_ready:
            return
        text = self._tab_status_text(tab, message)
        if tab is self.current_tab:
            self.set_status(text, level=level, progress=progress)
        else:
            self._record_status(text, level=level, progress=progress)

    def _tab_status_text(self, tab: SorterTab, message: str) -> str:
        return f"{tab.name}: {message}" if len(self.tabs) > 1 and message else message

    def drain_all(self) -> int:
        """Deliver every queued event, the tabs' buses first. Returns how many.

        Tabs first because a tab's handlers post ``sorters/updated`` on the
        app bus; draining that afterwards puts the dashboard row on screen in
        the same tick as the change it shows. A worker result (app bus) that
        posts to a tab is picked up on the next tick, 50 ms later.
        """
        count = 0
        for tab in list(self.tabs):
            count += tab.bus.drain(max_items=128)
        return count + self.bus.drain(max_items=128)

    def _muted_label(self, text: str, parent: QWidget | None = None) -> QLabel:
        """A label in the muted role — registered so a theme switch recolors it."""
        label = QLabel(text, parent)
        label.setStyleSheet(f"color: {self.palette_colors['text_muted']};")
        self._muted_labels.append(label)
        return label

    def _placeholder_page(self, text: str = PLACEHOLDER_TEXT) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        label = self._muted_label(text, page)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)
        return page

    def _add_page(self, name: str, page: QWidget) -> None:
        self._pages_by_name[name] = page
        self.pages.addWidget(page)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget(self)
        sidebar.setObjectName("sidebar")
        column = QVBoxLayout(sidebar)
        column.setContentsMargins(6, 8, 6, 8)
        column.setSpacing(4)

        self.sidebar_buttons: dict[str, QToolButton] = {}
        # Which motif each button carries, so a theme switch can re-ink them.
        self._sidebar_icon_names: dict[str, str] = {}
        # The group owns exclusivity; keep the reference or it is collected.
        self._sidebar_group = QButtonGroup(self)
        self._sidebar_group.setExclusive(True)
        for icon_name, name in ACTIVITIES:
            column.addWidget(self._activity_button(sidebar, icon_name, name))
        self.sidebar_separator = self._sidebar_separator(sidebar)
        column.addWidget(self.sidebar_separator)
        for icon_name, name in MODE_ACTIVITIES:
            column.addWidget(self._activity_button(sidebar, icon_name, name))
        self.sidebar_settings_separator = self._sidebar_separator(sidebar)
        column.addWidget(self.sidebar_settings_separator)
        column.addWidget(self._activity_button(sidebar, *SETTINGS_ACTIVITY))
        column.addStretch(1)

        # Width follows the widest label's font metrics, not a constant — a
        # fixed pixel width clips "Community" on fonts wider than the dev box.
        metrics = sidebar.fontMetrics()
        widest = max(metrics.horizontalAdvance(name) for name in self.sidebar_buttons)
        sidebar.setFixedWidth(max(SIDEBAR_WIDTH, widest + 24))

        self.sidebar_buttons["Sort"].setChecked(True)
        self._paint_sidebar_icons()
        self.show_page("Sort")
        return sidebar

    def _sidebar_separator(self, parent: QWidget) -> QFrame:
        """The hairline splitting the always-live surfaces from the mode pair.

        A QFrame, not a styled QWidget: only the former paints a stylesheet
        background without a paintEvent of its own, which is what keeps the
        colour palette-driven and re-themed by the stylesheet alone.
        """
        line = QFrame(parent)
        line.setObjectName("sidebarSeparator")
        line.setFixedHeight(1)
        return line

    def _activity_button(self, parent: QWidget, icon_name: str, name: str) -> QToolButton:
        button = QToolButton(parent)
        button.setText(name)
        button.setIconSize(QSize(SIDEBAR_ICON_SIZE, SIDEBAR_ICON_SIZE))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        button.setCheckable(True)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.clicked.connect(lambda _checked=False, page=name: self.open_activity(page))
        # The checked state is a different ink, and the group flips two buttons
        # per click — so both ends of the swap repaint themselves.
        button.toggled.connect(lambda _on=False, page=name: self._paint_sidebar_icon(page))
        self._sidebar_group.addButton(button)
        self.sidebar_buttons[name] = button
        self._sidebar_icon_names[name] = icon_name
        return button

    def _paint_sidebar_icon(self, name: str) -> None:
        """Ink one sidebar icon for its current state, from the live palette.

        The three colors are the ones theme.py's ``#sidebar QToolButton``
        rules put on the label, so icon and text always agree.
        """
        button = self.sidebar_buttons[name]
        if button.property("unavailable"):
            color = unavailable_ink(self.palette_colors)
        else:
            color = self.palette_colors["text_highlight" if button.isChecked() else "text_muted"]
        button.setIcon(vector_icon(self._sidebar_icon_names[name], color, SIDEBAR_ICON_SIZE))

    def _paint_sidebar_icons(self) -> None:
        for name in self.sidebar_buttons:
            self._paint_sidebar_icon(name)

    def go_to_activity(self, name: str) -> None:
        """Navigate as if the sidebar button had been clicked, checked state included.

        What an in-page "take me there" button wants: ``open_activity`` alone
        switches the page but leaves the sidebar pointing at where the user
        was.
        """
        button = self.sidebar_buttons.get(name)
        if button is not None:
            button.setChecked(True)
        self.open_activity(name)

    def _open_settings_section(self, name: str) -> None:
        self.go_to_activity("Settings")
        items = self.settings_list.findItems(name, Qt.MatchFlag.MatchExactly)
        if items:
            self.settings_list.setCurrentItem(items[0])

    def _build_settings_page(self) -> QWidget:
        """Camera, Serial and Image Processing belong to the front tab; the rest are shared."""
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(12, 12, 12, 12)
        row.setSpacing(12)
        self.settings_list = QListWidget(page)
        self.settings_list.setFixedWidth(160)
        self.settings_pages = QStackedWidget(page)
        builders: dict[str, Callable[[], QWidget]] = {
            "Theme": self._build_theme_section,
            WINFORMS_IMPORT_SECTION: lambda: build_winforms_import_section(self),
        }
        for name in SETTINGS_SECTIONS:
            self.settings_list.addItem(name)
            if name in self.tab_stacks:
                self.settings_pages.addWidget(self.tab_stacks[name])
                continue
            build = builders.get(name)
            self.settings_pages.addWidget(build() if build else self._placeholder_page())
        self.settings_list.currentRowChanged.connect(self.settings_pages.setCurrentIndex)
        self.settings_list.setCurrentRow(0)
        row.addWidget(self.settings_list)
        row.addWidget(self.settings_pages, 1)
        return page

    def _build_models_page(self) -> QWidget:
        # Kept on self: mode/changed and navigating to the page both refresh it.
        self.models_page = build_models_page(self)
        self.models_page.set_images_hook(self._open_model_images)
        self.models_page.set_headstamps_hook(self._open_headstamps)
        self.models_page.set_evaluate_hook(self._open_evaluator)
        return self.models_page

    def _current_page_name(self) -> str:
        return next(
            (name for name, w in self._pages_by_name.items() if w is self.pages.currentWidget()),
            "Sort",
        )

    def open_help(self) -> None:
        """F1 / Help menu: the guide dock, opened at the current context's topic."""
        page = self._current_page_name()
        section = None
        if page == "Settings":
            item = self.settings_list.currentItem()
            section = item.text() if item is not None else None
        self.help_view.show_topic(topic_for(page, section))
        self.reveal_dock(self.help_dock)

    def reveal_dock(self, dock: Any) -> None:
        """Open a panel and bring it to the front — the QtAds show/raise pair.

        ``toggleView(True)`` re-opens a closed panel (QWidget ``show()`` does
        not: QtAds tracks closed-ness itself, and a closed dock has been taken
        out of its area). ``setAsCurrentTab`` is the raise: a panel sharing a
        tab group with another is otherwise "open" but behind it.
        """
        dock.toggleView(True)
        dock.setAsCurrentTab()
        container = dock.floatingDockContainer()
        if container is not None:
            container.raise_()

    def open_serial_monitor(self) -> None:
        """Settings → Serial's "Open serial monitor" button; View toggles it directly."""
        self.reveal_dock(self.serial_dock)

    def _open_model_images(self, model: Any) -> None:
        from .dialog_model_images import ModelImagesDialog

        ModelImagesDialog(self, self.current_tab.config, model.id).exec()

    def slot_targets(self) -> list[Any]:
        """Every sorter tab as a headstamp-editor "Slots for" target, front tab marked."""
        from .dialog_headstamps import SlotTarget

        return [SlotTarget(t.sorter_id, t.name, t.config, t.bus, front=t is self.current_tab) for t in self.tabs]

    def _open_headstamps(self, model: Any) -> None:
        from .dialog_headstamps import HeadstampManagerDialog

        tab = self.current_tab
        HeadstampManagerDialog(self, tab.config, model.id, bus=tab.bus, slot_targets=self.slot_targets).exec()

    def _open_evaluator(self, model: Any) -> None:
        from .dialog_model_evaluator import ModelEvaluatorDialog

        ModelEvaluatorDialog(self, self, model).exec()

    def _build_theme_section(self) -> QWidget:
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(QLabel("Theme", page))
        self.theme_combo = QComboBox(page)
        self.theme_combo.addItems(theme_names())
        self.theme_combo.setCurrentText(self.theme_name)
        # Connected last: setCurrentText must not count as the user choosing.
        self.theme_combo.currentTextChanged.connect(self.set_theme)
        self.theme_combo.setMaximumWidth(240)
        column.addWidget(self.theme_combo)
        self.theme_edit_button = QPushButton("Edit theme…", page)
        self.theme_edit_button.setMaximumWidth(240)
        self.theme_edit_button.clicked.connect(self._open_theme_editor)
        column.addWidget(self.theme_edit_button)
        column.addStretch(1)
        return page

    def _open_theme_editor(self) -> None:
        from .dialog_theme_editor import ThemeEditorDialog

        ThemeEditorDialog(self, self).exec()

    def _build_dock(self, title: str, widget: QWidget, area: Any, *, scroll_area: bool = True) -> Any:
        """One QtAds panel: closable, movable, floatable, hinted on its tab.

        ``scroll_area=False`` for content that lays itself out from the space
        it is given. QtAds wraps a panel's widget in a ``QScrollArea`` by
        default, which hands that widget its *preferred* size and scrolls the
        difference — so the widget is never told the panel got smaller, and
        answers a question nobody asked (issue #101).
        """
        dock = ads.CDockWidget(self.dock_manager, title)
        insert_mode = (
            ads.CDockWidget.eInsertMode.AutoScrollArea if scroll_area else ads.CDockWidget.eInsertMode.ForceNoScrollArea
        )
        dock.setWidget(widget, insert_mode)
        # The tab *is* the drag handle in QtAds, so that is where the hint goes
        # (a QDockWidget put it on the whole panel, which was never the target).
        dock.setTabToolTip(DOCK_DRAG_HINT)
        self.dock_manager.addDockWidget(area, dock)
        return dock

    def _build_serial_dock(self) -> None:
        # The monitor subscribes serial/* itself and keeps the full session
        # history — a dock that exists from startup needs no backlog replay.
        self.serial_monitor = build_serial_monitor(self.current_tab)
        # Bottom, like Arduino IDE's monitor / VS Code's terminal (JL).
        self.serial_dock = self._build_dock("Serial Monitor", self.serial_monitor, ads.BottomDockWidgetArea)

    def _build_history_dock(self) -> None:
        self.history_view = build_history_view(self.current_tab)
        # No scroll area: the tile grid sizes itself to the panel and drops
        # what doesn't fit, which only works if it is told the panel's real
        # size (issue #101).
        self.history_dock = self._build_dock(
            "Classification History",
            self.history_view,
            ads.RightDockWidgetArea,
            scroll_area=False,
        )
        # Supplementary, so it starts out of the way; View re-opens it.
        self.history_dock.toggleView(False)

    def _redock_panels(self) -> None:
        """Return every open panel to its home area, un-floated.

        View → "Re-dock panels", the always-works escape hatch. Re-issuing
        each ``addDockWidget`` is the whole implementation: in QtAds that
        pulls the panel out of wherever it ended up (a floating container, a
        tab group, another edge) and rebuilds the startup layout. Closed
        panels are skipped — ``addDockWidget`` re-opens a closed dock, and a
        panel the user switched off in View must stay off.
        """
        for attr, area in DOCK_HOMES:
            dock = getattr(self, attr)
            if dock.isClosed():
                continue
            self.dock_manager.addDockWidget(area, dock)

    def _build_community_page(self) -> QWidget:
        self.community_page = build_community_page(self)
        self.community_page.on_auth_changed = self._on_auth_changed
        return self.community_page

    def _apply_auth_visibility(self) -> None:
        signed_in = self.community_page.is_signed_in()
        self._set_activity_visible("Community", signed_in)
        self.signin_button.setText("Sign out" if signed_in else "Sign in")
        self._update_identity_label(signed_in)

    def _update_identity_label(self, signed_in: bool) -> None:
        """Display name (or email) next to the Sign out button, display-only.

        Read straight off the auth object's decoded claims — same source the
        removed Community-page banner used, not the community server's
        profile metadata, so this never blocks on a network call.
        ``existing_auth_manager()`` never constructs one (CLAUDE.md: building
        must not touch MSAL) — if ``signed_in`` is True one already exists.
        """
        if not signed_in:
            self.identity_label.hide()
            self.identity_label.setText("")
            self.identity_label.setToolTip("")
            return
        auth = self.community_page.existing_auth_manager()
        try:
            name, email = auth.identity() if auth is not None else (None, None)
        except Exception:
            name, email = None, None
        name = (name or "").strip()
        email = (email or "").strip()
        self.identity_label.setText(name or email or "(unknown)")
        self.identity_label.setToolTip(email)
        self.identity_label.show()

    def _on_auth_changed(self) -> None:
        self._apply_auth_visibility()

    def _on_signin_clicked(self) -> None:
        if self.community_page.is_signed_in():
            self.community_page.sign_out()
        else:
            self.community_page.open_login()

    # ----- import from the Windows app ----------------------------------------

    def _offer_winforms_import(self) -> None:
        """First-run offer. Silent, and cheap, when there is nothing to offer."""
        try:
            maybe_offer_first_run(self)
        except Exception:
            # An import that cannot even be offered must not take the launch
            # with it — Settings keeps the same dialog reachable.
            log.exception("first-run Windows-app import offer failed")

    def after_winforms_import(self, result: Any) -> None:
        """Re-read everything the import may have rewritten.

        It can touch the model library, the active model, every headstamp and
        slot, and three settings sections at once, so every tab re-runs the
        same refresh a mode switch does rather than trying to be surgical.
        """
        for tab in self.tabs:
            tab.reload()
        self.models_page.refresh()
        self.set_status("Imported from the Windows app.")

    # ----- updates ------------------------------------------------------------

    def open_update_dialog(self, *, check: bool = False) -> None:
        from .dialog_update import UpdateDialog

        self._update_dialog = UpdateDialog(
            self, info=self._update_info, app=self, pending=self._pending_update, check_on_open=check
        )
        self._update_dialog.open()

    def note_pending_update(self, pending: Any) -> None:
        self._pending_update = pending
        self.update_button.setText("Restart to update")
        self.update_button.show()

    def note_update_info(self, info: Any) -> None:
        self._update_info = info
        if self._pending_update is not None:
            return  # a staged update outranks a fresh finding
        if info is None:
            self.update_button.hide()
        else:
            self.update_button.setText(f"Update to {info.version}")
            self.update_button.show()

    def _startup_update_check(self) -> None:
        """Silent: staged update first, then an opt-out-able check."""
        from ..update import updater

        pending = updater.pending_update()
        if pending is not None:
            self.note_pending_update(pending)
            return
        if updater.checks_disabled() or self._load_setting(updater.SETTING_CHECK_ON_STARTUP) is False:
            return
        self.run_worker(
            updater.check_for_update,
            on_done=self.note_update_info,
            on_error=lambda _exc: None,  # silent by design; Help menu re-checks loudly
        )

    def _build_help_dock(self) -> None:
        # A dock, not a free window (JL): pin the guide beside the work while
        # learning, toggle it away after.
        self.help_view = build_help_window(self)
        self.help_dock = self._build_dock("User Guide", self.help_view, ads.RightDockWidgetArea)
        self.help_dock.toggleView(False)

    def _build_themes_dock(self) -> None:
        """Every theme in one list, applied on the click (Seth via JL).

        Settings → Theme is where a theme is *configured*; this is where one is
        *tried*, which is a different activity — you want the whole list in
        front of you and the app repainting under it. Both drive ``set_theme``
        and both are re-read by ``refresh_theme_picker``, so neither can drift
        from the registry or from each other.
        """
        panel = QWidget(self)
        column = QVBoxLayout(panel)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(6)
        self.theme_list = QListWidget(panel)
        self.theme_list.setObjectName("themeList")
        self.theme_list.addItems(theme_names())
        self.theme_list.setCurrentRow(self._theme_row(self.theme_name))
        # Connected last, like the combo: seeding the selection is not a choice.
        self.theme_list.currentTextChanged.connect(self.set_theme)
        column.addWidget(self.theme_list, 1)
        self.theme_dock_edit_button = QPushButton("Edit theme…", panel)
        self.theme_dock_edit_button.clicked.connect(self._open_theme_editor)
        column.addWidget(self.theme_dock_edit_button)

        self.themes_dock = self._build_dock("Themes", panel, ads.RightDockWidgetArea)
        self.themes_dock.toggleView(False)
        # A theme saved from an editor opened before this panel was lands in
        # the registry without passing through here.
        self.themes_dock.viewToggled.connect(self._on_themes_dock_toggled)

    def _build_messages_dock(self) -> None:
        self.messages_view = build_messages_view(self, self.status_log)
        # No scroll area: the log wraps to the panel's width and scrolls itself.
        self.messages_dock = self._build_dock(
            "Messages", self.messages_view, ads.RightDockWidgetArea, scroll_area=False
        )
        self.messages_dock.toggleView(False)

    def open_messages(self) -> None:
        self.reveal_dock(self.messages_dock)

    def _on_themes_dock_toggled(self, opened: bool) -> None:
        if opened:
            self.refresh_theme_picker()

    @staticmethod
    def _theme_row(name: str) -> int:
        names = theme_names()
        return names.index(name) if name in names else 0

    def refresh_theme_picker(self) -> None:
        """Re-read the theme registry into both pickers (the theme editor's hook)."""
        if hasattr(self, "theme_combo"):
            blocked = self.theme_combo.blockSignals(True)
            self.theme_combo.clear()
            self.theme_combo.addItems(theme_names())
            self.theme_combo.blockSignals(blocked)
        if hasattr(self, "theme_list"):
            blocked = self.theme_list.blockSignals(True)
            self.theme_list.clear()
            self.theme_list.addItems(theme_names())
            self.theme_list.blockSignals(blocked)
        self._sync_theme_pickers()

    def _sync_theme_pickers(self) -> None:
        """Point both pickers at the live theme without re-applying it."""
        if hasattr(self, "theme_combo"):
            blocked = self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentText(self.theme_name)
            self.theme_combo.blockSignals(blocked)
        if hasattr(self, "theme_list"):
            blocked = self.theme_list.blockSignals(True)
            self.theme_list.setCurrentRow(self._theme_row(self.theme_name))
            self.theme_list.blockSignals(blocked)

    def _build_menus(self) -> None:
        # menuBar().addMenu(str) hands the QMenu back with Python ownership; the
        # menus have to be kept alive here or shiboken deletes them.
        self.menus: dict[str, Any] = {}
        file_menu = self.menus["File"] = self.menuBar().addMenu("&File")
        open_data = file_menu.addAction("Open data folder")
        open_data.triggered.connect(self._open_data_folder)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)

        toggle = self.serial_dock.toggleViewAction()
        toggle.setText("Serial Monitor")
        toggle.setShortcut(QKeySequence("Ctrl+Shift+M"))  # not the Windows app's Ctrl+K: delete-to-end-of-line on Linux
        self.menus["View"] = self.menuBar().addMenu("&View")
        self.menus["View"].addAction(toggle)
        history_toggle = self.history_dock.toggleViewAction()
        history_toggle.setText("Classification History")
        self.menus["View"].addAction(history_toggle)
        help_toggle = self.help_dock.toggleViewAction()
        help_toggle.setText("User Guide panel")
        self.menus["View"].addAction(help_toggle)
        themes_toggle = self.themes_dock.toggleViewAction()
        themes_toggle.setText("Themes")
        self.menus["View"].addAction(themes_toggle)
        messages_toggle = self.messages_dock.toggleViewAction()
        messages_toggle.setText("Messages")
        self.menus["View"].addAction(messages_toggle)
        self.menus["View"].addSeparator()
        # The always-works escape hatch (Seth: floated the history panel and
        # couldn't get it back): drag-to-dock takes dexterity and has failed
        # users on more than one platform; one menu action can't.
        redock = self.menus["View"].addAction("Re-dock panels")
        redock.triggered.connect(self._redock_panels)

        self.menus["Help"] = self.menuBar().addMenu("&Help")
        guide = self.menus["Help"].addAction("User Guide")
        guide.setShortcut(QKeySequence.StandardKey.HelpContents)  # F1
        guide.triggered.connect(self.open_help)
        check = self.menus["Help"].addAction("Check for updates…")
        check.triggered.connect(lambda: self.open_update_dialog(check=True))
        support = self.menus["Help"].addAction("Export support package…")
        support.triggered.connect(self._open_support_dialog)
        self.menus["Help"].addSeparator()
        about = self.menus["Help"].addAction("About")
        about.triggered.connect(self._show_about)
        license_action = self.menus["Help"].addAction("License")
        license_action.triggered.connect(self._show_license)

    # ----- navigation ---------------------------------------------------------

    def open_activity(self, name: str) -> None:
        """What a sidebar click does: every activity is a page of its own.

        From the dashboard, a sidebar click goes back to the front sorter tab.
        """
        if self.dashboard_showing():
            self._activity = name
            self.show_tab(self.current_tab)
            return
        self.show_page(name)

    def show_page(self, name: str) -> None:
        if name == DASHBOARD_TITLE:
            self._show_dashboard()
            return
        self._activity = name
        self.pages.setCurrentWidget(self._pages_by_name[name])
        tab = self.current_tab
        if name == "Sort":
            tab.enter_sort()
        elif name == "Models":
            self.models_page.refresh(announce=True)
        elif name == "Community":
            self.community_page.refresh_auth_state()
        elif name == "Train":
            # Headstamps and images change from the Models page and from
            # imports; the counts are read off disk every time, never cached.
            tab.train_page.refresh()
        elif name == AI_CONFIG_ACTIVITY:
            # A click on the *muted* entry has to land on the explainer naming
            # the active model, not on whatever the last mode change left.
            tab.ai_page.refresh_mode()

    def _open_data_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(app_data_dir())))

    def _show_about(self) -> None:
        from .dialog_about import build_about_dialog

        build_about_dialog(self).exec()

    def _open_support_dialog(self) -> None:
        """Help → Export support package…: the report to paste on Discord."""
        from .dialog_support import open_support_dialog

        open_support_dialog(self)

    def _show_license(self) -> None:
        from .dialog_about import build_license_dialog

        build_license_dialog(self).exec()

    def _apply_mode_visibility(self) -> None:
        """The mode inks the Train / AI Config pair. **Neither is ever hidden**
        (JL: a hidden activity is one nobody finds).

        At most one of the two is live — a trainable local model makes it
        Train, no active model or an active openai-mode model makes it AI
        Config (both classify over HTTP, and the page edits whichever config
        is in effect) — and a community model makes it neither. The other
        goes muted: still clickable, with the explainer behind it
        (train_page's and ai_page's unavailable panels) saying why and what
        to do.
        """
        from ..data.models import is_openai_model, is_trainable

        model = self.current_tab.active_model()
        train_live = is_trainable(model)
        ai_live = model is None or is_openai_model(model)
        self._set_activity_unavailable("Train", not train_live)
        self._set_activity_unavailable(AI_CONFIG_ACTIVITY, not ai_live)
        self.sidebar_buttons["Train"].setToolTip(ACTIVITY_TOOLTIP_LIVE if train_live else TRAIN_TOOLTIP_MUTED)
        self.sidebar_buttons[AI_CONFIG_ACTIVITY].setToolTip(
            ACTIVITY_TOOLTIP_LIVE if ai_live else AI_CONFIG_TOOLTIP_MUTED
        )

    def _set_activity_unavailable(self, name: str, unavailable: bool) -> None:
        """Ink a sidebar button as "leads somewhere, but not usable right now".

        A dynamic property rather than ``setEnabled(False)``: the click has to
        keep working, since the explainer it opens is the whole point. Qt only
        re-reads a property-keyed rule on a re-polish, so ask for one.
        """
        button = self.sidebar_buttons[name]
        button.setProperty("unavailable", bool(unavailable))
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        # The icon is a QIcon, which no stylesheet reaches — same split as
        # checked/unchecked (see _paint_sidebar_icon).
        self._paint_sidebar_icon(name)

    def _set_activity_visible(self, name: str, visible: bool) -> None:
        """Community only — it is the one activity that comes and goes (auth)."""
        button = self.sidebar_buttons[name]
        button.setVisible(visible)
        page = self._pages_by_name.get(name)
        if not visible and page is not None and self.pages.currentWidget() is page:
            self.sidebar_buttons["Sort"].setChecked(True)
            self.show_page("Sort")

    # ----- theme --------------------------------------------------------------

    def _load_setting(self, key: str) -> Any:
        """Read a settings row, or None if there's no DB / it can't be read."""
        if self.db is None:
            return None
        try:
            from ..data.repository import SettingsRepo

            return SettingsRepo(self.db).get(key)
        except Exception:
            return None

    def _save_setting(self, key: str, value: Any) -> None:
        if self.db is None:
            return
        try:
            from ..data.repository import SettingsRepo

            SettingsRepo(self.db).set(key, value)
        except Exception:
            # A preference that can't be persisted still applies this session.
            pass

    @staticmethod
    def _bytes_to_setting(data: bytes) -> str:
        """Base64 text: SettingsRepo stores JSON-encoded values, not raw bytes."""
        return base64.b64encode(bytes(data)).decode("ascii")

    @staticmethod
    def _setting_to_bytes(value: Any) -> bytes | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return base64.b64decode(value.encode("ascii"))
        except (ValueError, TypeError):
            return None

    def _restore_window_state(self) -> None:
        """Dock layout + the model table's column widths, from the last session.

        The dock half is the manager's own XML state, not ``QMainWindow``'s —
        with QtAds the window has no QDockWidgets left to save. A blob written
        by the pre-QtAds build is simply not this format; ``restoreState``
        answers False and the panels keep their built-in homes, so the switch
        needs no migration.
        """
        state = self._setting_to_bytes(self._load_setting(SETTING_WINDOW_STATE))
        if state is not None:
            self.dock_manager.restoreState(QByteArray(state))
        columns = self._setting_to_bytes(self._load_setting(SETTING_MODELS_COLUMNS))
        if columns is not None:
            self.models_page.restore_header_state(columns)

    def _save_window_state(self) -> None:
        dock_state = bytes(self.dock_manager.saveState().data())
        self._save_setting(SETTING_WINDOW_STATE, self._bytes_to_setting(dock_state))
        self._save_setting(SETTING_MODELS_COLUMNS, self._bytes_to_setting(self.models_page.header_state()))

    def set_theme(self, name: str) -> None:
        """Switch palettes live and remember the choice."""
        resolved = resolve_theme(name)
        self._apply_theme(resolved)
        self._save_setting(SETTING_THEME, resolved)
        # Whichever picker was used, the other one follows — signals blocked,
        # so the follower never re-applies what just happened.
        self._sync_theme_pickers()

    def _apply_theme(self, name: str) -> None:
        self.theme_name = name
        self.palette_colors = THEMES[name]
        self.setStyleSheet(build_stylesheet(self.palette_colors))
        muted = f"color: {self.palette_colors['text_muted']};"
        for label in self._muted_labels:
            label.setStyleSheet(muted)
        # Colors baked into rich text / per-line paints need a hand re-render.
        for tab in self.tabs:
            tab.apply_palette()
        if hasattr(self, "sidebar_buttons"):
            self._paint_sidebar_icons()
        if hasattr(self, "serial_monitor"):
            self.serial_monitor.apply_palette()
        if hasattr(self, "history_view"):
            self.history_view.apply_palette()
        if hasattr(self, "models_page"):
            self.models_page.apply_palette()
        if hasattr(self, "messages_view"):
            self.messages_view.apply_palette()
        if hasattr(self, "dashboard_page"):
            self.dashboard_page.apply_palette()
        # Indicator dots and tab markers carry state, not a palette role a
        # stylesheet can reach.
        if self._shell_ready:
            self._paint_indicators()
            for tab in self.tabs:
                self._paint_tab_marker(tab)

    # ----- status -------------------------------------------------------------

    def set_status(self, message: str, *, level: str = INFO, progress: bool | None = None) -> None:
        """The one way to write the status bar — main thread only, like any widget.

        Every line is also kept, untruncated, in ``status_log`` (the Messages
        panel). ``level=ERROR`` marks a failure; ``progress`` overrides the
        trailing-"…" inference of an in-progress line (see message_log.py).
        """
        text = str(message)
        self.statusBar().showMessage(text)
        self._record_status(text, level=level, progress=progress)

    def _record_status(self, text: str, *, level: str = INFO, progress: bool | None = None) -> None:
        """Keep a status line in the Messages panel without showing it on the bar."""
        entry = self.status_log.add(text, level=level, progress=progress)
        # A file-only trail of what the operator was shown; the panel isn't saved.
        if not entry.progress:
            log.debug("status [%s] %s", entry.level, entry.text)

    def _in_status_message_area(self, pos: Any) -> bool:
        """Is ``pos`` (status-bar coordinates) left of every permanent widget?"""
        bar = self.statusBar()
        children = bar.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
        left_edge = min((w.geometry().x() for w in children if w.isVisible()), default=bar.width())
        return pos.x() < left_edge

    def eventFilter(self, watched: Any, event: Any) -> bool:
        if watched is self._status_bar:
            kind = event.type()
            if (
                kind == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
                and self._in_status_message_area(event.position().toPoint())
            ):
                self.open_messages()
                return True
            if kind == QEvent.Type.ToolTip and self._in_status_message_area(event.pos()):
                QToolTip.showText(event.globalPos(), MESSAGES_HINT, self._status_bar)
                return True
        return super().eventFilter(watched, event)

    def _indicator_html(self, message: str, *, connected: bool) -> str:
        color = self.palette_colors["success" if connected else "error"]
        return f'<span style="color: {color};">●</span> {html.escape(str(message))}'

    def _paint_indicators(self) -> None:
        """Camera and serial dots, for the front tab."""
        tab = self.current_tab
        for label, (message, connected) in (
            (self.camera_label, tab.camera_state),
            (self.serial_label, tab.serial_state),
        ):
            label.setText(self._indicator_html(message, connected=connected))

    def _paint_model_update_button(self) -> None:
        version = self.current_tab.model_update_version
        if version is None:
            self.model_update_button.hide()
            return
        self.model_update_button.setText(MODEL_UPDATE_BUTTON.format(version=version))
        self.model_update_button.show()

    def refresh_device_indicator(self) -> None:
        """Show where local classification runs, once that is known.

        Reads only the device `local_inference` has already picked — never
        imports torch, never triggers the probe or its benchmarks — so it is
        free and safe on the UI thread. Hidden until the first classify(),
        and in AI Config mode, where classification is an HTTP call and no
        local device is involved.
        """
        text: str | None = None
        if classifier.uses_local_inference(self.db, model_id=self.current_tab.config.active_model_id):
            text = local_inference.device_description()
        if text:
            self.device_label.setText(f"Inference: {text}")
            self.device_label.show()
        else:
            self.device_label.hide()

    def _warm_device_indicator(self) -> None:
        """Pick the inference device ahead of the first classify, off-thread.

        The status bar should say where classification will run without the
        user having to feed a case first. Guarded so an AI Config user never
        pays a torch import (the gate rule, §5): only when a local model is
        active and torch is already installed. `is_available()` imports torch
        and runs the device probe on the worker — the same work the first
        classify would have paid — and the on_done just repaints the label.
        Startup-only (behind ``auto_connect``): a model activated later gets
        its indicator from the first classification instead.
        """
        if not classifier.uses_local_inference(self.db, model_id=self.current_tab.config.active_model_id):
            return
        if not local_inference.is_installed() or local_inference.device_description():
            return
        self.run_worker(
            local_inference.is_available,
            on_done=lambda _ok: self.refresh_device_indicator(),
            on_error=lambda _exc: None,
        )

    # ----- startup serial -----------------------------------------------------

    def _auto_connect_serial(self) -> None:
        """One tab: the full walk, as before tabs existed. Several: saved ports only.

        With several tabs each opens only its own saved port, and all of them
        open on one worker in tab order, so no two tabs race for a board and
        the device claims land in a predictable order on the main thread.
        """
        if len(self.tabs) == 1:
            self.tabs[0].auto_connect_serial()
            return
        jobs: list[tuple[SorterTab, str]] = []
        for tab in self.tabs:
            port = (tab.config.serial.get("port") or "").strip()
            if not port:
                tab.set_status("No saved port. Connect in Settings → Serial.")
            elif port == EMULATED_PORT:
                tab.connect_serial(port)
            else:
                jobs.append((tab, port))
        if not jobs:
            return

        def work() -> list[tuple[SorterTab, str, Any]]:
            opened: set[str] = set()
            results: list[tuple[SorterTab, str, Any]] = []
            for tab, port in jobs:
                # Two tabs saved the same port: the first one to open it has it.
                broker = None if port in opened else tab.open_saved_port_blocking(port)
                if broker is not None:
                    opened.add(port)
                results.append((tab, port, broker))
            return results

        def done(results: list[tuple[SorterTab, str, Any]]) -> None:
            for tab, port, broker in results:
                if tab in self.tabs:
                    tab.finish_saved_port(port, broker)
                elif broker is not None:
                    broker.stop()

        self.set_status(f"Auto-connecting {len(jobs)} sorter(s) to their saved ports…")
        self.run_worker(
            work, on_done=done, on_error=lambda exc: self.set_status(f"Auto-connect error: {exc}", level=ERROR)
        )

    def notify(self, title: str, text: str) -> None:
        """User-facing warning. Tests patch this — never let a modal open there."""
        QMessageBox.warning(self, title, text)

    # ----- worker dispatch ----------------------------------------------------

    def run_worker(
        self,
        fn: Callable[[], Any],
        *,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Run `fn` in a daemon thread and post the result back via the bus.

        Workers must never touch widgets; the drain timer delivers the result
        on the main thread.
        """
        # A monotonic token, never id(fn): CPython reuses freed addresses, so
        # two sequential workers can share an id — and with the subscriptions
        # left in place, the second worker's result would also be delivered to
        # the first worker's stale callback (a cartridge list arriving in a
        # username handler, in practice).
        token = next(self._worker_tokens)
        topic_done = f"worker/done/{token}"
        topic_err = f"worker/err/{token}"

        def _deliver_done(payload: Any) -> None:
            _unsubscribe()
            if on_done is not None:
                on_done(payload)

        def _deliver_err(exc: Any) -> None:
            _unsubscribe()
            if on_error is not None:
                on_error(exc)

        def _unsubscribe() -> None:
            self.bus.unsubscribe(topic_done, _deliver_done)
            self.bus.unsubscribe(topic_err, _deliver_err)

        self.bus.subscribe(topic_done, _deliver_done)
        self.bus.subscribe(topic_err, _deliver_err)

        def _run() -> None:
            try:
                self.bus.post(topic_done, fn())
            except Exception as exc:
                traceback.print_exc()
                self.bus.post(topic_err, exc)

        threading.Thread(target=_run, daemon=True).start()

    # ----- lifecycle ----------------------------------------------------------

    def closeEvent(self, event: Any) -> None:
        # No confirm-on-close while a run is active: stop every controller and
        # go, the way this app has always closed.
        # The timers first: a closed-but-not-destroyed window (every test
        # window, and the real one between close and quit) must go inert.
        # Left running, each keeps draining its buses and repainting its
        # preview forever, and hundreds of those zombie ticks in one event
        # pump is what took an access violation on the Windows CI runner.
        for timer in (self._bus_timer, self._preview_timer):
            try:
                timer.stop()
            except Exception:
                pass
        for tab in self.tabs:
            tab.shutdown()
        try:
            self._save_window_state()
        except Exception:
            # A preference that can't be persisted must never block shutdown.
            pass
        try:
            # A floated panel is its own top-level window parented to the dock
            # manager, so closing the main window leaves it on screen. QtAds's
            # own teardown call hides the manager and every floating container
            # together; without it the app "won't quit" with a panel torn off.
            self.dock_manager.hideManagerAndFloatingWidgets()
        except Exception:
            pass
        super().closeEvent(event)


def default_qpa_platform() -> None:
    """Linux: prefer XCB (XWayland) over native Wayland.

    A floated panel is frozen under native Wayland — JL hit this live
    (release a floating dock and it can no longer be moved or resized): a
    frameless top-level window can't ask a Wayland compositor to move/resize
    it, an upstream Qt/Wayland limitation, not something fixable in
    application code. It outlived the move to QtAds, whose floating
    containers are the same kind of top-level window, and XCB via XWayland
    still doesn't have the gap.

    ``setdefault`` so an explicit ``QT_QPA_PLATFORM`` (env, or a test's
    ``offscreen``) always wins. The semicolon list, never a bare ``"xcb"``:
    Qt tries entries left to right, so this still falls back to native
    Wayland on a box missing an XWayland dependency (JL hit
    ``libxcb-cursor0`` missing) instead of failing to start.
    """
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb;wayland")


def run_app(config: Any) -> int:
    default_qpa_platform()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QtMainWindow(config)
    window.resize(1024, 768)
    window.show()
    return app.exec()
