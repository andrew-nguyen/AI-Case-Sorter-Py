"""The "All sorters" dashboard (``ui/dashboard_page.py``), AC6.

One row per sorter tab. A row repaints as soon as its tab's state moves, with
no timer. Start/Stop on a row runs the tab's own pre-flight checks, and a
click on the name opens that tab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("PySide6")

from sorter.hardware.serial_emulator import EMULATED_PORT
from sorter.ui.dashboard_page import (
    COL_BOARD,
    COL_CAMERA,
    COL_CASES,
    COL_CROP,
    COL_MODEL,
    COL_NAME,
    COL_RESULT,
    COL_RUN,
    DASHBOARD_TITLE,
    RUN_STATE_IDLE,
)
from sorter.ui.sorter_tab import AI_CONFIG_MODEL_LABEL

from .conftest import seed_model, tab


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CASESORTER_DATA_DIR", str(tmp_path / "data"))


def cell(window: Any, row: int, col: int) -> str:
    return window.dashboard_page.table.item(row, col).text()


def test_the_dashboard_is_the_fixed_first_tab_with_a_row_per_sorter(window) -> None:
    window.new_sorter()

    window.tab_bar.setCurrentIndex(0)

    assert window.tab_bar.tabText(0) == DASHBOARD_TITLE
    assert window.dashboard_showing()
    assert window._close_buttons.get(0) is None
    assert window.tab_bar.tabButton(0, window.tab_bar.ButtonPosition.RightSide) is None
    table = window.dashboard_page.table
    assert table.rowCount() == 2
    assert [cell(window, r, COL_NAME) for r in range(2)] == ["Sorter 1", "Sorter 2"]
    # A fresh sorter: nothing connected, AI Config mode, idle, nothing sorted.
    assert cell(window, 1, COL_MODEL) == AI_CONFIG_MODEL_LABEL
    assert cell(window, 1, COL_RUN) == RUN_STATE_IDLE
    assert cell(window, 1, COL_CASES) == "0"


def test_a_row_follows_its_tab_without_a_timer(window, config) -> None:
    seed_model(config, {"WIN": 1}, name="Range brass")
    tab(window).bus.post("mode/changed", None)
    window.drain_all()
    sorter = tab(window)

    sorter.connect_serial(EMULATED_PORT)
    sorter.bus.post(
        "run/history",
        {"label": "WIN", "confidence": 93.0, "slot": 1, "image": np.zeros((32, 32, 3), np.uint8)},
    )
    sorter.bus.post("run/result", {"ok": True, "slot": 1})
    window.drain_all()

    assert cell(window, 0, COL_MODEL) == "Range brass"
    assert EMULATED_PORT in cell(window, 0, COL_BOARD)
    assert "connected" in cell(window, 0, COL_BOARD).lower()
    assert cell(window, 0, COL_CAMERA)
    assert cell(window, 0, COL_RESULT) == "WIN (93%)"
    assert cell(window, 0, COL_CASES) == "1"
    thumb = window.dashboard_page.table.item(0, COL_CROP).icon()
    assert not thumb.isNull(), "the last crop is not shown"


def test_start_on_a_row_runs_the_sort_pages_checks(window) -> None:
    """No board: the row's Start is disabled, and pressing the tab's own
    Start path says what the Sort page says."""
    button = window.dashboard_page.run_buttons[tab(window).sorter_id]

    assert not button.isEnabled()
    window.dashboard_page._toggle(tab(window))

    assert "Connect to the board first" in tab(window).last_status
    assert not tab(window).is_running


def test_start_on_a_row_refuses_a_model_that_cannot_classify(window, config) -> None:
    seed_model(config, {"WIN": 1}, name="No checkpoint")
    tab(window).bus.post("mode/changed", None)
    window.drain_all()
    tab(window).connect_serial(EMULATED_PORT)
    window.drain_all()
    notices: list[tuple[str, str]] = []
    window.notify = lambda title, text: notices.append((title, text))

    window.dashboard_page.run_buttons[tab(window).sorter_id].click()

    assert notices and notices[0][0] == "Model not ready"
    assert not tab(window).is_running


def test_clicking_a_name_opens_that_sorter(window) -> None:
    window.new_sorter()
    window.tab_bar.setCurrentIndex(0)

    window.dashboard_page._on_cell_clicked(0, COL_NAME)

    assert window.current_tab is tab(window)
    assert window.tab_bar.currentIndex() == 1
    assert not window.dashboard_showing()


def test_rows_follow_rename_and_close(window) -> None:
    window.new_sorter()
    window.ask_text = lambda *_a: "Line B"
    window.rename_tab(tab(window, 1))

    assert cell(window, 1, COL_NAME) == "Line B"

    window.close_tab(tab(window, 1))

    assert window.dashboard_page.table.rowCount() == 1
