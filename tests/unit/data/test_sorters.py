"""Tests for the sorter-tab roster and the single-machine upgrade."""

from __future__ import annotations

from pathlib import Path

import pytest

from sorter.data.config import Config
from sorter.data.db import Database
from sorter.data.repository import HeadstampRepo, ModelRepo, SettingsRepo
from sorter.data.sorters import (
    create_sorter,
    delete_sorter,
    ensure_default_sorter,
    front_sorter_id,
    list_sorters,
    rename_sorter,
    set_front_sorter_id,
)


def _new_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "x.db")
    db.ensure_initialized()
    return db


def test_upgrade_carries_a_single_machine_install_into_sorter_1(tmp_path: Path) -> None:
    """AC2: an install from before sorter tabs opens as "Sorter 1" looking
    exactly as it did — same port, camera, crop settings, active model, run
    options, bins and active template."""
    db = _new_db(tmp_path)
    settings = SettingsRepo(db)
    model = ModelRepo(db).list()[0]
    HeadstampRepo(db).add(model.id, "WIN", slot=3)
    HeadstampRepo(db).add(model.id, "FC", slot=0)
    # What a pre-tabs build left behind: global keys only.
    settings.set("serial", {"port": "COM7", "baud": 19200})
    settings.set("camera", {"index": 2})
    settings.set("image_proc", {"primer_mode": "use"})
    settings.set("default_model_id", model.id)
    settings.set("run_confidence_floor", 64)
    settings.set("run_package_mode", True)
    settings.set(f"active_slot_template:{model.id}:standard", 99)

    roster = ensure_default_sorter(db)

    assert [(r.id, r.name) for r in roster] == [(1, "Sorter 1")]
    assert front_sorter_id(db) == 1
    tab = Config(db, sorter_id=1).load()
    assert tab.serial["port"] == "COM7"
    assert tab.serial["baud"] == 19200
    assert tab.camera["index"] == 2
    assert tab.image_proc["primer_mode"] == "use"
    assert tab.active_model_id == model.id
    assert tab.run_confidence_floor == 64
    assert tab.run_package_mode is True
    assert settings.get(f"sorter:1:active_slot_template:{model.id}:standard") == 99
    assert tab.slot_for_headstamp("WIN") == 3
    assert tab.slot_for_headstamp("FC") == 0
    # The legacy keys stay, so an older build opening this DB still works.
    assert settings.get("default_model_id") == model.id
    assert settings.get("serial")["port"] == "COM7"


def test_upgrade_runs_once(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    ensure_default_sorter(db)
    Config(db, sorter_id=1).set_run_confidence_floor(10)
    SettingsRepo(db).set("run_confidence_floor", 90)

    assert [r.id for r in ensure_default_sorter(db)] == [1]
    assert Config(db, sorter_id=1).run_confidence_floor == 10


def test_two_sorters_route_the_same_model_to_different_bins(tmp_path: Path) -> None:
    """Each tab keeps its own live layout; the model's stored slots are only
    the starting point a tab copies the first time it uses the model."""
    db = _new_db(tmp_path)
    ensure_default_sorter(db)
    second = create_sorter(db)
    model = ModelRepo(db).list()[0]
    HeadstampRepo(db).add(model.id, "WIN", slot=3)
    first_tab = Config(db, sorter_id=1).load()
    second_tab = Config(db, sorter_id=second.id).load()
    first_tab.set_active_model_id(model.id)
    second_tab.set_active_model_id(model.id)

    first_tab.set_headstamp_slot("WIN", 5)

    assert first_tab.slot_for_headstamp("WIN") == 5
    assert second_tab.slot_for_headstamp("WIN") == 3
    assert HeadstampRepo(db).list_for_model(model.id)[0].slot == 3


def test_run_options_are_per_sorter(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    ensure_default_sorter(db)
    second = create_sorter(db)
    first_tab = Config(db, sorter_id=1)
    second_tab = Config(db, sorter_id=second.id)

    first_tab.set_run_package_mode(True)
    first_tab.set_run_confidence_floor(80)

    assert second_tab.run_package_mode is False
    assert second_tab.run_confidence_floor != 80


def test_names_default_to_the_lowest_free_number_and_must_be_unique(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    ensure_default_sorter(db)
    bench = create_sorter(db)
    assert bench.name == "Sorter 2"
    rename_sorter(db, bench.id, "Bench")
    assert create_sorter(db).name == "Sorter 2"

    with pytest.raises(ValueError):
        rename_sorter(db, bench.id, "sorter 1")
    with pytest.raises(ValueError):
        create_sorter(db, "   ")
    with pytest.raises(ValueError):
        create_sorter(db, "x" * 41)
    assert [r.name for r in list_sorters(db)] == ["Sorter 1", "Bench", "Sorter 2"]


def test_closing_a_sorter_forgets_its_settings_and_moves_the_front_tab(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    ensure_default_sorter(db)
    second = create_sorter(db)
    tab = Config(db, sorter_id=second.id).load()
    tab.serial["port"] = "COM4"
    tab.save()
    set_front_sorter_id(db, second.id)

    delete_sorter(db, second.id)

    assert [r.id for r in list_sorters(db)] == [1]
    assert front_sorter_id(db) == 1
    assert SettingsRepo(db).items_with_prefix(f"sorter:{second.id}:") == []
    # A tab opened later with the same id starts clean, not on COM4.
    reopened = create_sorter(db)
    assert reopened.id == second.id
    assert Config(db, sorter_id=reopened.id).load().serial["port"] != "COM4"


def test_the_last_sorter_cannot_be_closed(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    ensure_default_sorter(db)
    with pytest.raises(ValueError):
        delete_sorter(db, 1)
    assert [r.id for r in list_sorters(db)] == [1]
