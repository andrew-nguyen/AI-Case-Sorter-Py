"""Tests for the SQLite-backed Config shim."""

from __future__ import annotations

from pathlib import Path

from sorter.data.config import DEFAULT_INIT_SETTINGS, DEFAULTS, Config
from sorter.data.db import Database
from sorter.data.repository import ModelRepo


def _new_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.ensure_initialized()
    return db


def _activate_seeded_model(db: Database) -> int:
    """Activate the auto-seeded model so headstamp persistence has a target."""
    seed = ModelRepo(db).list()[0]
    Config(db, sorter_id=1).set_active_model_id(seed.id)
    return seed.id


def test_settings_round_trip(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    cfg = Config(db, sorter_id=1).load()
    cfg.save_api({**cfg.api, "api_key": "abc", "model": "test-model"})
    cfg.serial["baud"] = 19200
    cfg.save()

    reloaded = Config(db, sorter_id=1).load()
    assert reloaded.api["api_key"] == "abc"
    assert reloaded.api["model"] == "test-model"
    assert reloaded.serial["baud"] == 19200


def test_ai_server_settings_are_shared_and_serial_is_per_sorter(tmp_path: Path) -> None:
    """The AI server section is app-wide, so a second tab must see an edit made
    on the first. A tab saving its own serial settings must not write back a
    stale copy of the shared section over that edit."""
    db = _new_db(tmp_path)
    first = Config(db, sorter_id=1).load()
    second = Config(db, sorter_id=2).load()

    first.save_api({**first.api, "api_key": "shared"})
    second.serial["port"] = "COM9"
    second.save()

    assert Config(db, sorter_id=2).load().api["api_key"] == "shared"
    assert Config(db, sorter_id=1).load().api["api_key"] == "shared"
    assert Config(db, sorter_id=1).load().serial["port"] != "COM9"
    assert Config(db, sorter_id=2).load().serial["port"] == "COM9"


def test_add_remove_headstamps_via_helpers(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    _activate_seeded_model(db)
    cfg = Config(db, sorter_id=1).load()

    assert cfg.add_headstamp("WIN", slot=3)
    assert not cfg.add_headstamp("WIN")  # duplicate
    assert cfg.add_headstamp("FC", slot=5)

    reloaded = Config(db, sorter_id=1).load()
    names = sorted(h["name"] for h in reloaded.headstamps)
    assert names == ["FC", "WIN"]

    assert cfg.remove_headstamp("WIN")
    assert not cfg.remove_headstamp("WIN")  # already gone
    assert [h["name"] for h in Config(db, sorter_id=1).load().headstamps] == ["FC"]

    cfg.clear_headstamps()
    assert Config(db, sorter_id=1).load().headstamps == []


def test_save_does_not_overwrite_headstamps(tmp_path: Path) -> None:
    """Regression: config.save() must not wipe headstamps mutated through the repo.

    The Headstamps editor / Train tab / Community import all add headstamps
    directly via HeadstampRepo. An unrelated config.save() (e.g. saving
    serial settings) used to silently delete them because save() rewrote the
    headstamps table from a stale in-memory cache.
    """
    db = _new_db(tmp_path)
    _activate_seeded_model(db)
    cfg = Config(db, sorter_id=1).load()
    cfg.add_headstamp("ALPHA")

    # Simulate an unrelated save (e.g. user saved serial settings).
    cfg.serial["baud"] = 19200
    cfg.save()

    reloaded = Config(db, sorter_id=1).load()
    assert reloaded.serial["baud"] == 19200
    assert [h["name"] for h in reloaded.headstamps] == ["ALPHA"]


def test_defaults_returned_when_no_settings_row(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    cfg = Config(db, sorter_id=1).load()
    assert cfg.api["endpoint_url"] == DEFAULTS["api"]["endpoint_url"]
    assert cfg.serial["init_settings"]["feedspeed"] == DEFAULT_INIT_SETTINGS["feedspeed"]


def test_partial_settings_filled_with_defaults(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    db.conn.execute(
        "INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)",
        ("api", '{"model": "x"}'),
    )
    cfg = Config(db, sorter_id=1).load()
    assert cfg.api["model"] == "x"
    assert cfg.api["endpoint_url"] == DEFAULTS["api"]["endpoint_url"]


def test_run_confidence_floor_default_and_persist(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    cfg = Config(db, sorter_id=1).load()
    assert cfg.run_confidence_floor == 30  # default
    cfg.set_run_confidence_floor(75)
    assert Config(db, sorter_id=1).load().run_confidence_floor == 75
    cfg.set_run_confidence_floor(150)  # clamped to 0..100
    assert Config(db, sorter_id=1).load().run_confidence_floor == 100
    cfg.set_run_confidence_floor(-5)
    assert Config(db, sorter_id=1).load().run_confidence_floor == 0


def test_run_store_images_default_and_persist(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    cfg = Config(db, sorter_id=1).load()
    assert cfg.run_store_images == "none"  # default
    cfg.set_run_store_images("above")
    assert Config(db, sorter_id=1).load().run_store_images == "above"
    cfg.set_run_store_images("bogus")  # invalid -> ignored
    assert Config(db, sorter_id=1).load().run_store_images == "above"


def test_slot_lookup(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    _activate_seeded_model(db)
    cfg = Config(db, sorter_id=1).load()
    cfg.add_headstamp("WIN", slot=2)
    cfg.add_headstamp("FC", slot=5)

    reloaded = Config(db, sorter_id=1).load()
    assert reloaded.slot_for_headstamp("WIN") == 2
    assert reloaded.slot_for_headstamp("FC") == 5
    assert reloaded.slot_for_headstamp("Unknown") is None


def test_set_headstamp_slot_persists_in_both_modes(tmp_path: Path) -> None:
    """The Run tab's slot-details checkboxes call set_headstamp_slot()
    to route a headstamp to a slot. It must persist for both
    local-model and AI Config storage paths.
    """
    db = _new_db(tmp_path)
    _activate_seeded_model(db)
    cfg = Config(db, sorter_id=1).load()
    cfg.add_headstamp("WIN", slot=0)
    assert cfg.set_headstamp_slot("WIN", 4)
    assert Config(db, sorter_id=1).load().slot_for_headstamp("WIN") == 4
    assert cfg.set_headstamp_slot("WIN", 0)
    assert Config(db, sorter_id=1).load().slot_for_headstamp("WIN") == 0
    assert not cfg.set_headstamp_slot("MISSING", 1)

    # AI Config mode
    Config(db, sorter_id=1).set_active_model_id(None)
    cfg = Config(db, sorter_id=1).load()
    cfg.add_headstamp("LOADED")
    assert cfg.set_headstamp_slot("LOADED", 3)
    assert Config(db, sorter_id=1).load().slot_for_headstamp("LOADED") == 3


def test_headstamps_in_ai_config_mode_persist_via_settings(tmp_path: Path) -> None:
    """AI Config mode (no active model) keeps its own headstamp list.

    The DB headstamps table requires a real model FK, so AI-mode headstamps
    live in the settings key/value table instead. The Config public surface
    hides that storage detail from the tabs.
    """
    db = _new_db(tmp_path)
    cfg = Config(db, sorter_id=1).load()
    assert cfg.headstamps == []

    assert cfg.add_headstamp("LOADED-FROM-SERVER", slot=2)
    assert not cfg.add_headstamp("LOADED-FROM-SERVER")  # duplicate
    assert cfg.add_headstamp("ANOTHER")

    reloaded = Config(db, sorter_id=1).load()
    assert sorted(h["name"] for h in reloaded.headstamps) == ["ANOTHER", "LOADED-FROM-SERVER"]
    assert reloaded.slot_for_headstamp("LOADED-FROM-SERVER") == 2

    assert reloaded.remove_headstamp("LOADED-FROM-SERVER")
    assert [h["name"] for h in Config(db, sorter_id=1).load().headstamps] == ["ANOTHER"]

    Config(db, sorter_id=1).load().clear_headstamps()
    assert Config(db, sorter_id=1).load().headstamps == []


def test_headstamps_track_active_model(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    _activate_seeded_model(db)
    cfg = Config(db, sorter_id=1).load()
    cfg.add_headstamp("FOR-MODEL-1", slot=1)

    # Create a second cartridge + model and switch active.
    cart_row = db.conn.execute("INSERT INTO cartridges(name) VALUES ('45ACP')").lastrowid
    new_model_id = db.conn.execute(
        "INSERT INTO models(name, cartridge_id, model_mode) VALUES ('Other', ?, 'convnext_tiny')",
        (cart_row,),
    ).lastrowid
    assert new_model_id is not None
    Config(db, sorter_id=1).set_active_model_id(new_model_id)

    assert cfg.headstamps == []
    cfg.add_headstamp("FOR-MODEL-2", slot=7)

    # Switch back and verify the first model's headstamps survive.
    original_model_id = db.conn.execute(
        "SELECT id FROM models WHERE cartridge_id != ? LIMIT 1", (cart_row,)
    ).fetchone()["id"]
    Config(db, sorter_id=1).set_active_model_id(original_model_id)
    assert [h["name"] for h in cfg.headstamps] == ["FOR-MODEL-1"]
