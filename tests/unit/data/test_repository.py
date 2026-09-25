"""Repository CRUD and constraint tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from sorter.data.config import Config
from sorter.data.db import Database
from sorter.data.models import Model
from sorter.data.repository import (
    CartridgeRepo,
    HeadstampRepo,
    ModelRepo,
    SettingsRepo,
)
from sorter.data.sorters import create_sorter, ensure_default_sorter


def _new_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.ensure_initialized()
    return db


def test_cartridge_crud(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    repo = CartridgeRepo(db)
    initial_count = len(repo.list())
    new_cart = repo.create(".223 Rem")
    assert new_cart.id
    found = repo.find_by_name(".223 Rem")
    assert found is not None and found.id == new_cart.id
    assert len(repo.list()) == initial_count + 1
    repo.rename(new_cart.id, ".223 Remington")
    renamed = repo.get(new_cart.id)
    assert renamed is not None and renamed.name == ".223 Remington"


def test_model_round_trip(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    cart_repo = CartridgeRepo(db)
    repo = ModelRepo(db)

    cart = cart_repo.create("45ACP")
    model = Model(name="my45", cartridge_id=cart.id, model_mode="convnext_small")
    model.training_config.epochs = 25
    model.image_processing.primer_mode = "use"
    saved = repo.create(model)
    assert saved.id

    loaded = repo.get(saved.id)
    assert loaded is not None
    assert loaded.name == "my45"
    assert loaded.model_mode == "convnext_small"
    assert loaded.training_config.epochs == 25
    assert loaded.image_processing.primer_mode == "use"


def test_model_rejects_unsupported_mode(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    repo = ModelRepo(db)
    cart_id = CartridgeRepo(db).create("xx").id
    with pytest.raises(ValueError):
        repo.create(Model(name="bad", cartridge_id=cart_id, model_mode="resnet50"))


def test_cannot_delete_last_model_in_cartridge(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    repo = ModelRepo(db)
    cart_id = CartridgeRepo(db).create("99mm").id
    only_model = repo.create(Model(name="only", cartridge_id=cart_id, model_mode="convnext_tiny"))
    with pytest.raises(ValueError):
        repo.delete(only_model.id)


def test_cannot_delete_a_model_active_on_any_sorter(tmp_path: Path) -> None:
    """Deleting a model some tab is sorting with would leave that tab pointing
    at nothing mid-shift, so the repo refuses and names the tab."""
    db = _new_db(tmp_path)
    ensure_default_sorter(db)
    second = create_sorter(db, "Bench")
    repo = ModelRepo(db)
    # Add a sibling so the "last in cartridge" rule does not pre-empt the
    # active-model rule.
    seed_model = repo.list()[0]
    repo.create(Model(name="sibling", cartridge_id=seed_model.cartridge_id, model_mode="convnext_tiny"))
    Config(db, sorter_id=second.id).set_active_model_id(seed_model.id)

    with pytest.raises(ValueError, match="Bench"):
        repo.delete(seed_model.id)
    assert repo.get(seed_model.id) is not None

    Config(db, sorter_id=second.id).set_active_model_id(None)
    repo.delete(seed_model.id)
    assert repo.get(seed_model.id) is None


def test_active_model_is_per_sorter(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    ensure_default_sorter(db)
    second = create_sorter(db)
    seed_model = ModelRepo(db).list()[0]
    first_tab = Config(db, sorter_id=1)
    first_tab.set_active_model_id(seed_model.id)

    assert first_tab.active_model_id == seed_model.id
    assert Config(db, sorter_id=second.id).active_model_id is None
    assert SettingsRepo(db).active_model_ids() == {seed_model.id}

    first_tab.set_active_model_id(None)
    assert first_tab.active_model_id is None
    assert SettingsRepo(db).active_model_ids() == set()


def test_headstamp_replace_atomically(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    repo = HeadstampRepo(db)
    # The seeded model is no longer auto-activated, but we can still target
    # it directly through the repo.
    model_id = ModelRepo(db).list()[0].id
    repo.add(model_id, "ALPHA", slot=1)
    repo.add(model_id, "BETA", slot=2)

    repo.replace_for_model(
        model_id,
        [
            {"name": "GAMMA", "slot": 3},
            {"name": "DELTA", "slot": 4},
        ],
    )
    names = sorted(h.name for h in repo.list_for_model(model_id))
    assert names == ["DELTA", "GAMMA"]


def test_headstamps_cascade_on_model_delete(tmp_path: Path) -> None:
    db = _new_db(tmp_path)
    cart_id = CartridgeRepo(db).create("9x18").id
    model = ModelRepo(db).create(Model(name="m1", cartridge_id=cart_id, model_mode="convnext_tiny"))
    # The row itself is the point: ModelRepo.delete refuses to remove the last
    # model in a cartridge, so the delete below needs a second model to get past
    # that guard and reach the cascade. Created, never referenced by name.
    ModelRepo(db).create(Model(name="m2", cartridge_id=cart_id, model_mode="convnext_tiny"))
    HeadstampRepo(db).add(model.id, "X")
    HeadstampRepo(db).add(model.id, "Y")
    ModelRepo(db).delete(model.id)
    assert HeadstampRepo(db).list_for_model(model.id) == []


def test_find_by_community_uid_prefers_the_active_duplicate(tmp_path: Path) -> None:
    """Libraries built by older versions can hold several rows for one UID
    (every update was imported as a new model). The lookup has to pick one
    row deterministically — the active model, else the oldest."""
    db = _new_db(tmp_path)
    cart = CartridgeRepo(db).get_or_create("9mm")
    repo = ModelRepo(db)
    first = repo.create(Model(name="Comm", cartridge_id=cart.id, community_model_uid="uid-dup"))
    second = repo.create(Model(name="Comm (2)", cartridge_id=cart.id, community_model_uid="uid-dup"))

    # No active model: oldest wins.
    dup = repo.find_by_community_uid("uid-dup")
    assert dup is not None and dup.id == first.id

    # A model active on some sorter tab wins, whichever duplicate it is.
    ensure_default_sorter(db)
    tab = Config(db, sorter_id=1)
    tab.set_active_model_id(second.id)
    dup = repo.find_by_community_uid("uid-dup")
    assert dup is not None and dup.id == second.id
    tab.set_active_model_id(first.id)
    dup = repo.find_by_community_uid("uid-dup")
    assert dup is not None and dup.id == first.id

    assert repo.find_by_community_uid("nope") is None
