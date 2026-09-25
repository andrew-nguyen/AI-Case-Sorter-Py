"""The sorter-tab roster: which sorting machines this install drives.

Each sorter tab drives one physical machine. What a tab owns — its serial and
camera settings, the live copy of the image-processing settings, its active
model, the five run options, its live slot layouts and its active-template
pointers — is stored in the ``settings`` table under ``sorter:<id>:<name>``
(see ``Config``, which is constructed per tab). This module owns the roster
itself: the ordered list of tabs under the ``sorters`` key, id allocation,
naming, deletion, the front-tab pointer, and the one-shot upgrade that turns a
single-machine install into one tab.

No schema change: the roster is configuration, and an older build that opens
the database simply ignores the new keys and keeps reading its global ones,
which ``ensure_default_sorter`` deliberately leaves in place.
"""

from __future__ import annotations

from dataclasses import dataclass

from .db import Database
from .repository import (
    ACTIVE_MODEL_NAME,
    SORTERS_KEY,
    SettingsRepo,
    sorter_key,
    sorter_prefix,
)

FRONT_SORTER_KEY = "ui.front_sorter"
DEFAULT_NAME_STEM = "Sorter"
MAX_SORTER_NAME = 40

# Settings sections and options a tab owns. Each is stored under the tab's
# namespace with the same name the single-machine install used as its global
# key, which is what lets the upgrade copy them by name. The five `run_*`
# options are the Sort page's run options; `sort_while_training` is the Train
# page's, and is per tab because the Train page is.
SORTER_SCOPED_SECTIONS = ("serial", "image_proc", "camera")
SORTER_SCOPED_OPTIONS = (
    "run_confidence_floor",
    "run_store_images",
    "run_package_mode",
    "run_package_size",
    "run_auto_select_trays",
    "sort_while_training",
)

# The pre-tab install kept these globally. Read once, by the upgrade.
_LEGACY_ACTIVE_MODEL_KEY = "default_model_id"
_LEGACY_TEMPLATE_POINTER_PREFIX = "active_slot_template:"


@dataclass
class SorterRecord:
    id: int
    name: str


def list_sorters(db: Database) -> list[SorterRecord]:
    """The roster in tab order. Malformed entries are skipped, never raised."""
    raw = SettingsRepo(db).get(SORTERS_KEY) or []
    out: list[SorterRecord] = []
    seen: set[int] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("id")
        if not isinstance(sid, int) or sid in seen:
            continue
        seen.add(sid)
        out.append(SorterRecord(sid, str(entry.get("name") or f"{DEFAULT_NAME_STEM} {sid}")))
    return out


def _save(db: Database, records: list[SorterRecord]) -> None:
    SettingsRepo(db).set(SORTERS_KEY, [{"id": r.id, "name": r.name} for r in records])


def next_default_name(records: list[SorterRecord]) -> str:
    """The lowest free "Sorter N"."""
    taken = {r.name.casefold() for r in records}
    n = 1
    while f"{DEFAULT_NAME_STEM} {n}".casefold() in taken:
        n += 1
    return f"{DEFAULT_NAME_STEM} {n}"


def _validate_name(records: list[SorterRecord], name: str, *, exclude: int | None = None) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Enter a name for the sorter.")
    if len(name) > MAX_SORTER_NAME:
        raise ValueError(f"Sorter names are limited to {MAX_SORTER_NAME} characters.")
    if any(r.name.casefold() == name.casefold() and r.id != exclude for r in records):
        raise ValueError(f"A sorter named “{name}” already exists.")
    return name


def create_sorter(db: Database, name: str | None = None) -> SorterRecord:
    """Append a new, unconnected sorter to the roster and return it.

    The id is one past the highest in use. Reusing a closed tab's id is safe
    because closing deletes that tab's whole key namespace.
    """
    with db.transaction():
        records = list_sorters(db)
        record = SorterRecord(
            max((r.id for r in records), default=0) + 1,
            _validate_name(records, name) if name is not None else next_default_name(records),
        )
        # A stale namespace can only exist if an earlier delete was interrupted;
        # clearing it keeps the new tab from inheriting another machine's port.
        SettingsRepo(db).delete_prefix(sorter_prefix(record.id))
        _save(db, [*records, record])
    return record


def rename_sorter(db: Database, sorter_id: int, name: str) -> SorterRecord:
    """Rename a sorter. Raises ValueError on an empty, too-long or taken name."""
    with db.transaction():
        records = list_sorters(db)
        record = next((r for r in records if r.id == sorter_id), None)
        if record is None:
            raise ValueError("That sorter no longer exists.")
        record.name = _validate_name(records, name, exclude=sorter_id)
        _save(db, records)
    return record


def delete_sorter(db: Database, sorter_id: int) -> None:
    """Remove a sorter and every setting it owns. Models and images are untouched.

    Refuses to remove the last sorter: the app always drives at least one.
    """
    with db.transaction():
        records = list_sorters(db)
        remaining = [r for r in records if r.id != sorter_id]
        if len(remaining) == len(records):
            return
        if not remaining:
            raise ValueError("The last sorter can't be closed.")
        _save(db, remaining)
        SettingsRepo(db).delete_prefix(sorter_prefix(sorter_id))
        if front_sorter_id(db) == sorter_id:
            set_front_sorter_id(db, remaining[0].id)


def front_sorter_id(db: Database) -> int | None:
    value = SettingsRepo(db).get(FRONT_SORTER_KEY)
    return value if isinstance(value, int) else None


def set_front_sorter_id(db: Database, sorter_id: int) -> None:
    SettingsRepo(db).set(FRONT_SORTER_KEY, int(sorter_id))


def ensure_default_sorter(db: Database) -> list[SorterRecord]:
    """Guarantee a non-empty roster, upgrading a single-machine install once.

    The first time this runs on a database with no roster, it creates
    "Sorter 1" carrying the install's current serial, camera and
    image-processing settings, its active model, its run options and its
    active-template pointers, so an upgraded install looks exactly as it did.
    The legacy global keys are copied, not moved, so an older build opening
    the same database still works. The live slot layout needs no copy here:
    ``Config`` seeds a tab's layout from the model's stored slots the first
    time the tab reads it, and at upgrade those stored slots *are* the live
    layout.

    Idempotent: a roster that already has a sorter is returned untouched.
    """
    with db.transaction():
        existing = list_sorters(db)
        if existing:
            return existing
        settings = SettingsRepo(db)
        record = SorterRecord(1, f"{DEFAULT_NAME_STEM} 1")
        settings.delete_prefix(sorter_prefix(record.id))
        for name in (*SORTER_SCOPED_SECTIONS, *SORTER_SCOPED_OPTIONS):
            value = settings.get(name)
            if value is not None:
                settings.set(sorter_key(record.id, name), value)
        for key, value in settings.items_with_prefix(_LEGACY_TEMPLATE_POINTER_PREFIX):
            settings.set(sorter_key(record.id, key), value)
        legacy_active = settings.get(_LEGACY_ACTIVE_MODEL_KEY)
        if legacy_active is not None:
            try:
                settings.set(sorter_key(record.id, ACTIVE_MODEL_NAME), int(legacy_active))
            except (TypeError, ValueError):
                pass
        _save(db, [record])
        set_front_sorter_id(db, record.id)
    return [record]
