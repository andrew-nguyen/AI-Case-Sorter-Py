"""SQLite-backed configuration for one sorter tab.

`Config` is constructed for a sorter tab (`Config(db, sorter_id=...)`, see
`data/sorters.py`). The tab's serial, camera and image-processing sections,
its active model, its run options, its live slot layouts and its
active-template pointers are stored under that tab's `sorter:<id>:` settings
namespace. The AI server section (`config.api`), headstamp and parent rows,
and sorting-template definitions are shared by every tab.

Slot assignments have two layers. The `headstamps.slot` /
`headstamp_parents.slot` columns and `package_slots:<model|ai>` hold a
model's *default* layout. Each tab copies that default into its own live
layout the first time it reads a model, and routes from its own copy after
that, so two machines can run the same model with different bins.

The DEFAULTS structure stays here as the canonical fallback when no settings
row exists yet.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from .db import Database
from .models import SLOT_TEMPLATE_MODES, SlotTemplate
from .repository import (
    ACTIVE_MODEL_NAME,
    HeadstampParentRepo,
    HeadstampRepo,
    SettingsRepo,
    SlotTemplateRepo,
    sorter_key,
)
from .sorters import SORTER_SCOPED_SECTIONS

DEFAULT_INIT_SETTINGS: dict[str, int | str] = {
    "feedhomingoffset": 0,
    "sorthomingoffset": 0,
    "feedspeed": 90,
    "sortspeed": 90,
    "feedsteps": 70,
    "sortsteps": 20,
    "slotdropdelay": 300,
    "notificationdelay": 160,
    "automotorstandbytimeout": 0,
    "feedmotorcurrent": 900,
    "sortmotorcurrent": 900,
    "fan": 100,
    "debounceTimeout": 500,
    "debounceTime": 300,
    "cameraledlevel": 130,
    "airdropenabled": 0,
    "airdroppredelay": 50,
    "airdropdsignalduration": 70,
    "airdroppostdelay": 50,
}


DEFAULTS: dict[str, Any] = {
    "api": {
        "endpoint_url": "http://localhost:8000",
        "api_key": "nokey",
        "model": "9mm",
        "prompt": "Not used for local AI Server",
        "image_quality": 100,
        "image_scale": 100,
    },
    "serial": {
        "port": "",
        "baud": 9600,
        "slot_quantity": 8,
        "handshake_timeout_s": 4.0,
        "init_on_startup": False,
        "init_settings": dict(DEFAULT_INIT_SETTINGS),
        "log_traffic": False,  # hardware/serial_log.py
    },
    "image_proc": {
        "strategy": "hough",
        "primer_mode": "hide",
        "primer_radius": 135,
        "hough": {
            "dp": 2.0,
            "min_dist": 500,
            "param1": 100,
            "param2": 60,
            "min_radius": 150,
            "max_radius": 250,
        },
        "linescan": {
            "scan_precision": 1,
            "scan_sensitivity": 5.0,
            "padding_pct": 5,
            "bg_cliff": 0,
        },
    },
    "camera": {
        "device_index": 0,
        "device_chosen": False,
        "width": 640,
        "height": 480,
    },
}


# The shared AI server settings are app-level; everything else Config caches
# belongs to the sorter tab it was built for (see `data/sorters.py`).
_API_SECTION = "api"
# AI Config mode (no active model) keeps its own headstamp list. The DB
# `headstamps` table requires a real model_id FK, so we stash AI Config
# headstamps in the key/value settings table instead.
_AI_HEADSTAMPS_KEY = "ai_config_headstamps"
# Parent-classification routing toggle. The model-level key
# (``use_parent_runtime:<model id>``) is the model's default, written by the
# WinForms import; a tab's own copy lives in its namespace and is what routes.
_USE_PARENT_RUNTIME_KEY = "use_parent_runtime"

# Run options, stored per tab under these names.
_RUN_CONFIDENCE_FLOOR_KEY = "run_confidence_floor"
_RUN_STORE_IMAGES_KEY = "run_store_images"
# Valid "store images" modes (internal value -> meaning):
#   none   never store
#   above  store only when confidence >= floor
#   below  store only when confidence < floor
#   all    store every classified case
STORE_IMAGES_MODES = ("none", "above", "below", "all")
DEFAULT_CONFIDENCE_FLOOR = 30

# Package mode (batch sorting). When on, the same headstamp may be assigned to
# several slots; the run fills one slot to `run_package_size` then advances to
# the next configured slot, halting when every slot for a headstamp is full.
# Package assignments are kept separate from the single-slot routing so a
# headstamp can live in multiple bins at once.
_RUN_PACKAGE_MODE_KEY = "run_package_mode"
_RUN_PACKAGE_SIZE_KEY = "run_package_size"
# `package_slots:<model id|ai>` is a model's *default* package layout (what a
# WinForms import writes and a tab seeds from), not the live one.
_PACKAGE_SLOTS_KEY = "package_slots"
DEFAULT_PACKAGE_SIZE = 50

# Auto-select trays: when on, an above-floor headstamp that isn't assigned to
# any slot is auto-routed to the first empty slot.
_RUN_AUTO_SELECT_KEY = "run_auto_select_trays"

# Live slot layouts. Each tab keeps its own, per model and per run mode, under
# `sorter:<id>:slots:<model id|ai>:<mode>`, in exactly the payload shape a
# sorting template stores. The `headstamps.slot` / `headstamp_parents.slot`
# columns and `package_slots:<model|ai>` are the model's *default* layout: a
# tab copies them into its own layout the first time it reads that scope, and
# after that they no longer move the tab.
_LIVE_SLOTS_KEY = "slots"

# Sorting templates: named layouts, shared per model AND per run mode —
# package mode's many-to-many assignments are a different shape. The *active*
# template is per tab and kept in lock-step with that tab's live layout (see
# `sync_active_slot_template`), so switching templates is a straight
# save-current / load-next swap.
# Key: `sorter:<id>:active_slot_template:<model id|ai>:<mode>` -> template id.
_ACTIVE_TEMPLATE_KEY = "active_slot_template"
DEFAULT_SLOT_TEMPLATE_NAME = "Default"

# Sort While Training: send xf:<slot> for a labelled case instead of xf:0
# during training.
_SORT_WHILE_TRAINING_KEY = "sort_while_training"


def _merge_defaults(defaults: Any, loaded: Any) -> Any:
    """Recursive default merge: any key missing in `loaded` falls back to defaults."""
    if isinstance(defaults, dict) and isinstance(loaded, dict):
        out: dict[str, Any] = {}
        for k, v in defaults.items():
            out[k] = _merge_defaults(v, loaded.get(k, v))
        for k in loaded:
            if k not in out:
                out[k] = loaded[k]
        return out
    return loaded if loaded is not None else defaults


def _positive_slot_map(raw: Any) -> dict[str, int]:
    """`{name: slot}` with every slot a positive int; anything else dropped."""
    out: dict[str, int] = {}
    if not isinstance(raw, dict):
        return out
    for name, slot in raw.items():
        try:
            value = int(slot)
        except (TypeError, ValueError):
            continue
        if name and value > 0:
            out[str(name)] = value
    return out


def _package_slot_lists(raw: Any) -> dict[str, list[str]]:
    """`{slot: [names]}` with every slot key a positive int string."""
    out: dict[str, list[str]] = {}
    if not isinstance(raw, dict):
        return out
    for key, names in raw.items():
        try:
            slot = int(key)
        except (TypeError, ValueError):
            continue
        if slot <= 0:
            continue
        out[str(slot)] = [str(n) for n in (names or []) if n]
    return out


_LIVE_KEY_RE = re.compile(r"^sorter:\d+:slots:(?P<scope>[^:]+):(?P<mode>[^:]+)$")


def rename_in_live_layouts(db: Database, model_id: int | None, old: str, new: str, *, parent: bool = False) -> None:
    """Carry a headstamp (or parent) rename into every sorter tab's live layout.

    A tab's live layout is keyed by name so that it survives a headstamp being
    deleted and re-added. The cost is that a rename, which keeps the row and
    its slot column, would otherwise leave every tab's copy pointing at a name
    that no longer exists, and that tab would silently drop the case into the
    catch-all. `model_id` None is AI Config mode's headstamp list.
    """
    if not old or not new or old == new:
        return
    settings = SettingsRepo(db)
    scope = str(model_id) if model_id is not None else "ai"
    with db.transaction():
        for key, payload in settings.items_with_prefix("sorter:"):
            match = _LIVE_KEY_RE.match(key)
            if match is None or match["scope"] != scope or not isinstance(payload, dict):
                continue
            changed = False
            if match["mode"] == "package":
                if parent:
                    continue
                slots = _package_slot_lists(payload.get("slots"))
                for slot, names in slots.items():
                    if old in names:
                        slots[slot] = [new if n == old else n for n in names]
                        changed = True
                payload = {"slots": slots}
            else:
                section = "parents" if parent else "headstamps"
                mapping = _positive_slot_map(payload.get(section))
                if old in mapping:
                    mapping[new] = mapping.pop(old)
                    payload = {**payload, section: mapping}
                    changed = True
            if changed:
                settings.set(key, payload)


class Config:
    """One sorter tab's view of the persisted settings.

    Every `Config` belongs to exactly one sorter tab (`sorter_id`, required):
    its serial, camera and image-processing sections, active model, run
    options, live slot layouts and active-template pointers are that tab's.
    The AI server section (`api`), headstamp rows, parent rows and template
    definitions are shared across tabs.

    Headstamps live in their own SQLite table (managed by HeadstampRepo) and
    are read fresh on every access — they are NOT part of the cached
    settings snapshot, because they get mutated through several call paths
    (Models tab editor, Train tab Save, Community import) and caching a
    stale snapshot here used to silently wipe rows whenever any other tab
    happened to call ``config.save()``. The shared `api` section is not
    cached either, for the same reason: another tab's `Config` may change it
    at any time, so `api` hands back a fresh copy and `save_api()` writes
    exactly the values it is given.
    """

    def __init__(self, db: Database, *, sorter_id: int) -> None:
        self.db = db
        self.sorter_id = int(sorter_id)
        self.settings = SettingsRepo(db)
        self.headstamps_repo = HeadstampRepo(db)
        self.parents_repo = HeadstampParentRepo(db)
        self.templates_repo = SlotTemplateRepo(db)
        self.data: dict[str, Any] = {s: copy.deepcopy(DEFAULTS[s]) for s in SORTER_SCOPED_SECTIONS}

    def _key(self, name: str) -> str:
        return sorter_key(self.sorter_id, name)

    def load(self) -> Config:
        for section in SORTER_SCOPED_SECTIONS:
            self.data[section] = self._merged(self.settings.get(self._key(section)), section)
        return self

    @staticmethod
    def _merged(stored: Any, section: str) -> dict[str, Any]:
        if stored is None:
            return copy.deepcopy(DEFAULTS[section])
        return _merge_defaults(copy.deepcopy(DEFAULTS[section]), stored)

    def save(self) -> None:
        """Persist this tab's cached sections. Touches neither headstamps nor `api`."""
        with self.db.transaction() as _:
            for section in SORTER_SCOPED_SECTIONS:
                self.settings.set(self._key(section), self.data[section])

    def save_api(self, values: dict[str, Any]) -> None:
        """Persist `values` as the shared AI server section (every tab sees it)."""
        self.settings.set(_API_SECTION, self._merged(values, _API_SECTION))

    # --- public surface -----------------------------------------------------

    @property
    def api(self) -> dict[str, Any]:
        """A fresh copy of the shared AI server settings.

        Editing the returned dict changes nothing until it is passed to
        `save_api()`.
        """
        return self._merged(self.settings.get(_API_SECTION), _API_SECTION)

    # ----- active model (per tab) -------------------------------------------

    @property
    def active_model_id(self) -> int | None:
        """This tab's active model, or None for AI Config mode. Read fresh."""
        value = self.settings.get(self._key(ACTIVE_MODEL_NAME))
        return value if isinstance(value, int) else None

    def set_active_model_id(self, model_id: int | None) -> None:
        """Activate `model_id` on this tab, or AI Config mode for None."""
        if model_id is None:
            self.settings.delete(self._key(ACTIVE_MODEL_NAME))
        else:
            self.settings.set(self._key(ACTIVE_MODEL_NAME), int(model_id))

    @property
    def headstamps(self) -> list[dict[str, Any]]:
        """The active context's headstamps with this tab's live slots. Read fresh.

        In AI Config mode (no active model) headstamps live in a settings
        entry instead of the model-scoped headstamps table.
        """
        names = self._headstamp_names(self.active_model_id)
        slots = self._live_standard()["headstamps"]
        return [{"name": name, "slot": int(slots.get(name, 0))} for name in names]

    def _headstamp_names(self, model_id: int | None) -> list[str]:
        if model_id is None:
            return [e["name"] for e in self._read_ai_headstamps()]
        return [h.name for h in self.headstamps_repo.list_for_model(model_id)]

    # ----- AI Config-mode headstamp storage ---------------------------------

    def _read_ai_headstamps(self) -> list[dict[str, Any]]:
        raw = self.settings.get(_AI_HEADSTAMPS_KEY) or []
        out: list[dict[str, Any]] = []
        for entry in raw:
            name = (entry or {}).get("name") if isinstance(entry, dict) else None
            if not name:
                continue
            out.append({"name": str(name), "slot": int((entry or {}).get("slot", 0))})
        return out

    def _write_ai_headstamps(self, entries: list[dict[str, Any]]) -> None:
        self.settings.set(_AI_HEADSTAMPS_KEY, entries)

    # ----- live slot layouts (per tab) --------------------------------------

    @staticmethod
    def _scope(model_id: int | None) -> str:
        return str(model_id) if model_id is not None else "ai"

    def _live_key(self, model_id: int | None, mode: str) -> str:
        return self._key(f"{_LIVE_SLOTS_KEY}:{self._scope(model_id)}:{mode}")

    def _default_layout(self, model_id: int | None, mode: str) -> dict[str, Any]:
        """The model's stored default layout, in template-payload shape."""
        if mode == "package":
            raw = self.settings.get(f"{_PACKAGE_SLOTS_KEY}:{self._scope(model_id)}")
            return {"slots": _package_slot_lists(raw)}
        if model_id is None:
            return {
                "headstamps": {e["name"]: int(e["slot"]) for e in self._read_ai_headstamps() if int(e["slot"]) > 0},
                "parents": {},
            }
        return {
            "headstamps": {
                h.name: int(h.slot) for h in self.headstamps_repo.list_for_model(model_id) if int(h.slot) > 0
            },
            "parents": {p.name: int(p.slot) for p in self.parents_repo.list_for_model(model_id) if int(p.slot) > 0},
        }

    def _live(self, model_id: int | None, mode: str) -> dict[str, Any]:
        """This tab's live layout for `(model_id, mode)`, seeded on first read.

        The seed is persisted immediately, so from then on the model's default
        layout (the slot columns) no longer moves this tab.
        """
        key = self._live_key(model_id, mode)
        with self.db.transaction():
            stored = self.settings.get(key)
            if not isinstance(stored, dict):
                stored = self._default_layout(model_id, mode)
                self.settings.set(key, stored)
        if mode == "package":
            return {"slots": _package_slot_lists(stored.get("slots"))}
        return {
            "headstamps": _positive_slot_map(stored.get("headstamps")),
            "parents": _positive_slot_map(stored.get("parents")),
        }

    def _write_live(self, model_id: int | None, mode: str, payload: dict[str, Any]) -> None:
        self.settings.set(self._live_key(model_id, mode), payload)

    def _live_standard(self) -> dict[str, Any]:
        return self._live(self.active_model_id, "standard")

    # ----- headstamp mutations ----------------------------------------------

    def add_headstamp(self, name: str, slot: int = 0) -> bool:
        """Add a headstamp for the active context. Returns False if `name`
        is empty or already present. A positive `slot` is recorded both as the
        model's default and in this tab's live layout.
        """
        if not name:
            return False
        active_id = self.active_model_id
        if active_id is None:
            current = self._read_ai_headstamps()
            if any(e["name"] == name for e in current):
                return False
            current.append({"name": name, "slot": int(slot)})
            self._write_ai_headstamps(current)
        else:
            existing = {h.name for h in self.headstamps_repo.list_for_model(active_id)}
            if name in existing:
                return False
            try:
                self.headstamps_repo.add(active_id, name, slot)
            except Exception:
                return False
        if int(slot) > 0:
            self.set_headstamp_slot(name, int(slot))
        return True

    def remove_headstamp(self, name: str) -> bool:
        active_id = self.active_model_id
        if active_id is None:
            current = self._read_ai_headstamps()
            remaining = [e for e in current if e["name"] != name]
            if len(remaining) == len(current):
                return False
            self._write_ai_headstamps(remaining)
        else:
            match = next((h for h in self.headstamps_repo.list_for_model(active_id) if h.name == name), None)
            if match is None:
                return False
            self.headstamps_repo.delete(match.id)
        live = self._live_standard()
        if live["headstamps"].pop(name, None) is not None:
            self._write_live(active_id, "standard", live)
        return True

    def clear_headstamps(self) -> None:
        active_id = self.active_model_id
        if active_id is None:
            self._write_ai_headstamps([])
        else:
            self.headstamps_repo.clear_for_model(active_id)
        live = self._live_standard()
        live["headstamps"] = {}
        self._write_live(active_id, "standard", live)

    def set_headstamp_slot(self, name: str, slot: int) -> bool:
        """Assign a headstamp of the active context to a slot on this tab.

        Returns False if the headstamp doesn't exist for the active context.
        Writes this tab's live layout only — the model's default layout and
        every other tab are untouched.
        """
        active_id = self.active_model_id
        if name not in self._headstamp_names(active_id):
            return False
        live = self._live_standard()
        if int(slot) > 0:
            live["headstamps"][name] = int(slot)
        else:
            live["headstamps"].pop(name, None)
        self._write_live(active_id, "standard", live)
        self.sync_active_slot_template("standard")
        return True

    @property
    def serial(self) -> dict[str, Any]:
        return self.data["serial"]

    @property
    def image_proc(self) -> dict[str, Any]:
        return self.data["image_proc"]

    @property
    def camera(self) -> dict[str, Any]:
        return self.data["camera"]

    def slot_for_headstamp(self, name: str) -> int | None:
        """Resolve the physical bin for a classified label on this tab.

        In parent-classification mode a child label routes to *its parent's*
        slot, while an orphan (parentless) headstamp routes to its own slot.
        Otherwise routing is the standard per-headstamp lookup. Returns None
        when the label maps to nothing (caller falls back to catch-all).
        """
        mid = self.active_model_id
        live = self._live_standard()
        if mid is not None and self.use_parent_classifications:
            parents = {p.id: p for p in self.parents_repo.list_for_model(mid)}
            if parents:
                headstamps = self.headstamps_repo.list_for_model(mid)
                hs = next((h for h in headstamps if h.name == name), None)
                if hs is not None:
                    if hs.parent_id is not None and hs.parent_id in parents:
                        return int(live["parents"].get(parents[hs.parent_id].name, 0))
                    return int(live["headstamps"].get(hs.name, 0))
                # The label may already be a parent name (parent-trained model).
                parent = next((p for p in parents.values() if p.name == name), None)
                return int(live["parents"].get(parent.name, 0)) if parent is not None else None

        if name in self._headstamp_names(mid):
            return int(live["headstamps"].get(name, 0))
        return None

    # ----- parent classifications --------------------------------------------

    def model_has_parents(self) -> bool:
        """True when the active local model has at least one parent defined.

        Drives whether the "Use Parent Classifications" run option is shown.
        Always False in AI Config mode (those headstamps have no parents).
        """
        mid = self.active_model_id
        if mid is None:
            return False
        return bool(self.parents_repo.list_for_model(mid))

    @property
    def use_parent_classifications(self) -> bool:
        """This tab's parent-routing toggle for its active model.

        Falls back to the model's default (the model-level key) until the tab
        sets its own. False in AI Config mode.
        """
        mid = self.active_model_id
        if mid is None:
            return False
        own = self.settings.get(self._key(f"{_USE_PARENT_RUNTIME_KEY}:{mid}"))
        if own is not None:
            return bool(own)
        return bool(self.settings.get(f"{_USE_PARENT_RUNTIME_KEY}:{mid}", False))

    def set_use_parent_classifications(self, value: bool) -> bool:
        mid = self.active_model_id
        if mid is None:
            return False
        self.settings.set(self._key(f"{_USE_PARENT_RUNTIME_KEY}:{mid}"), bool(value))
        return True

    def parents_with_slots(self) -> list[dict[str, Any]]:
        """[{id, name, slot}] for the active model's parents (empty in AI mode)."""
        mid = self.active_model_id
        if mid is None:
            return []
        slots = self._live_standard()["parents"]
        return [
            {"id": p.id, "name": p.name, "slot": int(slots.get(p.name, 0))}
            for p in self.parents_repo.list_for_model(mid)
        ]

    def headstamps_with_parents(self) -> list[dict[str, Any]]:
        """[{name, slot, parent_id}] for the active model (parent_id None in AI mode)."""
        mid = self.active_model_id
        slots = self._live_standard()["headstamps"]
        if mid is None:
            return [
                {"name": e["name"], "slot": int(slots.get(e["name"], 0)), "parent_id": None}
                for e in self._read_ai_headstamps()
            ]
        return [
            {"name": h.name, "slot": int(slots.get(h.name, 0)), "parent_id": h.parent_id}
            for h in self.headstamps_repo.list_for_model(mid)
        ]

    def set_parent_slot(self, parent_id: int, slot: int) -> bool:
        """Assign a parent classification to a slot on this tab. Local models only."""
        mid = self.active_model_id
        if mid is None:
            return False
        parent = self.parents_repo.get(int(parent_id))
        if parent is None or parent.model_id != mid:
            return False
        live = self._live_standard()
        if int(slot) > 0:
            live["parents"][parent.name] = int(slot)
        else:
            live["parents"].pop(parent.name, None)
        self._write_live(mid, "standard", live)
        self.sync_active_slot_template("standard")
        return True

    def parent_for_headstamp(self, name: str) -> str | None:
        """The parent classification name for a child headstamp, or None.

        Returns None in AI Config mode, for orphan (parentless) headstamps, or
        for unknown labels. Independent of the runtime toggle — callers decide
        whether to surface it.
        """
        mid = self.active_model_id
        if mid is None:
            return None
        hs = next(
            (h for h in self.headstamps_repo.list_for_model(mid) if h.name == name),
            None,
        )
        if hs is None or hs.parent_id is None:
            return None
        parent = self.parents_repo.get(hs.parent_id)
        return parent.name if parent else None

    # ----- run options (per tab) ---------------------------------------------

    @property
    def run_confidence_floor(self) -> int:
        """Minimum confidence (%) a prediction must reach to leave the catch-all.

        Predictions below this route to slot 0. 0 disables the floor.
        """
        try:
            return int(self.settings.get(self._key(_RUN_CONFIDENCE_FLOOR_KEY), DEFAULT_CONFIDENCE_FLOOR))
        except (TypeError, ValueError):
            return DEFAULT_CONFIDENCE_FLOOR

    def set_run_confidence_floor(self, value: int) -> None:
        self.settings.set(self._key(_RUN_CONFIDENCE_FLOOR_KEY), max(0, min(100, int(value))))

    @property
    def run_store_images(self) -> str:
        """One of STORE_IMAGES_MODES; controls run-image capture."""
        value = self.settings.get(self._key(_RUN_STORE_IMAGES_KEY), "none")
        return value if value in STORE_IMAGES_MODES else "none"

    def set_run_store_images(self, mode: str) -> None:
        if mode in STORE_IMAGES_MODES:
            self.settings.set(self._key(_RUN_STORE_IMAGES_KEY), mode)

    # ----- package mode (batch sorting) --------------------------------------

    @property
    def run_package_mode(self) -> bool:
        return bool(self.settings.get(self._key(_RUN_PACKAGE_MODE_KEY), False))

    def set_run_package_mode(self, value: bool) -> None:
        self.settings.set(self._key(_RUN_PACKAGE_MODE_KEY), bool(value))

    @property
    def run_package_size(self) -> int:
        try:
            value = int(self.settings.get(self._key(_RUN_PACKAGE_SIZE_KEY), DEFAULT_PACKAGE_SIZE))
        except (TypeError, ValueError):
            value = DEFAULT_PACKAGE_SIZE
        return value if value > 0 else DEFAULT_PACKAGE_SIZE

    def set_run_package_size(self, value: int) -> None:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return
        self.settings.set(self._key(_RUN_PACKAGE_SIZE_KEY), max(1, value))

    def package_slot_map(self) -> dict[int, list[str]]:
        """slot -> [headstamp names] for this tab's package layout."""
        slots = self._live(self.active_model_id, "package")["slots"]
        return {int(k): list(v) for k, v in slots.items()}

    def headstamps_in_package_slot(self, slot: int) -> list[str]:
        return list(self.package_slot_map().get(int(slot), []))

    def slots_for_headstamp_package(self, name: str) -> list[int]:
        """Every (non-catch-all) slot the headstamp is assigned to in package mode."""
        return sorted(s for s, names in self.package_slot_map().items() if s > 0 and name in names)

    def set_package_slot_headstamp(self, slot: int, name: str, enabled: bool) -> None:
        """Add/remove a headstamp from a package slot's assignment list.

        Unlike the single-slot routing this is many-to-many: a headstamp may be
        ticked into several slots so the run can fill them in batches.
        """
        if int(slot) <= 0 or not name:
            return
        mid = self.active_model_id
        live = self._live(mid, "package")
        key = str(int(slot))
        names = list(live["slots"].get(key, []))
        if enabled:
            if name not in names:
                names.append(name)
        else:
            names = [n for n in names if n != name]
        live["slots"][key] = names
        self._write_live(mid, "package", live)
        self.sync_active_slot_template("package")

    # ----- sorting templates --------------------------------------------------

    def slot_template_mode(self) -> str:
        """Which template list applies right now: 'package' or 'standard'."""
        return "package" if self.run_package_mode else "standard"

    def _active_template_key(self, mode: str, model_id: int | None = None) -> str:
        scope = self._scope(self.active_model_id if model_id is None else model_id)
        return self._key(f"{_ACTIVE_TEMPLATE_KEY}:{scope}:{mode}")

    def _resolve_template_mode(self, mode: str | None) -> str:
        if mode is None:
            return self.slot_template_mode()
        if mode not in SLOT_TEMPLATE_MODES:
            raise ValueError(f"Unsupported slot-template mode: {mode!r}")
        return mode

    def capture_slot_assignments(self, mode: str | None = None) -> dict[str, Any]:
        """Snapshot this tab's live assignments for `mode` into a template payload.

        Only non-catch-all assignments are stored; anything not mentioned is
        restored to slot 0 (unassigned) when the payload is applied. Standard
        mode keeps only names the active context still has.
        """
        mode = self._resolve_template_mode(mode)
        mid = self.active_model_id
        live = self._live(mid, mode)
        if mode == "package":
            return {"slots": {k: v for k, v in sorted(live["slots"].items(), key=lambda kv: int(kv[0])) if v}}
        names = set(self._headstamp_names(mid))
        parent_names = {p.name for p in self.parents_repo.list_for_model(mid)} if mid is not None else set()
        return {
            "headstamps": {n: s for n, s in live["headstamps"].items() if n in names},
            "parents": {n: s for n, s in live["parents"].items() if n in parent_names},
        }

    def apply_slot_assignments(
        self,
        assignments: dict[str, Any] | None,
        mode: str | None = None,
    ) -> None:
        """Write a template payload onto this tab's live layout.

        Names the payload doesn't mention are cleared to slot 0, so applying a
        template fully replaces the layout rather than merging into it. Unknown
        names (a headstamp deleted since the template was saved) are ignored.
        """
        mode = self._resolve_template_mode(mode)
        data = assignments if isinstance(assignments, dict) else {}
        mid = self.active_model_id
        if mode == "package":
            self._write_live(mid, "package", {"slots": _package_slot_lists(data.get("slots"))})
            return
        names = set(self._headstamp_names(mid))
        parent_names = {p.name for p in self.parents_repo.list_for_model(mid)} if mid is not None else set()
        self._write_live(
            mid,
            "standard",
            {
                "headstamps": {n: s for n, s in _positive_slot_map(data.get("headstamps")).items() if n in names},
                "parents": {n: s for n, s in _positive_slot_map(data.get("parents")).items() if n in parent_names},
            },
        )

    def list_slot_templates(self, mode: str | None = None) -> list[SlotTemplate]:
        """Templates for the active model + `mode`, newest scope seeded lazily.

        The first read of a scope with no templates creates "Default" holding
        whatever this tab currently has assigned, so upgrading users keep their
        layout and land on a named template without doing anything.
        """
        mode = self._resolve_template_mode(mode)
        mid = self.active_model_id
        rows = self.templates_repo.list_for_scope(mid, mode)
        if rows:
            return rows
        with self.db.transaction() as _:
            template = self.templates_repo.create(
                mid,
                mode,
                DEFAULT_SLOT_TEMPLATE_NAME,
                self.capture_slot_assignments(mode),
            )
            self.settings.set(self._active_template_key(mode), template.id)
        return [template]

    def active_slot_template(self, mode: str | None = None) -> SlotTemplate:
        """The template currently driving this tab's live assignments for `mode`."""
        mode = self._resolve_template_mode(mode)
        rows = self.list_slot_templates(mode)
        stored = self.settings.get(self._active_template_key(mode))
        for row in rows:
            if row.id == stored:
                return row
        # Pointer missing or stale (template deleted elsewhere, or a tab that
        # has never used this model): adopt the first one without applying it —
        # the live assignments are what the tab last worked with, and the next
        # sync writes them into it.
        self.settings.set(self._active_template_key(mode), rows[0].id)
        return rows[0]

    def sync_active_slot_template(self, mode: str | None = None) -> None:
        """Persist this tab's live assignments into its active template.

        Called after every slot-assignment mutation so the active template
        never drifts from what the Sort page shows — there is no explicit
        "save template" step.
        """
        mode = self._resolve_template_mode(mode)
        template = self.active_slot_template(mode)
        self.templates_repo.update_assignments(
            template.id,
            self.capture_slot_assignments(mode),
        )

    def activate_slot_template(
        self,
        template_id: int,
        mode: str | None = None,
    ) -> SlotTemplate | None:
        """Switch this tab to another template: save the outgoing one, load the incoming.

        Returns the newly active template, or None when `template_id` doesn't
        belong to the active model + mode.
        """
        mode = self._resolve_template_mode(mode)
        mid = self.active_model_id
        target = self.templates_repo.get(int(template_id))
        if target is None or target.mode != mode or target.model_id != mid:
            return None
        current = self.active_slot_template(mode)
        if current.id == target.id:
            return current
        with self.db.transaction() as _:
            self.templates_repo.update_assignments(
                current.id,
                self.capture_slot_assignments(mode),
            )
            self.apply_slot_assignments(target.assignments, mode)
            self.settings.set(self._active_template_key(mode), target.id)
        return target

    def create_slot_template(
        self,
        name: str,
        *,
        copy_current: bool = True,
        mode: str | None = None,
    ) -> SlotTemplate:
        """Create a template and make it active on this tab.

        With `copy_current` the new template starts as a copy of the live
        assignments (the outgoing template keeps its own copy); without it the
        slots are cleared so the user starts from a blank layout.

        Raises ValueError on an empty or duplicate name.
        """
        mode = self._resolve_template_mode(mode)
        name = (name or "").strip()
        if not name:
            raise ValueError("Enter a name for the template.")
        mid = self.active_model_id
        if self.templates_repo.find_by_name(mid, mode, name) is not None:
            raise ValueError(f"A template named “{name}” already exists.")
        current = self.active_slot_template(mode)
        payload = (
            self.capture_slot_assignments(mode)
            if copy_current
            else ({"slots": {}} if mode == "package" else {"headstamps": {}, "parents": {}})
        )
        with self.db.transaction() as _:
            # Flush the outgoing template first so nothing unsaved is lost.
            self.templates_repo.update_assignments(
                current.id,
                self.capture_slot_assignments(mode),
            )
            template = self.templates_repo.create(mid, mode, name, payload)
            self.apply_slot_assignments(payload, mode)
            self.settings.set(self._active_template_key(mode), template.id)
        return template

    def rename_slot_template(self, template_id: int, name: str) -> SlotTemplate | None:
        """Rename a template. Raises ValueError on an empty or duplicate name."""
        template = self.templates_repo.get(int(template_id))
        if template is None:
            return None
        name = (name or "").strip()
        if not name:
            raise ValueError("Enter a name for the template.")
        if name == template.name:
            return template
        clash = self.templates_repo.find_by_name(template.model_id, template.mode, name)
        if clash is not None and clash.id != template.id:
            raise ValueError(f"A template named “{name}” already exists.")
        self.templates_repo.rename(template.id, name)
        template.name = name
        return template

    def delete_slot_template(self, template_id: int) -> SlotTemplate | None:
        """Delete a template and return whichever one is active on this tab afterwards.

        Deleting this tab's active template loads the next remaining one into
        this tab. Another tab that had it active adopts the first remaining
        template the next time it reads its pointer, without its live layout
        changing. The last template in a scope can't be deleted — there is
        always somewhere for the current assignments to live.
        """
        template = self.templates_repo.get(int(template_id))
        if template is None:
            return None
        mode = template.mode
        remaining = [t for t in self.templates_repo.list_for_scope(template.model_id, mode) if t.id != template.id]
        if not remaining:
            raise ValueError("A model needs at least one sorting template.")
        key = self._active_template_key(mode, template.model_id) if template.model_id is not None else None
        if key is None:
            key = self._key(f"{_ACTIVE_TEMPLATE_KEY}:ai:{mode}")
        was_active = self.settings.get(key) == template.id
        # Only touch the live assignments when the template being deleted is
        # the one currently loaded for this tab's active model.
        loaded = was_active and template.model_id == self.active_model_id
        with self.db.transaction() as _:
            self.templates_repo.delete(template.id)
            if was_active:
                if loaded:
                    self.apply_slot_assignments(remaining[0].assignments, mode)
                self.settings.set(key, remaining[0].id)
        return remaining[0] if was_active else self.active_slot_template(mode)

    # ----- auto-select / sort-while-training toggles -------------------------

    @property
    def run_auto_select_trays(self) -> bool:
        return bool(self.settings.get(self._key(_RUN_AUTO_SELECT_KEY), False))

    def set_run_auto_select_trays(self, value: bool) -> None:
        self.settings.set(self._key(_RUN_AUTO_SELECT_KEY), bool(value))

    @property
    def sort_while_training(self) -> bool:
        return bool(self.settings.get(self._key(_SORT_WHILE_TRAINING_KEY), False))

    def set_sort_while_training(self, value: bool) -> None:
        self.settings.set(self._key(_SORT_WHILE_TRAINING_KEY), bool(value))

    # ----- empty-slot discovery (auto-select trays) --------------------------

    def first_empty_slot(self, *, package: bool | None = None) -> int | None:
        """The lowest slot number (>0) with no headstamp/parent assigned.

        Honours `serial.slot_quantity` for the upper bound. In package mode the
        package assignment map is consulted; otherwise the single-slot routing
        plus any parent-slot assignments are considered "occupied".
        """
        if package is None:
            package = self.run_package_mode
        slot_count = int(self.serial.get("slot_quantity", 8))
        occupied: set[int] = set()
        if package:
            for s, names in self.package_slot_map().items():
                if names:
                    occupied.add(int(s))
        elif self.use_parent_classifications:
            # Parent mode: parent groups and ungrouped headstamps occupy slots.
            for p in self.parents_with_slots():
                if int(p["slot"]) > 0:
                    occupied.add(int(p["slot"]))
            for h in self.headstamps_with_parents():
                if h["parent_id"] is None and int(h["slot"]) > 0:
                    occupied.add(int(h["slot"]))
        else:
            # Child mode: only per-headstamp slots matter. Parent-slot
            # assignments belong to the other runtime mode and must not push
            # auto-select past empty child slots.
            for entry in self.headstamps:
                slot = int(entry.get("slot", 0))
                if slot > 0:
                    occupied.add(slot)
        for slot in range(1, max(1, slot_count)):
            if slot not in occupied:
                return slot
        return None

    def assign_headstamp_to_empty_slot(self, name: str) -> int | None:
        """Route an unassigned headstamp to the first empty slot. Returns the
        slot it landed in, or None when there is no free slot.

        Respects existing assignments and only ever places one headstamp into
        an empty slot.
        """
        if not name:
            return None
        package = self.run_package_mode
        if package:
            if self.slots_for_headstamp_package(name):
                return None  # already assigned somewhere
        elif self.slot_for_headstamp(name):
            return None
        slot = self.first_empty_slot(package=package)
        if slot is None:
            return None
        if package:
            self.set_package_slot_headstamp(slot, name, True)
        else:
            self.set_headstamp_slot(name, slot)
        return slot
