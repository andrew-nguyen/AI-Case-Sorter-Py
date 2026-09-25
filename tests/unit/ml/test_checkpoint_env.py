"""A checkpoint's PyTorch floor: what it records, and what a failure says (#77).

No torch anywhere. The two things under test are both reachable without it:
`local_inference`'s message builder is pure string work over an exception and a
dict, and `classifier.torch_floor_problem` compares two version strings from
the model row. That is deliberate — the whole point of recording the floor on
the row is that it can be read on a machine whose torch is too old to open the
checkpoint at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sorter.data.models import CheckpointEnv, Model
from sorter.ml import classifier, local_inference

TRAINED_WITH = {"torch_version": "2.13.0", "torchvision_version": "0.28.0", "numpy_version": "2.3.1"}


# ----- what the checkpoint carries --------------------------------------------


def test_a_payload_without_provenance_reads_as_empty() -> None:
    # Every checkpoint trained before #77, and everything from elsewhere.
    assert local_inference.checkpoint_env({"classes": ["WIN"], "image_size": 480}) == {}


def test_a_payload_with_provenance_reads_back() -> None:
    assert local_inference.checkpoint_env({"classes": ["WIN"], **TRAINED_WITH}) == TRAINED_WITH


@pytest.mark.parametrize("payload", [None, "not-a-dict", 42, {"torch_version": ""}, {"torch_version": 2.13}])
def test_a_payload_that_is_not_a_dict_of_strings_reads_as_empty(payload) -> None:
    # A float version is exactly the shape that would render as "PyTorch 2.13"
    # and quietly mean the wrong thing.
    assert local_inference.checkpoint_env(payload) == {}


# ----- what a failed load says -------------------------------------------------


def _message(exc: Exception, env: dict[str, str], have: str = "2.9.1") -> str:
    return local_inference._incompatibility_message("/models/7/7.pth", exc, env, have)


def test_a_recorded_floor_names_both_versions() -> None:
    text = _message(RuntimeError("Missing key(s) in state_dict: features.0.weight"), TRAINED_WITH)

    assert "2.13.0" in text and "2.9.1" in text
    assert "Update PyTorch" in text
    # The raw error stays: it is what tells a maintainer which of the three
    # mechanisms fired.
    assert "Missing key(s)" in text


def test_no_recorded_floor_still_says_which_way_it_runs() -> None:
    """The retrofit case — every community model in circulation carries none."""
    text = _message(RuntimeError("PytorchStreamReader failed"), {})

    assert "2.9.1" in text
    assert "records no version of its own" in text
    assert "Updating PyTorch" in text


def test_the_numpy_rename_is_named_as_itself() -> None:
    # weights_only=True does not save you here: torch's allowlist permits numpy
    # arrays, so the checkpoint unpickles far enough to hit the rename.
    text = _message(ModuleNotFoundError("No module named 'numpy._core'"), {})

    assert "numpy 2" in text
    assert "numpy" in text.lower()
    assert "renamed" in text


def test_the_numpy_rename_beats_a_recorded_floor() -> None:
    """Two true statements; the specific one is the actionable one."""
    text = _message(ModuleNotFoundError("No module named 'numpy._core'"), TRAINED_WITH)

    assert "renamed" in text


def test_the_message_names_the_file_not_its_whole_path() -> None:
    text = _message(RuntimeError("boom"), TRAINED_WITH)

    assert "7.pth" in text
    assert "/models/7/" not in text


# ----- the pre-flight ----------------------------------------------------------


def _model(**kwargs) -> Model:
    return Model(id=7, name="Range brass", model_path="/models/7/7.pth", **kwargs)


def test_no_recorded_floor_never_refuses(monkeypatch) -> None:
    monkeypatch.setattr(local_inference, "installed_version", lambda: "2.9.1")
    assert classifier.torch_floor_problem(_model()) is None


def test_an_older_torch_than_the_model_needs_is_refused_before_a_run(monkeypatch) -> None:
    monkeypatch.setattr(local_inference, "installed_version", lambda: "2.9.1")
    problem = classifier.torch_floor_problem(_model(checkpoint_env=CheckpointEnv(torch="2.13.0")))

    assert problem is not None
    assert "2.13.0" in problem and "2.9.1" in problem
    assert "Range brass" in problem


def test_the_same_or_a_newer_torch_passes(monkeypatch) -> None:
    # A floor, not a match: newer reads older, which is the whole asymmetry.
    for have in ("2.13.0", "2.14.1"):
        monkeypatch.setattr(local_inference, "installed_version", lambda have=have: have)
        assert classifier.torch_floor_problem(_model(checkpoint_env=CheckpointEnv(torch="2.13.0"))) is None


def test_an_absent_torch_is_left_to_the_install_gate(monkeypatch) -> None:
    monkeypatch.setattr(local_inference, "installed_version", lambda: None)
    assert classifier.torch_floor_problem(_model(checkpoint_env=CheckpointEnv(torch="2.13.0"))) is None


def test_an_unparseable_version_fails_open(monkeypatch) -> None:
    """A source build is likelier than a lie, and refusing costs a working run."""
    monkeypatch.setattr(local_inference, "installed_version", lambda: "not-a-version")
    assert classifier.torch_floor_problem(_model(checkpoint_env=CheckpointEnv(torch="2.13.0"))) is None


def test_the_sort_preflight_asks_this_too(tmp_path, monkeypatch) -> None:
    """`checkpoint_problem` is what Start calls, so the floor has to reach it."""
    from sorter.data.db import Database
    from sorter.data.repository import ModelRepo

    db = Database(tmp_path / "test.db")
    db.ensure_initialized()
    model = ModelRepo(db).list()[0]
    checkpoint = tmp_path / "7.pth"
    checkpoint.write_bytes(b"not really a checkpoint")
    model.model_path = str(checkpoint)
    model.checkpoint_env = CheckpointEnv(torch="2.13.0")
    ModelRepo(db).update(model)

    monkeypatch.setattr(local_inference, "installed_version", lambda: "2.9.1")
    problem = classifier.checkpoint_problem(db, model_id=model.id)

    assert problem is not None and "2.13.0" in problem


# ----- the row round trip ------------------------------------------------------


def test_the_floor_survives_a_write_and_a_read(tmp_path) -> None:
    from sorter.data.db import Database
    from sorter.data.repository import ModelRepo

    db = Database(tmp_path / "test.db")
    db.ensure_initialized()
    repo = ModelRepo(db)
    model = repo.list()[0]
    model.checkpoint_env = CheckpointEnv(torch="2.13.0", torchvision="0.28.0", numpy="2.3.1")
    repo.update(model)

    reloaded = repo.get(model.id)
    assert reloaded is not None
    assert reloaded.checkpoint_env == CheckpointEnv(torch="2.13.0", torchvision="0.28.0", numpy="2.3.1")


def test_a_row_written_before_the_column_existed_reads_as_empty(tmp_path) -> None:
    from sorter.data.db import Database
    from sorter.data.repository import ModelRepo

    db = Database(tmp_path / "test.db")
    db.ensure_initialized()
    db.conn.execute("UPDATE models SET checkpoint_env_json = NULL")

    model = ModelRepo(db).list()[0]
    assert model.checkpoint_env.is_empty()


# ----- the payload invariant ---------------------------------------------------


# Every key a checkpoint this app writes may carry, and nothing else. Each one
# is a tensor state_dict, a list of str, a str, an int or a float — adding a key
# here means saying which. A numpy object among them is a trap with a delay on
# it: it loads fine everywhere it was written and raises `ModuleNotFoundError:
# numpy._core` on any machine running numpy 1.x, and `weights_only=True` does
# not intervene, because torch's allowlist permits numpy arrays.
APPROVED_PAYLOAD_KEYS = {
    "model_state_dict",
    "classes",
    "base",
    "image_size",
    "val_acc",
    "val_loss",
}
# The one spread allowed in a payload, and the only other thing that adds keys.
APPROVED_PAYLOAD_SPREAD = "checkpoint_env"


def _trainer_ast():
    """The trainer as source, not as an import — torch is the `[ml]` extra."""
    import ast

    import sorter.training

    path = Path(sorter.training.__file__).parent / "train_convnext.py"
    return ast.parse(path.read_text(encoding="utf-8"))


def _save_payloads(tree) -> list:
    """Every dict literal that reaches `torch.save`, including via a variable."""
    import ast

    names = {
        node.args[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save"
        and node.args
        and isinstance(node.args[0], ast.Name)
    }
    inline = [
        node.args[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "save"
        and node.args
        and isinstance(node.args[0], ast.Dict)
    ]
    bound = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Dict)
        and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
    ]
    assert inline or bound, "no torch.save payload found — did the trainer move?"
    return inline + bound


def test_the_checkpoint_payload_carries_only_approved_keys() -> None:
    import ast

    tree = _trainer_ast()
    keys: set[str] = set()
    for payload in _save_payloads(tree):
        keys.update(k.value for k in payload.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
    # `save_payload["val_acc"] = …` after the literal.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "save_payload"
            and isinstance(node.targets[0].slice, ast.Constant)
        ):
            keys.add(str(node.targets[0].slice.value))

    assert "model_state_dict" in keys
    assert keys <= APPROVED_PAYLOAD_KEYS, f"unapproved checkpoint keys: {sorted(keys - APPROVED_PAYLOAD_KEYS)}"


def test_the_only_spread_into_a_payload_is_the_env_helper() -> None:
    """A `**something()` is the other way a numpy value could get in."""
    import ast

    for payload in _save_payloads(_trainer_ast()):
        for key, value in zip(payload.keys, payload.values, strict=True):
            if key is not None:
                continue
            assert isinstance(value, ast.Call) and getattr(value.func, "id", None) == APPROVED_PAYLOAD_SPREAD, (
                "only checkpoint_env() may be spread into a checkpoint payload"
            )
