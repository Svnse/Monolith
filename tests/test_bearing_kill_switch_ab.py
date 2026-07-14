"""Bearing V0 kill-switch A/B harness — scaffolding.

The V0 ship gate is the session-shaped A/B test described in plan §8. This
file ships the runnable skeleton:

  * `load_fixture(path)` — fixture file parsing + shape validation.
  * `run_fixture(fixture, arm)` — turn-by-turn driver that calls a model.
  * `score_pair(on_result, off_result, fixture)` — criterion scoring.

What's NOT here (out of scope for V0 engineering work):

  * Real LLM dispatch in `run_fixture` — requires a wired chat surface,
    fixture sessions designed by E, and a budget for ~36 runs.
  * Real human rating in `score_pair` — criteria 1, 3, 4 need a human-in-loop.
  * Real fixture content — `tests/fixtures/bearing_ab/*.json` ships templates only.

The tests in this file verify fixture-file SHAPE, harness PLUMBING, and a
skip-marked end-to-end placeholder for the real A/B run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "bearing_ab"


# ── fixture loading ─────────────────────────────────────────────────


@dataclass(frozen=True)
class FixtureTurn:
    turn: int
    user: str
    notes: str = ""


@dataclass(frozen=True)
class Fixture:
    id: str
    shape: str  # "design" | "execution" | "single"
    turns: tuple[FixtureTurn, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_fixture(path: Path) -> Fixture:
    """Parse a fixture file. Raises ValueError on structural issues."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be an object")
    fid = data.get("id")
    if not isinstance(fid, str) or not fid:
        raise ValueError(f"{path}: missing 'id'")
    shape = data.get("shape")
    if shape not in ("design", "execution", "single"):
        raise ValueError(f"{path}: shape must be design|execution|single, got {shape!r}")
    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError(f"{path}: missing or empty 'turns'")
    turns = tuple(
        FixtureTurn(
            turn=int(t.get("turn", i + 1)),
            user=str(t.get("user", "")),
            notes=str(t.get("notes", "")),
        )
        for i, t in enumerate(raw_turns)
        if isinstance(t, dict)
    )
    metadata = {k: v for k, v in data.items() if k not in ("id", "shape", "turns")}
    return Fixture(id=fid, shape=shape, turns=turns, metadata=metadata)


# ── runner (skeleton; real LLM call out of scope) ───────────────────


@dataclass(frozen=True)
class ArmResult:
    arm: str  # "on" | "off"
    fixture_id: str
    responses: tuple[str, ...]  # model output per turn


def run_fixture(fixture: Fixture, arm: str, *, dispatcher=None) -> ArmResult:
    """Drive a fixture through a dispatcher; one model call per fixture turn.

    `dispatcher` is a callable `(messages, arm) -> response_text`. The default
    in tests is a stub — real implementations wire to the actual chat surface.
    """
    if arm not in ("on", "off"):
        raise ValueError(f"arm must be 'on' or 'off', got {arm!r}")
    if dispatcher is None:
        dispatcher = _stub_dispatcher
    messages: list[dict] = []
    responses: list[str] = []
    for t in fixture.turns:
        messages.append({"role": "user", "content": t.user})
        response = dispatcher(messages, arm)
        responses.append(response)
        messages.append({"role": "assistant", "content": response})
    return ArmResult(arm=arm, fixture_id=fixture.id, responses=tuple(responses))


def _stub_dispatcher(messages: list[dict], arm: str) -> str:
    """Deterministic stub — for harness tests only. Real runs need a wired
    chat surface (engine/agent_server or equivalent)."""
    return f"[stub:{arm}] response to turn {len(messages)}"


# ── scoring (skeleton; real rating out of scope) ────────────────────


@dataclass(frozen=True)
class ScoreReport:
    fixture_id: str
    criterion_1: float | None = None
    criterion_2: int | None = None
    criterion_3: float | None = None
    criterion_4_proceeded_without_check_pct: float | None = None
    criterion_5_within_5pct: bool | None = None
    notes: str = ""


def score_pair(
    on_result: ArmResult, off_result: ArmResult, fixture: Fixture
) -> ScoreReport:
    """Skeleton scoring. Criteria 1, 3, 4 require a human rater — this
    function returns None for those criteria until a rating surface is wired.

    Criterion 2 (closed-branch re-exploration) can be scored mechanically
    against fixture pre-annotations — implemented stub-only here.
    """
    if on_result.fixture_id != fixture.id or off_result.fixture_id != fixture.id:
        raise ValueError("arm result fixture_ids must match")
    return ScoreReport(
        fixture_id=fixture.id,
        notes="harness scaffolding — real scoring needs human rating + pre-annotation",
    )


# ── fixture-file shape tests ────────────────────────────────────────


def _fixture_files() -> list[Path]:
    if not FIXTURES_DIR.exists():
        return []
    return sorted(
        p for p in FIXTURES_DIR.glob("*.json")
        if not p.name.endswith(".annotations.json") and p.name != "manifest.json"
    )


def test_fixtures_dir_exists() -> None:
    assert FIXTURES_DIR.exists(), f"Expected fixtures dir at {FIXTURES_DIR}"


def test_at_least_one_fixture_present() -> None:
    files = _fixture_files()
    assert len(files) >= 1, (
        "At least one fixture file must exist (V0 ships templates; "
        "real A/B run requires the full 18 fixtures per plan §8)"
    )


@pytest.mark.parametrize("path", _fixture_files() or [pytest.param(None, marks=pytest.mark.skip(reason="no fixtures"))])
def test_fixture_file_parses(path: Path) -> None:
    fixture = load_fixture(path)
    assert fixture.id
    assert fixture.shape in ("design", "execution", "single")
    assert len(fixture.turns) >= 1


def test_load_fixture_rejects_missing_id(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('{"shape": "design", "turns": [{"turn": 1, "user": "u"}]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_fixture(p)


def test_load_fixture_rejects_bad_shape(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('{"id": "x", "shape": "alien", "turns": [{"turn": 1, "user": "u"}]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_fixture(p)


# ── runner plumbing tests ───────────────────────────────────────────


def test_run_fixture_dispatches_one_call_per_turn(tmp_path) -> None:
    p = tmp_path / "f.json"
    p.write_text(
        json.dumps({
            "id": "f",
            "shape": "single",
            "turns": [
                {"turn": 1, "user": "u1"},
                {"turn": 2, "user": "u2"},
            ],
        }),
        encoding="utf-8",
    )
    fixture = load_fixture(p)
    calls = []

    def _dispatcher(messages, arm):
        calls.append((len(messages), arm))
        return f"reply {len(calls)}"

    result = run_fixture(fixture, "on", dispatcher=_dispatcher)
    assert result.arm == "on"
    assert len(result.responses) == 2
    assert len(calls) == 2


def test_run_fixture_rejects_invalid_arm(tmp_path) -> None:
    p = tmp_path / "f.json"
    p.write_text(
        json.dumps({"id": "f", "shape": "single", "turns": [{"turn": 1, "user": "u"}]}),
        encoding="utf-8",
    )
    fixture = load_fixture(p)
    with pytest.raises(ValueError):
        run_fixture(fixture, "maybe")


def test_score_pair_mismatched_fixture_ids_raise(tmp_path) -> None:
    p = tmp_path / "f.json"
    p.write_text(
        json.dumps({"id": "f", "shape": "single", "turns": [{"turn": 1, "user": "u"}]}),
        encoding="utf-8",
    )
    fixture = load_fixture(p)
    a = ArmResult(arm="on", fixture_id="f", responses=("r",))
    b = ArmResult(arm="off", fixture_id="other", responses=("r",))
    with pytest.raises(ValueError):
        score_pair(a, b, fixture)


# ── full A/B run — skip-marked until real fixtures + dispatcher exist ─


@pytest.mark.skip(
    reason="Real A/B run requires (1) 18 designed fixtures, (2) wired chat surface for "
           "dispatcher, (3) human rater for criteria 1/3/4. V0 plan §8 marks this as the "
           "ship-gate test — operational prerequisites must land before this runs."
)
def test_full_ab_run_passes_v0_ship_gate() -> None:
    """Placeholder: runs all 18 fixtures × 2 arms, scores per §8, asserts
    criteria 1–4 pass on multi-turn AND criterion 5 confirms no single-turn
    regression."""
    multi_turn = [load_fixture(p) for p in _fixture_files() if "single" not in p.name]
    single_turn = [load_fixture(p) for p in _fixture_files() if "single" in p.name]
    assert len(multi_turn) >= 6, "Need ≥6 multi-turn fixtures"
    assert len(single_turn) >= 12, "Need ≥12 single-turn fixtures"
    # ... real run + scoring goes here once prerequisites are wired ...
