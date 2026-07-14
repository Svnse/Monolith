"""intent_extract — Seed α: the pure-code, always-on prediction floor.

Mines a FROZEN, code-checkable commitment from Monolith's OWN just-produced
answer (NOT a model of the user, NOT a <predict> envelope). The commitment:

  prediction_set = {
    directions: [{move, referent}],   # <=4 ranked next-moves the answer bets on
    referents:  [...],                # capped top-k SALIENT staked referents
    source: "floor",
  }

`move` uses core/turn_classifier's intent_tags vocabulary (the richer tag set
from the SAME _classify_intent the settler runs on the reply — apples-to-apples
membership; task_type's 3 values are too coarse to discriminate). `referents`
is CAPPED to the salient few (frequency x earliness) — NOT every noun — which is
the fix for the broad-frozen-set problem (a passing mention can't pollute).

Pure + deterministic. No LLM. The Monoline card (core/intent_card) may ENRICH
this set; the floor always stands alone so predict fires even if the card dies.
"""
from __future__ import annotations

from core.friction_differ import _content_tokens

_REFERENT_CAP = 8          # max salient referents in the frozen set
_DIRECTION_CAP = 4         # max next-move directions
# turn_classifier intent_tags that aren't discriminating "moves"
_NON_MOVE_TAGS = frozenset({"chat", "vent"})

# Discourse / affirmation fillers — NOT subject referents. Excluded so a filler
# opener ("great, …", "ok") can't masquerade as a new focus (false redirect) or
# pad the referent set. Standard stopwords are already stripped by _content_tokens;
# these are the conversational-glue extras that survive it.
_DISCOURSE_FILLER: frozenset[str] = frozenset("""
great ok okay yes yeah yep yup sure thanks thank good nice sounds sound let lets
hey hello hmm well cool awesome perfect fine alright gotcha please exactly totally
sense gonna wanna kinda basically literally honestly anyway
""".split())


def salient_referents(text: str, cap: int = _REFERENT_CAP) -> list[str]:
    """Top-`cap` salient content referents of `text`, ranked by frequency x
    earliness. Deterministic. Reuses friction_differ._content_tokens (same
    tokenizer/stopwords the settler uses on the reply -> comparable sets)."""
    import re
    words = [w.lower() for w in re.findall(r"[a-z0-9_]+", text or "", re.I)]
    # earliness weight: earlier tokens weigh more (the answer's lead is what it staked)
    n = len(words) or 1
    scores: dict[str, float] = {}
    content = _content_tokens(text)  # the stopword-filtered set
    for i, w in enumerate(words):
        if w not in content or w in _DISCOURSE_FILLER:
            continue
        earliness = 1.0 - (i / n) * 0.5   # 1.0 -> 0.5 across the text
        scores[w] = scores.get(w, 0.0) + earliness
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:cap]]


def _moves_from(text: str) -> list[str]:
    """The intent_tags the classifier reads off `text`, as candidate moves."""
    try:
        from core import turn_classifier
        shape = turn_classifier.classify([{"role": "user", "content": text or ""}], {})
        tags = [t for t in (shape.intent_tags or ()) if t not in _NON_MOVE_TAGS]
        return tags or list(shape.intent_tags or ())
    except Exception:
        return []


def mine_staked(public_answer: str, last_user_msg: str = "") -> dict:
    """Build the floor prediction_set from the answer's staked content.

    Mines from the ANSWER (what Monolith committed to), not from a guess about
    the user. `last_user_msg` is accepted for symmetry / future use; the floor
    does not depend on it."""
    referents = salient_referents(public_answer)
    moves = _moves_from(public_answer)
    # Pair each move with the top staked referent -> a forward "next-move" bet.
    top_ref = referents[0] if referents else ""
    directions = [{"move": m, "referent": top_ref} for m in moves[:_DIRECTION_CAP]]
    return {"directions": directions, "referents": referents, "source": "floor"}


def merge_prediction_sets(floor: dict, card: dict | None) -> dict:
    """Merge the card's enrichment INTO the floor (card never replaces floor).

    Union of referents (capped) and directions (capped); source reflects whether
    the card contributed. The floor's referents lead (they're already salient)."""
    if not card:
        return floor
    refs = list(dict.fromkeys((floor.get("referents") or []) + (card.get("referents") or [])))[:_REFERENT_CAP]
    # dedupe directions by (move, referent)
    seen = set()
    directions: list[dict] = []
    for d in (card.get("directions") or []) + (floor.get("directions") or []):
        key = (str(d.get("move", "")), str(d.get("referent", "")))
        if key in seen:
            continue
        seen.add(key)
        directions.append({"move": d.get("move", ""), "referent": d.get("referent", "")})
        if len(directions) >= _DIRECTION_CAP:
            break
    return {"directions": directions, "referents": refs, "source": "card+floor",
            "intent_read": card.get("intent_read", "")}
