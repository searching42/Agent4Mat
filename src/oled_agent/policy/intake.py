from __future__ import annotations

from typing import Dict, List, Tuple

INTENT_TARGETED = "targeted_difficult_regime_candidate"
INTENT_FALLBACK = "fallback_probe"
INTENT_BLIND = "blind_continuation"
INTENT_ROUTINE = "routine_validation"
INTENT_DEBUG_OR_ACCEPTANCE = "debug_or_acceptance"

ALLOWED_INTAKE_INTENTS = {
    INTENT_TARGETED,
    INTENT_FALLBACK,
    INTENT_BLIND,
    INTENT_ROUTINE,
    INTENT_DEBUG_OR_ACCEPTANCE,
}

ALLOWED_INTAKE_REFERENCE_REGIMES = {
    "aa_like",
    "ab_like",
    "ac_like",
    "ad_like",
    "ae_like",
    "af_like",
    "ag_like",
    "mixed_or_unknown",
}

EXPECTED_CLASS_MAP = {
    "classt": "Class T",
    "classs": "Class S",
    "classm": "Class M",
    "n/a": "n/a",
    "na": "n/a",
}

ALLOWED_INTAKE_EXPECTED_CLASSES = set(EXPECTED_CLASS_MAP.values())
DISALLOWED_INTAKE_SOURCE_NOTES = {"test", "continue", "see what happens"}


def normalize_expected_class(value: str) -> str:
    key = (value or "").strip().lower().replace(" ", "")
    return EXPECTED_CLASS_MAP.get(key, "")


def validate_intake_values(
    *,
    intent: str,
    reference_regime: str,
    source_note: str,
    expected_class: str,
) -> Tuple[Dict[str, str], List[str]]:
    normalized_intent = (intent or "").strip().lower()
    normalized_reference_regime = (reference_regime or "").strip().lower()
    normalized_source_note = (source_note or "").strip()
    normalized_expected_class = normalize_expected_class(expected_class or "")

    errors: List[str] = []

    if not normalized_intent:
        errors.append("missing intent")
    elif normalized_intent not in ALLOWED_INTAKE_INTENTS:
        errors.append(f"invalid intent: {normalized_intent}")

    if not normalized_reference_regime:
        errors.append("missing reference_regime")
    elif normalized_reference_regime not in ALLOWED_INTAKE_REFERENCE_REGIMES:
        errors.append(f"invalid reference_regime: {normalized_reference_regime}")

    if not normalized_source_note:
        errors.append("source_note must be non-empty")
    elif normalized_source_note.lower() in DISALLOWED_INTAKE_SOURCE_NOTES:
        errors.append("source_note placeholder is not allowed")

    if not normalized_expected_class:
        errors.append("missing or invalid expected_class")

    if normalized_intent == INTENT_DEBUG_OR_ACCEPTANCE and normalized_expected_class:
        if normalized_expected_class != "n/a":
            errors.append("debug_or_acceptance must use expected_class=n/a")
    elif normalized_intent and normalized_expected_class == "n/a":
        errors.append("non-debug intents cannot use expected_class=n/a")

    return {
        "intent": normalized_intent,
        "reference_regime": normalized_reference_regime,
        "source_note": normalized_source_note,
        "expected_class": normalized_expected_class,
    }, errors
