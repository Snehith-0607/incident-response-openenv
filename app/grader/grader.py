def grade(actions_taken: list[dict], expected_actions: list[dict]) -> tuple[float, str]:
    """
    Grade a sequence of actions against the expected solution.

    Scoring rules
    -------------
    - Each expected step is worth one point.
    - A taken action earns the point for position i when:
        • action["type"] matches expected[i]["type"]          (required)
        • action["target"] matches expected[i]["target"]      (if present in expected)
        • action["params"] matches expected[i]["params"]      (if present in expected)
    - Extra actions beyond len(expected) each deduct a penalty.
    - Final score is clamped to [0.0, 1.0].

    Parameters
    ----------
    actions_taken    : list of action dicts the agent executed
    expected_actions : list of action dicts representing the ideal solution

    Returns
    -------
    score       : float in [0.0, 1.0]
    explanation : human-readable breakdown
    """
    if not expected_actions:
        return 1.0, "No expected actions defined — nothing to grade."

    if not actions_taken:
        return 0.0, (
            f"No actions taken. Expected {len(expected_actions)} step(s). Score: 0.0"
        )

    total_expected = len(expected_actions)
    penalty_per_extra = 1  # extra actions cost one point each

    correct_steps  = 0
    wrong_steps    = 0
    step_lines     = []

    # ── Step-by-step comparison (walk expected positions) ─────────────
    for i, expected in enumerate(expected_actions):
        if i >= len(actions_taken):
            step_lines.append(f"  Step {i+1}: MISSING  — expected {_fmt(expected)}")
            continue

        taken = actions_taken[i]
        match, reason = _actions_match(taken, expected)

        if match:
            correct_steps += 1
            step_lines.append(f"  Step {i+1}: CORRECT  — {_fmt(taken)}")
        else:
            wrong_steps += 1
            step_lines.append(
                f"  Step {i+1}: WRONG    — took {_fmt(taken)} | expected {_fmt(expected)} ({reason})"
            )

    # ── Penalty for extra (unnecessary) actions ───────────────────────
    extra_count = max(0, len(actions_taken) - total_expected)
    penalty     = extra_count * penalty_per_extra

    # ── Score calculation ─────────────────────────────────────────────
    raw_score = (correct_steps - penalty) / total_expected
    score     = round(max(0.0, min(1.0, raw_score)), 4)

    # ── Explanation ───────────────────────────────────────────────────
    lines = [
        f"Grading Report",
        f"  Expected steps : {total_expected}",
        f"  Steps taken    : {len(actions_taken)}",
        f"  Correct        : {correct_steps}",
        f"  Wrong/missing  : {total_expected - correct_steps}",
        f"  Extra actions  : {extra_count} (penalty: -{penalty})",
        f"",
        "Step Breakdown:",
        *step_lines,
        f"",
        f"Score: {score}",
    ]

    if score == 1.0:
        lines.append("Result: PERFECT — exact match.")
    elif score >= 0.75:
        lines.append("Result: GOOD — mostly correct with minor deviations.")
    elif score >= 0.4:
        lines.append("Result: PARTIAL — significant errors or missing steps.")
    else:
        lines.append("Result: POOR — incorrect or incomplete response.")

    return score, "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────

def _actions_match(taken: dict, expected: dict) -> tuple[bool, str]:
    """Return (match, reason_if_not)."""
    if taken.get("type") != expected.get("type"):
        return False, f"type mismatch ('{taken.get('type')}' ≠ '{expected.get('type')}')"

    if "target" in expected and taken.get("target") != expected.get("target"):
        return False, f"target mismatch ('{taken.get('target')}' ≠ '{expected.get('target')}')"

    if "params" in expected:
        taken_params    = taken.get("params", {})
        expected_params = expected.get("params", {})
        for key, val in expected_params.items():
            if taken_params.get(key) != val:
                return False, f"params['{key}'] mismatch ({taken_params.get(key)!r} ≠ {val!r})"

    return True, ""


def _fmt(action: dict) -> str:
    parts = [action.get("type", "?")]
    if action.get("target"):
        parts.append(f"target={action['target']}")
    if action.get("params"):
        parts.append(f"params={action['params']}")
    return "  ".join(parts)


# ── Smoke tests ───────────────────────────────────────────────────────

if __name__ == "__main__":
    expected = [
        {"type": "restart_service", "target": "database"},
        {"type": "restart_service", "target": "auth-service"},
        {"type": "update_config",   "params": {"timeout_seconds": 10, "retry_limit": 5}},
    ]

    cases = [
        ("Perfect match", [
            {"type": "restart_service", "target": "database"},
            {"type": "restart_service", "target": "auth-service"},
            {"type": "update_config",   "params": {"timeout_seconds": 10, "retry_limit": 5}},
        ]),
        ("Wrong order", [
            {"type": "restart_service", "target": "auth-service"},
            {"type": "restart_service", "target": "database"},
            {"type": "update_config",   "params": {"timeout_seconds": 10, "retry_limit": 5}},
        ]),
        ("Partial (first two only)", [
            {"type": "restart_service", "target": "database"},
            {"type": "restart_service", "target": "auth-service"},
        ]),
        ("Extra unnecessary action", [
            {"type": "restart_service", "target": "database"},
            {"type": "restart_service", "target": "auth-service"},
            {"type": "update_config",   "params": {"timeout_seconds": 10, "retry_limit": 5}},
            {"type": "escalate"},
        ]),
        ("Completely wrong", [
            {"type": "escalate"},
            {"type": "clear_logs"},
        ]),
        ("No actions taken", []),
    ]

    for label, taken in cases:
        score, explanation = grade(taken, expected)
        print(f"\n{'='*60}")
        print(f"  TEST: {label}")
        print('='*60)
        print(explanation)
