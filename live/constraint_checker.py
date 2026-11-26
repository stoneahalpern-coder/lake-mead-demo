"""
constraint_checker.py — Mulholland Multiagent System
=====================================================
THE SYMBOLIC LAYER — Part 2: Constraint Validator

Validates every LLM-generated proposal before it enters the simulation.

  LLM proposes  →  constraint_checker validates  →  accept / clamp + log

Works for all 13 agents:
  - State agents    : checked against compact entitlements (legal_framework.py)
  - Non-state agents: checked against treaty/rights ceilings (NON_STATE_CEILINGS_MAF)
  - Non-consumptive : BOR and Environmental always get 0.0 — skip allocation check

This deterministic gate is what makes the system neurosymbolic: no matter
how persuasive the LLM is, it cannot hallucinate past a hard numeric constraint.
"""

from legal_framework import (
    ALL_AGENT_CEILINGS_MAF,   # Unified ceiling for all 13 agents
    UPPER_BASIN_STATES,
    LOWER_BASIN_STATES,
    UPPER_BASIN_TOTAL_MAF,
    LOWER_BASIN_TOTAL_MAF,
    MEXICO_TREATY_SHORTAGE_MAF,
)

# Agents that are non-consumptive — they claim 0 water, so skip allocation check
NON_CONSUMPTIVE = {"Federal_BOR", "Environmental_ESA"}


def check_proposal(proposal: dict, all_current_proposals: list[dict]) -> dict:
    """
    Validates one agent's proposal against all symbolic legal constraints.

    Args:
        proposal             : The agent's output dict (parsed JSON from GPT).
        all_current_proposals: All proposals already accepted this round.
                               Used for cross-agent basin-level checks.
    Returns:
        valid      (bool) : True if no hard violations found.
        violations (list) : Hard breaches — block the proposal.
        warnings   (list) : Soft issues — logged but don't block.
    """
    violations = []
    warnings   = []

    agent      = proposal.get("state", "Unknown")
    allocation = float(proposal.get("proposed_allocation_maf", 0))

    # ── CONSTRAINT 1: Non-consumptive agents must propose 0.0 ────────────────
    # BOR and Environmental are legal/operational actors, not water claimants.
    if agent in NON_CONSUMPTIVE and allocation > 0.01:
        violations.append(
            f"NON-CONSUMPTIVE VIOLATION [{agent}]: Proposed {allocation:.3f} MAF "
            f"but this agent does not claim a water allocation (must be 0.0)."
        )
        return {"valid": False, "violations": violations, "warnings": warnings}

    # ── CONSTRAINT 2: Allocation ≤ applicable ceiling ────────────────────────
    # Uses ALL_AGENT_CEILINGS_MAF which covers both compact states and
    # non-state actors (tribal, treaty, agricultural).
    ceiling = ALL_AGENT_CEILINGS_MAF.get(agent, 0)
    if agent not in NON_CONSUMPTIVE and ceiling > 0:
        if allocation > ceiling * 1.05:       # 5% rounding tolerance
            violations.append(
                f"CEILING VIOLATION [{agent}]: Proposes {allocation:.3f} MAF "
                f"but ceiling is {ceiling:.3f} MAF "
                f"(excess: {allocation - ceiling:.3f} MAF)."
            )

    # ── CONSTRAINT 3: No negative allocations ────────────────────────────────
    if allocation < 0:
        violations.append(
            f"INVALID [{agent}]: Negative allocation ({allocation:.3f} MAF) proposed."
        )

    # ── CONSTRAINT 4: Conservation commitment must be realistic ──────────────
    conservation_pct = float(proposal.get("conservation_commitment_pct", 0))
    if not (0 <= conservation_pct <= 50):
        violations.append(
            f"INFEASIBLE [{agent}]: Conservation commitment {conservation_pct:.0f}% "
            f"is outside implementable range (0–50%)."
        )

    # ── CONSTRAINT 5: Basin-level totals (warning only — not a hard block) ───
    # Only applies to compact basin states; non-state agents don't count
    # toward compact basin totals.
    if agent in UPPER_BASIN_STATES or agent in LOWER_BASIN_STATES:
        if all_current_proposals:
            upper_total = sum(
                float(p.get("proposed_allocation_maf", 0))
                for p in all_current_proposals
                if p.get("state") in UPPER_BASIN_STATES
            )
            lower_total = sum(
                float(p.get("proposed_allocation_maf", 0))
                for p in all_current_proposals
                if p.get("state") in LOWER_BASIN_STATES
            )
            if agent in UPPER_BASIN_STATES:
                upper_total += allocation
            elif agent in LOWER_BASIN_STATES:
                lower_total += allocation

            if agent in UPPER_BASIN_STATES and upper_total > UPPER_BASIN_TOTAL_MAF * 1.10:
                warnings.append(
                    f"BASIN WARNING: Upper Basin running total {upper_total:.2f} MAF "
                    f"exceeds compact ceiling {UPPER_BASIN_TOTAL_MAF} MAF by >10%."
                )
            if agent in LOWER_BASIN_STATES and lower_total > LOWER_BASIN_TOTAL_MAF * 1.10:
                warnings.append(
                    f"BASIN WARNING: Lower Basin running total {lower_total:.2f} MAF "
                    f"exceeds compact ceiling {LOWER_BASIN_TOTAL_MAF} MAF by >10%."
                )

    # ── CONSTRAINT 6: Mexico treaty floor ────────────────────────────────────
    # If any agent explicitly references a Mexico delivery figure, validate it.
    mexico_alloc = proposal.get("mexico_allocation_maf")
    if mexico_alloc is not None:
        if float(mexico_alloc) < MEXICO_TREATY_SHORTAGE_MAF:
            violations.append(
                f"TREATY VIOLATION [{agent}]: Proposed Mexico delivery "
                f"{mexico_alloc} MAF is below treaty floor "
                f"{MEXICO_TREATY_SHORTAGE_MAF} MAF (Minute 323)."
            )

    return {
        "valid":      len(violations) == 0,
        "violations": violations,
        "warnings":   warnings,
    }


def check_basin_agreement(proposals: list[dict]) -> dict:
    """
    System-level feasibility check after all agents in a round have proposed.

    Only counts compact basin states toward basin totals —
    non-state actors (tribal, federal, etc.) are excluded from compact math.

    Returns basin totals and feasibility flags.
    """
    upper_total = sum(
        float(p.get("proposed_allocation_maf", 0))
        for p in proposals if p.get("state") in UPPER_BASIN_STATES
    )
    lower_total = sum(
        float(p.get("proposed_allocation_maf", 0))
        for p in proposals if p.get("state") in LOWER_BASIN_STATES
    )
    grand_total = upper_total + lower_total

    return {
        "upper_basin_total_maf":  round(upper_total, 3),
        "lower_basin_total_maf":  round(lower_total, 3),
        "grand_total_maf":        round(grand_total, 3),
        "upper_basin_feasible":   upper_total <= UPPER_BASIN_TOTAL_MAF * 1.05,
        "lower_basin_feasible":   lower_total <= LOWER_BASIN_TOTAL_MAF * 1.05,
        "system_feasible":        grand_total <= 16.0,
    }
