"""
negotiation.py — Mulholland Multiagent System
==============================================
NEGOTIATION ENGINE

Orchestrates one complete simulation run across all 13 agents:

  For each round (1 → NUM_ROUNDS_PER_SIM):
    1. Each agent generates a proposal          ← NEURAL  (GPT)
    2. Constraint checker validates proposal    ← SYMBOLIC (rule engine)
    3. Violations logged; allocation clamped    ← SYMBOLIC
    4. Basin-level feasibility checked          ← SYMBOLIC (aggregate)

Agent order per round:
  States first (7) → Federal → Tribal (2) → International → Agricultural → Environmental
  States go first so their compact positions are visible before
  non-state actors respond — this mirrors real negotiation dynamics.
"""

import json
from agents             import NegotiationAgent, ALL_MANDATES, STATE_AGENTS
from constraint_checker import check_proposal, check_basin_agreement
from config             import NUM_ROUNDS_PER_SIM
from legal_framework    import ALL_AGENT_CEILINGS_MAF, AVAILABLE_SUPPLY_MAF
from conversation_log   import log_round_header, log_agent_turn

# Ordered agent list: states first, then non-state stakeholders
AGENT_ORDER = [
    # Category 1: Basin states (compact actors)
    "California", "Arizona", "Nevada",          # Lower Basin
    "Colorado", "Utah", "Wyoming", "New Mexico", # Upper Basin
    # Category 2: Federal
    "Federal_BOR",
    # Category 3: Tribal
    "Navajo_Nation", "CRIT",
    # Category 4: International
    "Mexico_IBWC",
    # Category 5: Water users / Environmental
    "IID", "Environmental_ESA",
]


def run_simulation(
    simulation_id: int,
    resume_state: dict | None = None,
    on_round_complete=None,
) -> dict:
    """
    Runs one complete Colorado River negotiation simulation across all 13 agents.

    Args:
        simulation_id : Integer identifier for this run (1–NUM_SIMULATIONS).
        resume_state  : Checkpoint state from a partially-completed run of this
                        same simulation_id (see checkpoint.py). Restores agent
                        memory and completed rounds, then continues from the
                        next unfinished round. None = start from round 1.
        on_round_complete : Optional callback invoked after each round completes,
                        as on_round_complete(simulation_id, round_num,
                        simulation_results, agent_states). main.py uses this to
                        write a checkpoint after every round.

    Returns:
        Dict containing rounds, final_proposals, constraint_violations,
        warnings, and basin_feasibility snapshots.
    """
    print(f"\n{'═'*66}")
    print(f"  SIMULATION {simulation_id:>2}  —  Colorado River Negotiation  (13 agents)")
    print(f"{'═'*66}")

    # Initialise all 13 agents fresh — clean history, no carryover between runs
    agents = {name: NegotiationAgent(name) for name in AGENT_ORDER}

    if resume_state:
        # Restore both halves of the state: the results accumulated so far, and
        # each agent's own memory. Restoring results without agent memory would
        # silently produce agents that had "forgotten" the rounds the results
        # say they participated in.
        simulation_results = resume_state["simulation_results"]
        for name, agent_state in resume_state.get("agent_states", {}).items():
            if name in agents:
                agents[name].load_state(agent_state)
        start_round = resume_state["next_round"]
        print(f"  ↻ Resuming from checkpoint — rounds 1–{start_round - 1} already complete.")
    else:
        simulation_results = {
            "simulation_id":         simulation_id,
            "rounds":                [],
            "constraint_violations": [],
            "warnings":              [],
            "basin_feasibility":     [],
            "final_proposals":       {},
        }
        start_round = 1

    # Previous round's basin totals — the live scarcity signal fed into each
    # agent's prompt so they negotiate toward closing the shortfall. On a resumed
    # run, seed it from the last completed round so pressure carries over.
    prev_basin = (
        simulation_results["basin_feasibility"][-1]
        if simulation_results["basin_feasibility"]
        else None
    )

    for round_num in range(start_round, NUM_ROUNDS_PER_SIM + 1):
        label = "(Opening positions)" if round_num == 1 else "(Response & adjustment)"
        print(f"\n  ── Round {round_num} of {NUM_ROUNDS_PER_SIM} {label} ──")

        # Scarcity/deadline context handed to every agent this round.
        basin_context = {
            "supply_maf":       AVAILABLE_SUPPLY_MAF,
            "total_rounds":     NUM_ROUNDS_PER_SIM,
            "last_claimed_maf": prev_basin["grand_total_maf"] if prev_basin else None,
            "last_deficit_maf": (
                round(prev_basin["grand_total_maf"] - AVAILABLE_SUPPLY_MAF, 3)
                if prev_basin else None
            ),
        }

        # Live transcript: banner for this round (see conversation_log.py)
        log_round_header(simulation_id, round_num, label)

        round_proposals  = []
        round_violations = []

        for agent_name, agent in agents.items():

            # Agent type label for console readability
            atype = agent.mandate["agent_type"].upper()[:4]
            print(f"    [{atype}] {agent_name:<20} → generating...", end=" ", flush=True)

            # ── NEURAL: GPT generates the proposal ──────────────────────────
            proposal = agent.generate_proposal(
                round_number    = round_num,
                other_proposals = round_proposals,  # All prior proposals visible
                basin_context   = basin_context,    # Live scarcity + deadline signal
            )

            # Applicable ceiling for this agent
            ceiling = ALL_AGENT_CEILINGS_MAF.get(agent_name, 0.0)

            # ── SYMBOLIC: Constraint checker validates ───────────────────────
            validation = check_proposal(proposal, round_proposals)

            if validation["valid"]:
                alloc = proposal.get("proposed_allocation_maf", 0)
                print(
                    f"✓  {alloc:.3f} MAF  |  "
                    f"conservation: {proposal.get('conservation_commitment_pct', 0):.0f}%  |  "
                    f"shortage: {proposal.get('shortage_sharing_position', 'n/a')}"
                )
            else:
                print(f"✗  VIOLATION — clamping to {ceiling:.3f} MAF")
                for v in validation["violations"]:
                    print(f"           ⚠  {v}")
                    round_violations.append({
                        "agent": agent_name,
                        "round": round_num,
                        "violation": v,
                    })
                # Clamp every out-of-range field before accepting. Allocation is
                # capped at the legal ceiling; conservation is capped to 0–50%
                # (the implementable range enforced in constraint_checker.py).
                original_alloc = float(proposal.get("proposed_allocation_maf", ceiling))
                original_cons  = float(proposal.get("conservation_commitment_pct", 0))
                clamped_cons   = max(0.0, min(original_cons, 50.0))
                proposal["proposed_allocation_maf"]     = min(original_alloc, ceiling)
                proposal["conservation_commitment_pct"] = clamped_cons
                proposal["_constraint_clamped"] = True

                # Build the list of corrections to feed back so the agent
                # self-corrects next round instead of re-anchoring on — and
                # ratcheting up — the same illegal number.
                corrections = []
                if original_alloc > ceiling:
                    corrections.append({
                        "field": "proposed_allocation_maf",
                        "original": original_alloc, "clamped": round(ceiling, 3),
                        "unit": "MAF", "noun": "allocation", "limit_word": "ceiling",
                    })
                if not (0 <= original_cons <= 50):
                    corrections.append({
                        "field": "conservation_commitment_pct",
                        "original": original_cons, "clamped": clamped_cons,
                        "unit": "%", "noun": "conservation commitment", "limit_word": "cap",
                    })
                if corrections:
                    agent.note_constraint_correction(round_num, corrections)

            # Log non-blocking warnings
            for w in validation["warnings"]:
                print(f"           ℹ  {w}")
                simulation_results["warnings"].append({
                    "agent": agent_name, "round": round_num, "warning": w
                })

            # Live transcript: archive this agent's exact generated output plus
            # the constraint verdict, verbatim (see conversation_log.py).
            log_agent_turn(
                simulation_id  = simulation_id,
                round_number   = round_num,
                agent_name     = agent_name,
                agent_type     = agent.mandate["agent_type"],
                raw_generation = agent.last_raw_generation,
                validation     = validation,
            )

            round_proposals.append(proposal)

        # ── SYMBOLIC: Basin-level aggregate feasibility check ─────────────────
        basin_check          = check_basin_agreement(round_proposals)
        basin_check["round"] = round_num
        simulation_results["basin_feasibility"].append(basin_check)
        prev_basin           = basin_check   # feed this round's totals into the next

        feasible_flag = "✓" if basin_check["system_feasible"] else "✗"
        deficit_maf   = round(basin_check["grand_total_maf"] - AVAILABLE_SUPPLY_MAF, 3)
        gap_flag      = "GAP CLOSED ✓" if deficit_maf <= 0.05 else f"shortfall {deficit_maf:+.3f} MAF ✗"
        print(
            f"\n  Basin (compact states only) — "
            f"Upper: {basin_check['upper_basin_total_maf']} MAF  |  "
            f"Lower: {basin_check['lower_basin_total_maf']} MAF  |  "
            f"Grand: {basin_check['grand_total_maf']} MAF  {feasible_flag}"
        )
        print(
            f"  Scarcity — claimed {basin_check['grand_total_maf']} MAF vs "
            f"available {AVAILABLE_SUPPLY_MAF} MAF  →  {gap_flag}"
        )

        simulation_results["rounds"].append({
            "round_number": round_num,
            "proposals":    round_proposals,
            "violations":   round_violations,
            "basin_check":  basin_check,
        })
        simulation_results["constraint_violations"].extend(round_violations)

        # Persist progress. Done at round granularity (not per-agent) because a
        # round is the natural consistent unit — every agent has spoken and the
        # basin check has run, so a resumed run never starts mid-round with a
        # half-populated round_proposals list.
        if on_round_complete:
            on_round_complete(
                simulation_id,
                round_num,
                simulation_results,
                {name: agent.get_state() for name, agent in agents.items()},
            )

    # Final proposals = last round proposals keyed by agent name
    last_round = simulation_results["rounds"][-1]["proposals"]
    simulation_results["final_proposals"] = {p["state"]: p for p in last_round}

    print(f"\n  Final allocations (Round {NUM_ROUNDS_PER_SIM}):")
    for name, proposal in simulation_results["final_proposals"].items():
        atype = ALL_MANDATES[name]["agent_type"].upper()[:4]
        print(
            f"    [{atype}] {name:<20}: "
            f"{proposal.get('proposed_allocation_maf', 0):.3f} MAF"
            f"  — {proposal.get('key_demand', '')[:55]}"
        )

    violations_count = len(simulation_results["constraint_violations"])
    print(f"\n  Simulation {simulation_id} complete. Violations caught: {violations_count}")

    return simulation_results
