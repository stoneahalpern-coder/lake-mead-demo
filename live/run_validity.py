"""
run_validity.py - counts replies the simulator had to replace or repair. Free, local, no model calls.

The negotiation code does not retry a bad model reply. If a reply is not valid JSON it substitutes a "safe fallback"
proposal (allocation = the agent's ceiling). If a reply is valid JSON but has no allocation, the code silently uses the
ceiling. Both look like real proposals in the output, so this module counts them.
"""

def _proposals(results):
    for sim in results.get("simulations", []):
        for rnd in sim.get("rounds", []):
            for p in rnd.get("proposals", []):
                yield sim.get("simulation_id"), rnd.get("round_number"), p

def validity(results):
    total = parse_err = no_alloc = clamped = 0
    by_agent = {}
    for sim, rnd, p in _proposals(results):
        total += 1
        bad = False
        if p.get("_parse_error"):
            parse_err += 1; bad = True
        elif "proposed_allocation_maf" not in p or p.get("proposed_allocation_maf") is None:
            no_alloc += 1; bad = True
        if p.get("_constraint_clamped"):
            clamped += 1
        if bad:
            by_agent[p.get("state", "?")] = by_agent.get(p.get("state", "?"), 0) + 1
    replaced = parse_err + no_alloc
    return {"replies": total, "parse_error_fallbacks": parse_err, "missing_allocation": no_alloc,
            "replaced_or_defaulted": replaced, "clamped_to_ceiling": clamped, "by_agent": by_agent,
            "valid": total > 0 and replaced == 0}

def warning_text(v):
    if v["replies"] == 0:
        return "INVALID RUN: no agent replies were found in the results."
    if v["replaced_or_defaulted"] == 0:
        return None
    return (f"INVALID RUN WARNING: {v['replaced_or_defaulted']} of {v['replies']} agent replies were not usable "
            f"({v['parse_error_fallbacks']} not valid JSON and replaced by a ceiling-value fallback, "
            f"{v['missing_allocation']} valid JSON with no allocation, silently defaulted to the ceiling). "
            f"Agents affected: {v['by_agent']}. The numbers for those agents are not model output. Do not use this run's results as a negotiation.")
