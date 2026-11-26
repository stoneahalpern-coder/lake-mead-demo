"""
conversation_log.py — Mulholland Multiagent System
===================================================
LIVE TRANSCRIPT — every agent's raw generated output, every round, every run.

Writes output/all_conversations.txt as the simulation runs: for each agent, at
each round, of each simulation, the exact text the model generated — its stated
demand, concession, rationale, and shortage stance, verbatim. Read it after a
run to understand *why* coalitions formed, why proposals were rejected, or why
agents converged — the raw narrative behind the numeric ZOPA analysis.

Design decisions:
  • APPEND-ONLY. Every `python main.py` run adds a new, clearly delimited RUN
    section rather than overwriting — so the file is a running archive across
    all runs, exactly as requested ("for every run that we do").
  • WRITTEN PER TURN (open → append → close each call), so a crash, SLURM time
    limit, or node preemption never loses the transcript up to that point — the
    same durability philosophy as checkpoint.py. On resume, only the rounds that
    actually re-run get re-logged (already-completed rounds are skipped upstream).
  • NO-OP UNTIL INITIALISED. If init_conversation_log() was never called (e.g. a
    unit test importing negotiation.py directly), every log_* function silently
    does nothing — negotiation.py can call them unconditionally.

The record here is the raw model text. The structured/parsed proposals still
live in the results JSON; this file is the human-readable counterpart.
"""

import os
from datetime import datetime

# Single fixed file — one growing archive, not a per-run timestamped file.
DEFAULT_PATH = os.path.join("output", "all_conversations.txt")

_log_path = None


def init_conversation_log(path: str = DEFAULT_PATH, metadata: dict | None = None) -> None:
    """
    Enables logging and writes a RUN header. Call once at the start of a run
    (main.py). Uses append mode, so prior runs' transcripts are preserved.

    Args:
        path     : Output file. Defaults to output/all_conversations.txt.
        metadata : Optional {label: value} pairs (model, num simulations, ...)
                   written into the run header for later reference.
    """
    global _log_path
    _log_path = path

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    lines = [
        "",
        "=" * 80,
        f"RUN STARTED  {datetime.now().isoformat(timespec='seconds')}",
    ]
    if metadata:
        lines.append("  |  ".join(f"{k}: {v}" for k, v in metadata.items()))
    lines += ["=" * 80, ""]
    _append("\n".join(lines) + "\n")


def log_round_header(simulation_id: int, round_number: int, label: str = "") -> None:
    """Writes a banner marking the start of one round of one simulation."""
    if _log_path is None:
        return
    banner = f" SIMULATION {simulation_id}  |  ROUND {round_number} {label} ".strip()
    _append("\n" + "-" * 80 + "\n" + banner + "\n" + "-" * 80 + "\n")


def log_agent_turn(
    simulation_id: int,
    round_number: int,
    agent_name: str,
    agent_type: str,
    raw_generation: str,
    validation: dict | None = None,
) -> None:
    """
    Appends one agent's raw generated output for this round, followed by the
    symbolic constraint layer's verdict (so you can see, in place, whether that
    proposal was accepted as-is or clamped for crossing a legal ceiling).

    Args:
        raw_generation : The exact text the model produced (agent.last_raw_generation).
        validation     : The dict from constraint_checker.check_proposal(), or None.
    """
    if _log_path is None:
        return

    atype = (agent_type or "").upper()
    out = f"\n---- [{atype}] {agent_name} ----\n"
    out += (raw_generation or "").strip() + "\n"

    if validation is not None:
        if validation.get("valid", True):
            out += "[constraint layer: OK — accepted as proposed]\n"
        else:
            viols = "; ".join(validation.get("violations", [])) or "constraint violation"
            out += f"[constraint layer: CLAMPED to legal ceiling — {viols}]\n"
        for w in validation.get("warnings", []):
            out += f"[constraint layer warning: {w}]\n"

    _append(out)


def _append(text: str) -> None:
    """Open-append-close so partial progress survives a crash mid-run."""
    with open(_log_path, "a", encoding="utf-8") as f:
        f.write(text)
