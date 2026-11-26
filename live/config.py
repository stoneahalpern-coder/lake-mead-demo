"""
config.py — Mulholland Multiagent System
=========================================
Central configuration. All tunable parameters live here so you never
need to hunt through other files to change a setting.
"""

import os
from dotenv import load_dotenv

# Load OPENAI_API_KEY from the .env file in the project root
load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# gpt-4o-mini is used here to keep API costs low during development.
# Swap to "gpt-4o" for higher-quality negotiations when ready.
MODEL = "gpt-4o-mini"

# Max tokens the LLM can use per agent response.
# 500 is enough for a well-reasoned JSON proposal without burning tokens.
MAX_TOKENS_PER_CALL = 500

# ── Simulation ────────────────────────────────────────────────────────────────
# Total LLM calls per run = NUM_SIMULATIONS × NUM_ROUNDS_PER_SIM × 13 agents.
# The defaults below (1 × 1 × 13 = 13 calls) are a quick smoke test. For real
# analysis use several rounds (agents only start conceding from round 2) and
# 20+ simulations (confidence bands in zopa_analyzer.py need ≥20 runs).
NUM_SIMULATIONS = 1

# Number of back-and-forth negotiation rounds per simulation.
# Round 1 = opening positions; round 2+ = responses and adjustments.
NUM_ROUNDS_PER_SIM = 1

# How many of its own most recent PAST rounds each agent carries into the
# round it's generating. This is a sliding window relative to the current
# round, not a fixed set of rounds:
#
#     generating round  6  ->  sees rounds  3, 4, 5
#     generating round  7  ->  sees rounds  4, 5, 6
#     generating round 100 ->  sees rounds 97, 98, 99
#
# Why it's bounded: each agent's history grows ~1.2k tokens per round (every
# round-2+ user message embeds all 12 other agents' proposals). Unbounded, it
# blows past the local vLLM server's context window (--max-model-len, 8192)
# at ~round 6-7 and every later call fails with a 400. With
# NUM_ROUNDS_PER_SIM=100 a bigger context only moves the wall — the window
# keeps cost per round constant, so round 100 is as affordable as round 6.
#
# This bounds only an agent's memory of its OWN older rounds. Every other
# agent's current proposals are rebuilt into each round's user message, so
# agents always see the full live negotiation state regardless of this value.
#
# 3 past rounds ≈ 5.3k prompt tokens — inside 8192 with room for the
# 500-token response. Raise it only alongside MAX_MODEL_LEN in
# config_open_source.py (which costs GPU memory for KV cache).
HISTORY_WINDOW_ROUNDS = 3

# ── Output ────────────────────────────────────────────────────────────────────
# Where result JSON / DOCX / transcripts are written (created automatically).
OUTPUT_DIR = "output"

# Note: the water supply the agents negotiate against is AVAILABLE_SUPPLY_MAF in
# legal_framework.py (13.0 MAF by default). run_scenario.py overrides it with
# the scenario file's supply_conditions.available_supply_maf.
