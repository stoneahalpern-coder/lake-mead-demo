"""
agents.py — Mulholland Multiagent System
=========================================
THE NEURAL LAYER — All 13 Negotiation Agents

Single source of truth for every agent in the simulation.
Covers all 5 stakeholder categories defined in Appendix B §(b):

  Category 1 — 7 Basin States   : CA, AZ, NV, CO, UT, WY, NM
  Category 2 — Federal Agencies : Bureau of Reclamation / DOI / USFWS
  Category 3 — Tribal Nations   : Navajo Nation, CRIT
  Category 4 — International    : Mexico / IBWC
  Category 5 — Water Users/Env  : IID (agricultural), Environmental/ESA

All 13 agents share one class — NegotiationAgent — which adapts its
system prompt based on each agent's type while sharing identical
proposal-generation mechanics and JSON output schema.

The symbolic constraint checker (constraint_checker.py) runs AFTER every
GPT call. The agents themselves are unaware of this validation step.

    TRIBAL AGENT NOTICE (Appendix B §g, Rule 5e):
    Navajo Nation and CRIT mandates are based on documented legal positions
    and court decisions (Winters v. US, Arizona v. California).
    AI simulation CANNOT capture tribal cultural values, traditional
    ecological knowledge, or community decision-making.
    Direct tribal governmental consultation is required for any real-world use.
"""

import json
import openai
from config          import OPENAI_API_KEY, MODEL, MAX_TOKENS_PER_CALL, HISTORY_WINDOW_ROUNDS
from legal_framework import ALL_AGENT_CEILINGS_MAF

# ── Lazy OpenAI client ────────────────────────────────────────────────────────
_client = None

def _get_openai_client():
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Create a .env file with:  OPENAI_API_KEY=sk-..."
            )
        _client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ── Active client override (set by main.py when using config_open_source.py) ──
# Strictly additive: if nothing ever calls set_active_client(), _get_client()
# and _active_model behave exactly as before (OpenAI, config.MODEL).
_active_client = None
_active_model  = MODEL

def set_active_client(client, model_name: str):
    """Swaps in a different chat.completions-compatible client (e.g. a local
    vLLM server via config_open_source.py) and the model name to request."""
    global _active_client, _active_model
    _active_client, _active_model = client, model_name

def _get_client():
    return _active_client or _get_openai_client()


# ═══════════════════════════════════════════════════════════════════════════════
# ALL_MANDATES — Single registry for all 13 negotiation agents
# ═══════════════════════════════════════════════════════════════════════════════

ALL_MANDATES = {

    # ── CATEGORY 1: SEVEN BASIN STATES ───────────────────────────────────────

    "California": {
        "agent_type":   "state",
        "full_name":    "State of California",
        "priority":     "Senior — Boulder Canyon Project Act 1928",
        "dependencies": (
            "Imperial Irrigation District (~3.1 MAF senior agricultural), "
            "Coachella Valley, Palo Verde IDs, Metropolitan Water District (junior municipal). "
            "~40 million residents depend on Colorado River water."
        ),
        "position": (
            "Protect senior agricultural rights above all else. Willing to discuss "
            "voluntary conservation with compensation but resists mandatory reallocation. "
            "Historically received surplus above 4.4 MAF — treats this as baseline."
        ),
        "negotiation_style": (
            "Assertive and legalistic. Emphasises senior priority, economic harm, "
            "and long-standing infrastructure reliance."
        ),
        "red_lines": [
            "Mandatory cuts to senior agricultural rights require full financial compensation.",
            "Minimum acceptable allocation: 3.8 MAF under any shortage scenario.",
            "IID's senior rights take precedence over MWD's junior rights — internal priority order is inviolable.",
        ],
    },

    "Arizona": {
        "agent_type":   "state",
        "full_name":    "State of Arizona",
        "priority":     "Junior CAP — Colorado River Basin Project Act 1968",
        "dependencies": (
            "Central Arizona Project (CAP) serves Phoenix and Tucson municipal supply. "
            "First-tier shortage exposure. Has already absorbed ~500k AF in Tier 1/2 cuts."
        ),
        "position": (
            "Has borne a disproportionate shortage burden. Seeks equitable burden-sharing "
            "from California. Open to conservation investment if Lower Basin load is redistributed."
        ),
        "negotiation_style": (
            "Pragmatic and compromise-oriented. Frames proposals around fairness. "
            "Natural coalition partner with Nevada to pressure California."
        ),
        "red_lines": [
            "Refuses additional shortage cuts without measurable California contribution.",
            "Minimum allocation to sustain Phoenix and Tucson: 1.8 MAF.",
        ],
    },

    "Nevada": {
        "agent_type":   "state",
        "full_name":    "State of Nevada",
        "priority":     "Junior Lower Basin",
        "dependencies": (
            "Las Vegas Valley Water District — ~90% of supply from Colorado River. "
            "SNWA achieves 99% indoor water recycling. Smallest compact allocation of any state."
        ),
        "position": (
            "Conservation champion — leads all states in efficiency. Advocates for "
            "conservation-first framework. Needs infrastructure investment credit for "
            "past conservation achievements."
        ),
        "negotiation_style": (
            "Moral framing: most efficient, gets least credit. "
            "Uses conservation leadership as bargaining chip. Coalition with Arizona."
        ),
        "red_lines": [
            "Allocation below 0.22 MAF is existential for Las Vegas — absolute floor.",
            "Conservation efficiency credits for already-built infrastructure are non-negotiable.",
        ],
    },

    "Colorado": {
        "agent_type":   "state",
        "full_name":    "State of Colorado",
        "priority":     "Upper Basin Senior — 1948 Compact (51.75% = 3.881 MAF)",
        "dependencies": (
            "Western Slope agricultural economy. Rapidly growing Denver metro. "
            "Has NOT fully developed compact entitlement — reserves it for future growth."
        ),
        "position": (
            "Strongly opposed to permanent demand reductions. "
            "Open to voluntary, compensated, temporary demand management only. "
            "Lee Ferry obligation must be met through system cooperation."
        ),
        "negotiation_style": (
            "Protective of future development rights. Leads Upper Basin coalition. "
            "Emphasises state sovereignty and resists federal mandates."
        ),
        "red_lines": [
            "Will not accept permanent reduction of compact entitlement under any scenario.",
            "Any demand management must be voluntary and financially compensated.",
        ],
    },

    "Utah": {
        "agent_type":   "state",
        "full_name":    "State of Utah",
        "priority":     "Upper Basin — 1948 Compact (23% = 1.725 MAF)",
        "dependencies": (
            "Agricultural communities in Colorado River watershed. "
            "Rapidly growing Wasatch Front (Salt Lake City). "
            "Lake Powell elevation directly affects Utah hydropower and recreation."
        ),
        "position": (
            "Supports Upper Basin solidarity; coordinates with Colorado. "
            "Developing infrastructure to use full compact entitlement."
        ),
        "negotiation_style": (
            "Collaborative within Upper Basin bloc. Seeks collective positions "
            "before engaging Lower Basin. Moderate and measured tone."
        ),
        "red_lines": [
            "Must preserve future development rights for the growing Wasatch Front.",
        ],
    },

    "Wyoming": {
        "agent_type":   "state",
        "full_name":    "State of Wyoming",
        "priority":     "Upper Basin — 1948 Compact (14% = 1.050 MAF)",
        "dependencies": (
            "Agricultural and ranching economy in Green River basin. "
            "Headwaters state — contributes significant natural flow to the system."
        ),
        "position": (
            "Least developed Upper Basin state. Reserves right to fully develop "
            "compact allocation. Deeply opposed to federal override of state water law."
        ),
        "negotiation_style": (
            "Reserved and principled. Firm on property rights and state sovereignty. "
            "Will walk away from agreements setting bad long-term precedents."
        ),
        "red_lines": [
            "No federal mandates without explicit state legislative consent.",
            "Voluntary-only demand management — no coercion.",
        ],
    },

    "New Mexico": {
        "agent_type":   "state",
        "full_name":    "State of New Mexico",
        "priority":     "Upper Basin junior — 1948 Compact (11.25% = 0.844 MAF)",
        "dependencies": (
            "Agricultural communities in San Juan–Chama region. "
            "Navajo Nation tribal water rights: substantial and largely unquantified. "
            "Albuquerque municipal supply via San Juan–Chama Project."
        ),
        "position": (
            "Will not sign any agreement ignoring tribal water rights. "
            "Advocates for Navajo Nation as a full stakeholder with a defined "
            "quantification pathway embedded in any final deal."
        ),
        "negotiation_style": (
            "Equity-focused and tribal-rights advocate. "
            "Uses Winters doctrine and federal trust responsibility as legal leverage. "
            "Will block agreement if tribal interests are sidelined."
        ),
        "red_lines": [
            "Any final agreement must include a defined legal pathway for Navajo Nation water rights quantification.",
            "Will not trade tribal rights for a larger state allocation.",
        ],
    },

    # ── CATEGORY 2: FEDERAL AGENCIES ─────────────────────────────────────────

    "Federal_BOR": {
        "agent_type":  "federal",
        "full_name":   "United States — Bureau of Reclamation / Dept. of Interior / USFWS",
        "authority": (
            "Bureau of Reclamation operates Hoover Dam, Glen Canyon Dam, and all "
            "federal Colorado River infrastructure. Has authority to declare shortages, "
            "set release schedules, and enforce compact compliance. "
            "USFWS enforces ESA critical habitat protections. "
            "DOI holds tribal trust responsibility for all federally recognised tribes."
        ),
        "position": (
            "Must balance all legal obligations simultaneously: compact enforcement, "
            "Mexico treaty delivery (1.5 MAF), ESA compliance, tribal trust, and "
            "infrastructure integrity. Has warned it will impose mandatory cuts if "
            "states cannot reach voluntary agreement. Strongly prefers interstate solution."
        ),
        "negotiation_style": (
            "Neutral arbiter and enforcer. Does not advocate for any single state. "
            "Frames proposals around legal obligation and system-wide sustainability. "
            "Uses threat of federal imposition to concentrate minds."
        ),
        "red_lines": [
            "Lake Powell must stay above 3,490 ft minimum pool — non-negotiable operational floor.",
            "Lake Mead must stay above 895 ft dead pool — below this water cannot be released.",
            "Mexico treaty delivery minimum 1.4 MAF under shortage (Minute 323) must be honored.",
            "ESA-listed species critical habitat flow requirements cannot be eliminated.",
        ],
    },

    # ── CATEGORY 3: TRIBAL NATIONS ────────────────────────────────────────────

    "Navajo_Nation": {
        "agent_type":  "tribal",
        "full_name":   "Navajo Nation",
        "authority": (
            "Largest Native American nation in the US. Reserved water rights under "
            "Winters v. United States (1908) — priority date predates most state "
            "appropriations. Substantial unquantified claims in ongoing litigation "
            "(Arizona v. Navajo Nation, US Supreme Court 2023). "
            "Approximately 30% of Navajo households currently lack running water."
        ),
        "position": (
            "Seeks formal quantification and recognition of reserved rights as a "
            "precondition to any basin-wide agreement. Will not accept any framework "
            "that forecloses unquantified reserved rights claims. "
            "Demands formal tribal consultation pathway in governance structure."
        ),
        "negotiation_style": (
            "Rights-first framing grounded in Winters doctrine and federal trust responsibility. "
            "Moral authority: 30% of households without water is a compelling humanitarian argument. "
            "Coalition with New Mexico and CRIT. Will block agreement if sidelined."
        ),
        "red_lines": [
            "No agreement accepted without a defined legal pathway for Navajo water rights quantification.",
            "Federal trust responsibility must be explicitly acknowledged in any new governance framework.",
            "Will not waive reserved rights claims in exchange for a fixed state-level allocation.",
            "Minimum immediate infrastructure investment required for households currently without water.",
        ],
        "_tribal_disclaimer": (
            "Configured based on documented legal positions and court records "
            "(Arizona v. Navajo Nation, Winters v. US). Cannot capture Navajo cultural, "
            "spiritual, or traditional ecological relationships with water, or community "
            "decision-making processes. Direct tribal governmental consultation is required "
            "before using simulation outputs for any real-world purpose."
        ),
    },

    "CRIT": {
        "agent_type":  "tribal",
        "full_name":   "Colorado River Indian Tribes (CRIT)",
        "authority": (
            "CRIT holds quantified reserved water rights of ~717,148 AF/year with a "
            "priority date of 1865 — among the most senior water rights on the entire river, "
            "predating all state compact allocations. Rights confirmed in Arizona v. California "
            "(1963–2006 decrees). Reservation spans AZ-CA border along the Lower Colorado."
        ),
        "position": (
            "Legally secure and quantified position allows negotiating from strength. "
            "Seeks water marketing opportunities — leasing conserved water to Lower Basin "
            "states for compensation is a key economic development strategy. "
            "Supports shortage-sharing only where CRIT receives fair compensation."
        ),
        "negotiation_style": (
            "Pragmatic and market-oriented. Secured legal position is the foundation. "
            "Interested in water marketing and lease revenue as economic development. "
            "Coalition with Navajo Nation on sovereignty recognition."
        ),
        "red_lines": [
            "Senior priority rights (pre-1865) cannot be subordinated to state compact rights.",
            "Any voluntary reductions must be compensated at fair market rate — no uncompensated cuts.",
            "Tribal sovereignty and government-to-government consultation must be in any new governance framework.",
        ],
        "_tribal_disclaimer": (
            "Configured based on Arizona v. California decree terms and documented CRIT "
            "governmental positions. Direct tribal consultation required for any real-world use."
        ),
    },

    # ── CATEGORY 4: INTERNATIONAL ────────────────────────────────────────────

    "Mexico_IBWC": {
        "agent_type":  "international",
        "full_name":   "United Mexican States — International Boundary and Water Commission (IBWC)",
        "authority": (
            "Mexican Water Treaty of 1944 guarantees Mexico 1.5 MAF/year under normal "
            "conditions, reducible to 1.4 MAF during declared US shortage (Minute 323). "
            "IBWC administers delivery at Morelos Dam near Imperial Dam. "
            "Minute 319 (2012) and Minute 323 (2017) extended cooperative framework — "
            "currently expiring and requiring renewal."
        ),
        "position": (
            "Treaty rights are sovereign obligations under international law — not "
            "subject to US interstate compact politics. Cooperates on binational "
            "conservation where Mexico receives tangible benefit. "
            "Seeks Minute 323 renewal beyond 2026 expiration. "
            "Raises salinity and water quality as linked treaty concerns."
        ),
        "negotiation_style": (
            "Diplomatically firm. Treaty rights as international law obligations. "
            "Willing to engage on cooperative conservation (Minute 319/323 model) "
            "but not on reducing treaty delivery floors. "
            "Will escalate to international arbitration as a last resort."
        ),
        "red_lines": [
            "Treaty delivery of 1.5 MAF under normal conditions is a sovereign right — requires bilateral treaty amendment to change.",
            "Shortage reduction below 1.4 MAF (Minute 323 floor) requires extraordinary justification and bilateral agreement.",
            "Water quality degradation harming Mexicali Valley agriculture constitutes a treaty violation.",
            "Minute 323 cooperative framework must be renewed — Mexico will not operate under an expired agreement.",
        ],
    },

    # ── CATEGORY 5: WATER USERS AND ENVIRONMENTAL INTERESTS ──────────────────

    "IID": {
        "agent_type":  "agricultural",
        "full_name":   "Imperial Irrigation District (IID)",
        "authority": (
            "Most senior water rights in California — ~3.1 MAF/year serving 500,000 "
            "acres of agriculture in Imperial Valley. Rights SENIOR to Metropolitan "
            "Water District and all California municipal users under prior appropriation. "
            "Quantification Settlement Agreement (QSA, 2003) formalised transfers of "
            "up to 200,000 AF/year to San Diego and Coachella in exchange for "
            "infrastructure investment."
        ),
        "position": (
            "Senior agricultural rights are the cornerstone of California's Colorado "
            "River position. Will defend senior priority aggressively. Open to "
            "additional compensated transfers only if paired with full infrastructure "
            "funding and voluntary farmer fallowing payments."
        ),
        "negotiation_style": (
            "Assertive defense of prior appropriation doctrine. Sharply distinguishes "
            "itself from MWD (junior). Junior users cut first, always. "
            "Economic argument: Imperial Valley agriculture generates ~$2B/year."
        ),
        "red_lines": [
            "Senior priority rights cannot be cut before junior municipal users (MWD) — prior appropriation is absolute.",
            "No involuntary fallowing — any land fallowing must be voluntary and compensated at fair market value.",
            "Infrastructure investment must be secured before IID accepts any new transfer obligations.",
        ],
    },

    "Environmental_ESA": {
        "agent_type":  "environmental",
        "full_name":   "Environmental Coalition — ESA / Ecosystem Flows",
        "authority": (
            "Endangered Species Act requirements for listed species: razorback sucker, "
            "bonytail chub, humpback chub, southwestern willow flycatcher, Yuma Ridgway's rail. "
            "Grand Canyon Adaptive Management Program (high-flow experiments, temperature control). "
            "Colorado River Basin Salinity Control Act water quality standards. "
            "NEPA environmental review requirements for any operational or infrastructure changes."
        ),
        "position": (
            "Environmental flows are ESA legal obligations — not optional amenities. "
            "Any agreement eliminating critical habitat flows triggers ESA Section 7 "
            "consultation and biological opinions that could block implementation for years. "
            "Advocates for minimum ecological flow baseline in new operating guidelines. "
            "Supports Colorado River Delta restoration (Minute 319 pulse flows)."
        ),
        "negotiation_style": (
            "Science-first. Cites peer-reviewed flow requirements and ESA mandates. "
            "Uses litigation risk as leverage: missing ecological flows = years of delay. "
            "Coalition with Mexico on Delta restoration. Accepts phased implementation "
            "if ecological floor guarantees are locked in upfront."
        ),
        "red_lines": [
            "Glen Canyon Dam temperature control device operations for native fish are a non-negotiable ESA obligation.",
            "Minimum 50,000 AF/year ecological flow to Colorado River Delta must be embedded in any agreement.",
            "Lake Powell must not drop below 3,525 ft (minimum power pool) — dam stability and ESA flows depend on this.",
            "Any agreement must include an ESA Section 7 consultation pathway to prevent litigation blocking implementation.",
        ],
    },
}


# ── Convenience groupings ─────────────────────────────────────────────────────

STATE_AGENTS          = [k for k, v in ALL_MANDATES.items() if v["agent_type"] == "state"]
TRIBAL_AGENTS         = [k for k, v in ALL_MANDATES.items() if v["agent_type"] == "tribal"]
CONSUMPTIVE_AGENTS    = [k for k in ALL_MANDATES if ALL_AGENT_CEILINGS_MAF.get(k, 0) > 0]
NON_CONSUMPTIVE_AGENTS= [k for k in ALL_MANDATES if ALL_AGENT_CEILINGS_MAF.get(k, 0) == 0]


# ═══════════════════════════════════════════════════════════════════════════════
# NegotiationAgent — Single class for all 13 agents
# ═══════════════════════════════════════════════════════════════════════════════

class NegotiationAgent:
    """
    Represents any of the 13 stakeholders in the Colorado River negotiation.

    NEURAL component: GPT generates proposals in structured JSON.
    SYMBOLIC component: constraint_checker.py validates them externally.

    System prompt adapts by agent_type; proposal generation is identical
    for all 13 agents — same GPT call, same JSON schema, same history pattern.
    """

    def __init__(self, agent_name: str):
        if agent_name not in ALL_MANDATES:
            raise ValueError(
                f"Unknown agent: '{agent_name}'. "
                f"Available: {list(ALL_MANDATES.keys())}"
            )
        self.name       = agent_name
        self.mandate    = ALL_MANDATES[agent_name]
        self.agent_type = self.mandate["agent_type"]
        self.ceiling    = ALL_AGENT_CEILINGS_MAF.get(agent_name, 0.0)

        self.conversation_history: list[dict] = []
        self.proposal_history:     list[dict] = []
        # Exact text of this agent's most recent model generation, kept so the
        # conversation logger (conversation_log.py) can archive verbatim output.
        self.last_raw_generation:  str = ""
        # A correction note set by the moderator when this agent's last proposal
        # was clamped (e.g. it claimed above its legal ceiling). Injected into
        # the NEXT round's prompt so the agent self-corrects instead of
        # re-anchoring on the rejected number. Consumed once, then cleared.
        self.pending_feedback:     str = ""

    def _build_system_prompt(self) -> str:
        """Constructs the GPT system prompt, adapting the identity framing by agent_type."""
        m             = self.mandate
        red_lines_str = "\n  ".join(f"• {r}" for r in m["red_lines"])

        # ── Identity framing ──────────────────────────────────────────────────
        type_intros = {
            "state":         f"You are the lead water-rights negotiator for the US state of {self.name}, representing your state in a high-stakes Colorado River Basin water allocation negotiation.",
            "federal":       f"You are the senior representative of {m['full_name']}, serving as neutral arbiter and enforcer — not an advocate for any single state.",
            "tribal":        f"You are the lead water-rights negotiator representing {m['full_name']}, advocating for tribal sovereignty, reserved water rights, and the economic and cultural needs of your nation.",
            "international": f"You are the senior representative of {m['full_name']}, representing Mexico's treaty rights under international law — not US compact law.",
            "agricultural":  f"You are the lead negotiator for {m['full_name']}, representing the most senior agricultural water rights in California.",
            "environmental": f"You are the lead representative of {m['full_name']}, advocating for legally mandated environmental flows, endangered species protections, and ecosystem health.",
        }
        identity = type_intros.get(self.agent_type, f"You are a negotiator for {m['full_name']}.")

        # ── Tribal disclaimer (mandatory per Appendix B §g.iv) ───────────────
        tribal_block = ""
        if "_tribal_disclaimer" in m:
            tribal_block = (
                f"\n⚠️  MANDATORY AI LIMITATION NOTICE (Appendix B §g, Rule 5e):\n"
                f"{m['_tribal_disclaimer']}\n"
                f"Acknowledge this if asked about cultural/spiritual aspects of water. "
                f"Never fabricate traditional ecological knowledge.\n"
            )

        # ── Authority field — state agents use priority, others use authority ─
        authority_text = m.get("authority", m.get("priority", ""))
        context_line   = f"Key dependencies : {m['dependencies']}\n" if "dependencies" in m else ""

        # ── Allocation ceiling instruction ────────────────────────────────────
        if self.ceiling > 0:
            ceiling_instruction = f"Maximum allocation you may claim: {self.ceiling:.3f} MAF. Never exceed this."
        else:
            ceiling_instruction = (
                "You are a non-consumptive actor. Set proposed_allocation_maf to 0.0. "
                "Your influence is through legal mandates and operational constraints, not water claims."
            )

        return f"""{identity}
{tribal_block}
─── YOUR MANDATE ─────────────────────────────────────────────────
Authority  : {authority_text}
{context_line}Position   : {m['position']}
Style      : {m['negotiation_style']}
{ceiling_instruction}

YOUR RED LINES (non-negotiable — never concede these):
  {red_lines_str}

─── YOUR OBJECTIVE ───────────────────────────────────────────────
The river is over-allocated: the parties' combined claims exceed the water
that physically exists. You have TWO jobs at once, and you fail if you drop
either one:
  (1) Defend the red lines above — never cross them.
  (2) Help the group reach a basin-wide agreement that closes the shortfall.
Your opening allocation, conservation level, and shortage-sharing stance are
NEGOTIABLE POSITIONS, not red lines. You are expected to ADJUST them across
the rounds as others move — trade lower-priority positions to protect your
red lines. Restating the same numbers round after round is itself a failure
of your mandate: deadlock forces uncontrolled shortage and litigation, which
leaves your own constituents worse off than a negotiated cut. When another
party makes a real concession, acknowledge it and reciprocate. (If you are a
non-consumptive actor, you "move" by adjusting your conservation and
shortage-sharing stance and offering operational flexibility — never by
claiming water.) Do all of this while staying fully in character.

─── RULES ────────────────────────────────────────────────────────
1. Stay fully in character as {m['full_name']} at all times.
2. Base all claims on real water law, treaty text, or court decisions.
3. Never fabricate legal citations or quantitative data.
4. Non-consumptive actors: keep proposed_allocation_maf at 0.0.
5. Treat your opening numbers as a starting position: move incrementally
   toward feasibility each round when others reciprocate — but never past a
   red line. If the shortfall shrinks, lock in that progress; don't re-open it.

─── RESPONSE FORMAT ──────────────────────────────────────────────
Respond ONLY with a valid JSON object — no preamble, no markdown:

{{
  "state": "{self.name}",
  "proposed_allocation_maf": <float>,
  "conservation_commitment_pct": <float 0–50>,
  "shortage_sharing_position": "<refuse|reluctant|moderate|willing|champion>",
  "key_demand": "<one sentence>",
  "concession_offered": "<one sentence>",
  "rationale": "<2–3 sentences grounded in legal/factual reality>"
}}"""

    def _trim_history(self) -> None:
        """
        Slides a fixed-width window over conversation_history: the agent always
        sees its HISTORY_WINDOW_ROUNDS most recent *past* rounds, plus the round
        it's answering right now. The window is relative to the current round,
        never anchored to specific ones — at HISTORY_WINDOW_ROUNDS=3:

            generating round  6  ->  sees rounds  3, 4, 5
            generating round  7  ->  sees rounds  4, 5, 6
            generating round 42  ->  sees rounds 39, 40, 41
            generating round 100 ->  sees rounds 97, 98, 99

        Why bound it at all: every round appends a (user, assistant) pair, and
        from round 2 on each user message embeds all 12 other agents' proposals
        — roughly 1.2k tokens per round. Left to grow, the prompt crosses the
        server's --max-model-len (8192) at ~round 6-7 and every later call fails
        with a 400 "maximum context length exceeded". With NUM_ROUNDS_PER_SIM at
        100, no context size fixes this on its own; a bigger window only moves
        the wall. A sliding window makes cost per round constant, so round 100
        is exactly as affordable as round 6.

        What is NOT lost: every other agent's *current* proposals are rebuilt
        into each round's live user message (sent for that call only, never
        stored), so the agent always sees the full live negotiation state. Only
        a compact record of its OWN past rounds lives in history — the bulky
        12-peer table is never replayed, which is what keeps cost per round flat.

        Each stored round is a (memory-note user, compact-reply assistant) pair
        = 2 messages. We keep the most recent HISTORY_WINDOW_ROUNDS pairs, then
        ensure the window opens on a `user` message so the chat API sees strict
        user/assistant alternation after the system prompt.
        """
        max_messages = max(2, HISTORY_WINDOW_ROUNDS * 2)
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]
        # Guarantee the window opens on a user turn (a leading assistant message
        # right after the system prompt is invalid/odd for the chat template).
        if self.conversation_history and self.conversation_history[0].get("role") != "user":
            self.conversation_history = self.conversation_history[1:]

    def get_state(self) -> dict:
        """Serialises this agent's memory for checkpointing (see checkpoint.py)."""
        return {
            "conversation_history": self.conversation_history,
            "proposal_history":     self.proposal_history,
        }

    def load_state(self, state: dict) -> None:
        """Restores memory saved by get_state(), so a resumed agent keeps its context."""
        self.conversation_history = state.get("conversation_history", [])
        self.proposal_history     = state.get("proposal_history", [])

    @staticmethod
    def _clip(text, limit: int = 160) -> str:
        """
        Caps a free-text field so the running prompt can't grow unbounded. The
        sliding history window bounds the NUMBER of remembered rounds, but each
        round embeds 12 peers' text fields (and stores this agent's own reply);
        as agents negotiate, that prose grows and eventually blows past the
        server's --max-model-len. Clipping each field keeps per-round cost flat.
        """
        s = "" if text is None else str(text)
        return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"

    def _build_pressure_text(self, round_number: int, basin_context: dict | None) -> str:
        """
        Turns the live basin numbers into an in-prompt scarcity + deadline signal.
        This is the forcing function that makes agents actually converge: they can
        see the shortfall they must collectively close, and the round it's due.
        Returns "" when no context is available (keeps behavior safe/optional).
        """
        if not basin_context:
            return ""

        supply       = basin_context.get("supply_maf")
        total_rounds = basin_context.get("total_rounds")
        claimed      = basin_context.get("last_claimed_maf")
        deficit      = basin_context.get("last_deficit_maf")

        # Round 1 (or no prior totals yet): frame the scarcity up front.
        if claimed is None or deficit is None:
            if supply is None:
                return ""
            return (
                f"\n\nSCARCITY CONTEXT — The seven basin states can realistically expect "
                f"only about {supply:.2f} MAF this year, far below the 15.0 MAF paper "
                f"compact total. A durable agreement will require shared cuts, so state "
                f"an opening position you can actually move from."
            )

        rounds_left = None
        if total_rounds is not None:
            rounds_left = max(0, total_rounds - round_number)

        deadline = f"This is round {round_number}"
        if total_rounds is not None:
            deadline += f" of {total_rounds}"
            if rounds_left is not None:
                deadline += f" — {rounds_left} round(s) left to close the gap"
        deadline += "."

        if deficit > 0.05:
            return (
                f"\nSCARCITY CHECK — Last round the seven basin states together claimed "
                f"{claimed:.2f} MAF, but only about {supply:.2f} MAF is physically "
                f"available: a shortfall of {deficit:.2f} MAF the group MUST close. "
                f"{deadline} If the gap is not closed by the final round, every party "
                f"defaults to uncontrolled shortage and litigation — worse for you than "
                f"a negotiated cut. Move a negotiable position toward closing it this "
                f"round without crossing your red lines."
            )
        # Gap effectively closed — hold the feasible deal, don't re-open it.
        return (
            f"\nSCARCITY CHECK — Last round total state claims were {claimed:.2f} MAF "
            f"against {supply:.2f} MAF available: the shortfall is essentially closed. "
            f"{deadline} Lock in this progress and hold a feasible agreement rather than "
            f"re-opening demands."
        )

    def _exhaustion_note(self) -> str:
        """
        If this agent has already exhausted its concession levers (it is sitting at
        the 50% conservation cap), tell it to HOLD. Otherwise the convergence
        pressure keeps nudging it to conserve *more*, it re-reads its own capped
        value and bumps it past 50%, and the checker clamps it again — a runaway
        ratchet (55→60→65…) with no brake. This stops the overshoot at the source.
        """
        if not self.proposal_history:
            return ""
        prev      = self.proposal_history[-1]["proposal"]
        prev_cons = float(prev.get("conservation_commitment_pct", 0) or 0)
        if prev_cons >= 49.5:
            return (
                "\nLIMIT REACHED — You are already at the maximum implementable "
                "conservation of 50%. Do NOT raise conservation above 50% (it will be "
                "rejected). You have exhausted this lever; hold your position and rely "
                "on reciprocal moves from others rather than offering more yourself."
            )
        return ""

    def _compute_conviction(self, allocation: float, conservation: float) -> float:
        """
        Derived 0–1 rigidity/conviction signal, computed from OBSERVABLE behaviour
        only — never self-reported by the model (LLM self-confidence is uncalibrated).

          high  → sitting at a hard limit and not moving (a wall)
          low   → actively flexing round-over-round (a dealmaker)

        Ingredients: how much of its legal entitlement it is still claiming, and how
        little it moved since last round (allocation + conservation deltas).
        """
        conservation = max(0.0, min(float(conservation), 50.0))
        prev = self.proposal_history[-1]["proposal"] if self.proposal_history else None

        if prev is not None:
            prev_alloc = float(prev.get("proposed_allocation_maf", allocation))
            prev_cons  = max(0.0, min(float(prev.get("conservation_commitment_pct", conservation) or 0), 50.0))
            alloc_delta = abs(allocation - prev_alloc) / self.ceiling if self.ceiling > 0 else 0.0
            cons_delta  = abs(conservation - prev_cons) / 50.0
            stillness   = 1.0 - min(alloc_delta + cons_delta, 1.0)
        else:
            stillness = 0.5   # round 1 — no prior move to measure; neutral

        if self.ceiling > 0:
            holding    = min(allocation / self.ceiling, 1.0)   # share of entitlement still claimed
            conviction = 0.5 * holding + 0.5 * stillness
        else:
            # Non-consumptive actors (BOR, Environmental) hold via posture, not water.
            conviction = stillness

        return round(max(0.0, min(conviction, 1.0)), 3)

    def generate_proposal(
        self,
        round_number: int,
        other_proposals: list[dict],
        basin_context: dict | None = None,
    ) -> dict:
        """
        Calls GPT to generate a negotiation proposal for this round.

        Round 1 : Opening position — no prior context.
        Round 2+: Sees all other proposals and responds/adjusts.

        basin_context (optional) carries the live scarcity signal so agents
        actually negotiate instead of restating positions:
            supply_maf       — water realistically available to the states
            total_rounds     — deadline (agreement must close by this round)
            last_claimed_maf — previous round's summed state claims (None in R1)
            last_deficit_maf — previous round's shortfall vs supply (None in R1)
        """
        pressure_text = self._build_pressure_text(round_number, basin_context)
        pressure_text += self._exhaustion_note()

        # A one-time correction if the moderator clamped our last proposal.
        feedback = self.pending_feedback
        feedback_block = f"{feedback}\n\n" if feedback else ""

        if round_number == 1:
            live_message = (
                f"{feedback_block}"
                f"Round {round_number} — Opening positions.\n"
                f"State your opening position as {self.mandate['full_name']}. "
                "What is your primary demand? What will you fight hardest to protect?"
                + pressure_text
            )
        else:
            other_text = json.dumps(
                [
                    {
                        "stakeholder":                p.get("state"),
                        "proposed_allocation_maf":    p.get("proposed_allocation_maf"),
                        "conservation_commitment_pct": p.get("conservation_commitment_pct"),
                        "shortage_sharing_position":  p.get("shortage_sharing_position"),
                        "key_demand":                 self._clip(p.get("key_demand")),
                        "concession_offered":         self._clip(p.get("concession_offered")),
                    }
                    for p in other_proposals
                ],
                indent=2,
            )
            live_message = (
                f"{feedback_block}"
                f"Round {round_number} — Responding to all stakeholders.\n\n"
                f"All proposals on the table:\n{other_text}\n"
                f"{pressure_text}\n"
                "How do you respond? Which positions align with yours? Which threaten "
                "your red lines? What position can you MOVE this round to help close the "
                "shortfall — and what do you ask for in return?"
            )

        # Two separate channels, deliberately kept apart so context can't grow:
        #   • live_message  — the FULL current table + pressure. Sent for THIS
        #     call only; never stored, so the bulky 12-peer table is not replayed
        #     every future round (that was blowing past --max-model-len).
        #   • memory_note   — a tiny placeholder stored in history to preserve
        #     user/assistant alternation. The agent's own trajectory is carried
        #     by its compact assistant replies; the current state of everyone
        #     else is always re-injected fresh via live_message.
        memory_note = f"[Round {round_number}] Reviewed the table and responded."

        # ── NEURAL: GPT generates the proposal ───────────────────────────────
        response = _get_client().chat.completions.create(
            model=_active_model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                *self.conversation_history,
                {"role": "user", "content": live_message},
            ],
            max_tokens=MAX_TOKENS_PER_CALL,
            temperature=0.7,
            response_format={"type": "json_object"},
            **({"extra_body": {"reasoning_effort": "low"}} if "gpt-oss" in _active_model else {}),  # gpt-oss reasons; low effort leaves room for JSON within max_tokens
        )

        raw_text = response.choices[0].message.content
        self.last_raw_generation = raw_text or ""

        try:
            proposal = json.loads(raw_text)
        except json.JSONDecodeError:
            print(f"  [WARNING] JSON parse error for {self.name} — using safe fallback.")
            proposal = {
                "state":                       self.name,
                "proposed_allocation_maf":     self.ceiling,
                "conservation_commitment_pct": 10,
                "shortage_sharing_position":   "moderate",
                "key_demand":                  f"Protect {self.mandate['full_name']} legal authority.",
                "concession_offered":          "Open to collaborative framework discussions.",
                "rationale":                   "Safe fallback due to JSON parse error.",
                "_parse_error":                True,
            }

        proposal["state"] = self.name   # Enforce correct identifier

        # Derived conviction (0–1) from observable behaviour — attached to the
        # proposal so the analysis layer can aggregate it across runs into the
        # rigidity map / confidence bands (see zopa_analyzer.py). Computed BEFORE
        # this round is appended to proposal_history, so it compares against the
        # previous round's values.
        proposal["conviction"] = self._compute_conviction(
            float(proposal.get("proposed_allocation_maf", 0) or 0),
            float(proposal.get("conservation_commitment_pct", 0) or 0),
        )

        # Store a COMPACT reply in the LLM context (drop the verbose rationale,
        # clip text fields) so memory stays bounded round-over-round. The full
        # proposal is still kept in proposal_history (checkpoint/results) and the
        # full raw text in last_raw_generation (verbatim transcript log), so
        # nothing is lost from the outputs — only the replayed context shrinks.
        compact_reply = json.dumps({
            "proposed_allocation_maf":     proposal.get("proposed_allocation_maf"),
            "conservation_commitment_pct": proposal.get("conservation_commitment_pct"),
            "shortage_sharing_position":   proposal.get("shortage_sharing_position"),
            "key_demand":                  self._clip(proposal.get("key_demand")),
            "concession_offered":          self._clip(proposal.get("concession_offered")),
            "conviction":                  proposal.get("conviction"),
        })
        self.conversation_history.append({"role": "user",      "content": memory_note})
        self.conversation_history.append({"role": "assistant", "content": compact_reply})
        self._trim_history()

        # Correction (if any) has now been delivered — clear it so it fires once.
        self.pending_feedback = ""

        self.proposal_history.append({"round": round_number, "proposal": proposal})
        return proposal

    def note_constraint_correction(self, round_number: int, corrections: list[dict]) -> None:
        """
        Called by the moderator when one or more of this agent's proposal fields
        were clamped for breaching a hard limit (allocation ceiling, conservation
        cap, …). Does two things per field so the agent SELF-CORRECTS next round
        instead of re-anchoring on — and ratcheting up — the rejected number:
          1. Queues a plain correction note for the next prompt.
          2. Rewrites its most recent stored reply to the clamped (legal) values,
             so its own memory shows the corrected numbers, not the illegal ones.

        Each entry in `corrections` is a dict with:
          field, original, clamped, unit ("MAF"|"%"), noun, limit_word.
        """
        if not corrections:
            return

        lines = []
        for c in corrections:
            lines.append(
                f"Your round {round_number} {c['noun']} of "
                f"{float(c['original']):.3f}{c['unit']} breached your legal "
                f"{c['limit_word']} of {float(c['clamped']):.3f}{c['unit']} and was "
                f"automatically clamped to {float(c['clamped']):.3f}{c['unit']}. "
                f"Do not exceed {float(c['clamped']):.3f}{c['unit']} again."
            )
        self.pending_feedback = "CORRECTION — " + "  ".join(lines)

        # Rewrite the most recent stored assistant reply so memory shows the
        # legal values (all clamped fields at once, in a single scan).
        for msg in reversed(self.conversation_history):
            if msg.get("role") == "assistant":
                try:
                    d = json.loads(msg["content"])
                    for c in corrections:
                        if c["field"] in d:
                            d[c["field"]] = round(float(c["clamped"]), 3)
                    msg["content"] = json.dumps(d)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                break

    def reset(self):
        """Clears conversation and proposal history between simulation runs."""
        self.conversation_history = []
        self.proposal_history     = []
