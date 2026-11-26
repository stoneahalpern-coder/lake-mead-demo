# Live run page: how it is built and how to deploy (NOT deployed yet)

What it is: live_server.py serves one page. A visitor picks an assumed supply and rounds and clicks Run. The real 13-agent engine
runs on the server, calls a hosted model, and the page shows each agent's proposal as rounds finish. Reply check and invalid-run warning included.

Status: code written and tested with a FAKE model only (test_live_server.py, 7 tests). No real model, account, host or key has been used.

Guards (server side): key only in the server environment; prompts only to an allow-listed host (api.groq.com); one run at a time;
rounds capped; supply from a fixed list; hard daily request budget counted on every real request; optional access code; pacing between requests.

To deploy (needs Stone's own accounts; not done):
1. Groq account and API key (check the FREE plan's real limits in the account's limits page first; the public rate-limit page shows Developer plan base limits).
2. A GitHub repo with this project (public or private; do not commit .env or a key). Pushing is a publication step.
3. Render free web service from render.yaml. Set HOSTED_API_KEY and LIVE_ACCESS_CODE in the Render dashboard.
4. Free instances sleep after 15 minutes idle and take about a minute to wake. Local files are lost on restart, so results are shown on the page, not stored.
5. Run a 1 round smoke run (13 requests) and read the reply check before showing anyone.
Run locally instead (any computer with Python): set HOSTED_API_KEY, HOSTED_MODEL, run `python live_server.py`, open http://localhost:8080.

## Limits you must know before exposing it
- Request and token counts are kept in memory (and in BUDGET_FILE if set). On a free Render instance the disk is ephemeral, so a restart, redeploy or wake-from-sleep RESETS the count. This is NOT a durable daily cap.
- Token counts use the provider-reported total when the reply includes one, else a chars/4 estimate. The up-front refusal uses a rough 3000 tokens per request guess. No free capacity number is assumed.
- Before any exposed deployment: (1) the server refuses to start without LIVE_ACCESS_CODE; share the code only with intended people; (2) set the durable limit at the provider: use the provider's free plan with no payment method or billing, or its own spend and usage limits, and check in the account that paid usage is off; (3) check the account's real rate limits and set DAILY_REQUEST_CAP, DAILY_TOKEN_CAP and PACE_SECONDS below them.
- Access code is a shared secret sent in the page request body over HTTPS. It is not per-user authentication and has no lockout.
