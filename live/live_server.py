"""
live_server.py - ADDITIVE. Small web server so a visitor can click "Run" and watch the 13-agent negotiation.
Standard library only (plus the `openai` package the project already uses). Does not edit any original file.

Safety design (all enforced server side):
  * The model key lives ONLY in the server environment (HOSTED_API_KEY). It is never sent to a browser or written to a file.
  * Prompts go ONLY to an allow-listed host (default api.groq.com). Any other host makes the server refuse to start.
  * One run at a time. Rounds capped (MAX_ROUNDS, default 2). Supply must be one of a fixed list.
  * A hard request budget per day (DAILY_REQUEST_CAP). The counter wraps the client, so it counts real requests exactly.
  * Optional access code (LIVE_ACCESS_CODE) so only people you give it to can start a run.
  * Pacing between requests (PACE_SECONDS) to stay under a provider's tokens-per-minute limit.
  * Output is labelled: AI agents under an assumed supply, not a forecast; replaced replies are counted and flagged.
Env: HOSTED_API_KEY, HOSTED_MODEL, [HOSTED_BASE_URL], [LIVE_ACCESS_CODE], [DAILY_REQUEST_CAP], [MAX_ROUNDS], [PACE_SECONDS], [PORT]
"""
import hmac, json, os, threading, time, datetime, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ALLOWED_HOSTS = {"api.groq.com"}
SUPPLIES = [9.0, 11.0, 13.0]          # assumed supply choices (scenario assumptions, not sourced)

class BudgetExceeded(Exception): pass

class Budget:
    """Counts requests AND tokens per UTC-free local day. Optionally saved to BUDGET_FILE so a restart on a host with a
    persistent disk keeps the count. On a free host with an ephemeral disk a restart or redeploy RESETS it: this is a
    best-effort guard, not a durable cap. The durable cap must be set at the provider (see DEPLOY_LIVE.md)."""
    def __init__(self, cap, token_cap=None, path=None):
        self.cap, self.token_cap, self.path = cap, token_cap, path
        self.used, self.tokens, self.day, self.lock = 0, 0, datetime.date.today().isoformat(), threading.Lock()
        if path and os.path.exists(path):
            try:
                d = json.load(open(path))
                if d.get("day") == self.day: self.used, self.tokens = int(d["used"]), int(d["tokens"])
            except Exception: pass
    def _roll(self):
        today = datetime.date.today().isoformat()
        if today != self.day: self.day, self.used, self.tokens = today, 0, 0
    def _save(self):
        if self.path:
            try: json.dump({"day": self.day, "used": self.used, "tokens": self.tokens}, open(self.path, "w"))
            except Exception: pass
    def take(self, est_tokens=0):
        with self.lock:
            self._roll()
            if self.used >= self.cap:
                raise BudgetExceeded(f"daily request budget of {self.cap} is used up")
            if self.token_cap is not None and self.tokens + est_tokens > self.token_cap:
                raise BudgetExceeded(f"daily token budget of {self.token_cap} would be exceeded")
            self.used += 1; self.tokens += est_tokens; self._save()
    def add_actual(self, actual, est):
        """Replace the pre-call estimate with the provider-reported total when the reply carries one."""
        with self.lock:
            self.tokens += max(0, actual - est); self._save()
    def left(self):
        with self.lock:
            self._roll(); return self.cap - self.used
    def tokens_left(self):
        with self.lock:
            self._roll(); return None if self.token_cap is None else self.token_cap - self.tokens

# Token-window guard. The free plan allows 8000 tokens per rolling minute (provider
# x-ratelimit-limit-tokens header, verified 2026-11-26). Fixed spacing alone cannot bound a
# minute window: two calls 50s apart can land in the same window. Before each call we wait
# until the ACTUAL provider-counted tokens of every call in the last TPM_WINDOW_SECONDS
# (from response.usage), plus this call's conservative estimate, fit in TPM_WINDOW_BUDGET
# (75% of the provider cap). Any two calls within 60s of each other share this window, so
# it bounds any provider window of <=60s regardless of phase. The estimate uses a real
# tokenizer (tiktoken o200k, the gpt-oss base encoding) plus per-message overhead and the
# full configured completion allowance. A single request whose estimate alone exceeds the
# budget is rejected visibly (never waited out or silently replaced), as is a window that
# stays full past TPM_MAX_WAIT_SECONDS.
TPM_WINDOW_SECONDS = 60
TPM_WINDOW_BUDGET = 6000
TPM_MAX_WAIT_SECONDS = int(os.environ.get("TPM_MAX_WAIT_SECONDS", "180"))

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")   # base encoding of gpt-oss (o200k_harmony)
except Exception:   # Render installs tiktoken; this fallback is for local dev only
    _ENC = None

class WindowBudgetExceeded(RuntimeError):
    """Surfaces as a visibly incomplete reply, never a silent replacement."""

def _estimate_call_tokens(messages, max_tokens):
    if _ENC is not None:
        n = 2
        for m in messages:
            n += 4
            for key in ("role", "name", "content"):
                n += len(_ENC.encode(str(m.get(key) or "")))
    else:
        n = 2 + sum((len(str(m.get("content") or "")) + 16) // 3 + 4 for m in messages)
    return n + int(max_tokens)

class _Completions:
    def __init__(self, real, budget, pace, state):
        self.real, self.budget, self.pace, self.state, self._last = real, budget, pace, state, 0.0
        self._win = []   # (timestamp, actual provider-counted tokens) of recent calls
    def create(self, **kw):
        if self.state.get("cancel"): raise RuntimeError("run cancelled")
        est = _estimate_call_tokens(kw.get("messages", []), kw.get("max_tokens") or 500)
        if est > TPM_WINDOW_BUDGET:
            raise WindowBudgetExceeded(
                f"single request estimate {est} tokens exceeds window budget {TPM_WINDOW_BUDGET}")
        waited = 0.0
        while True:
            now = time.time()
            self._win = [(t, e) for (t, e) in self._win if now - t < TPM_WINDOW_SECONDS]
            if sum(e for _, e in self._win) + est <= TPM_WINDOW_BUDGET:
                break
            sleep_for = max(self._win[0][0] + TPM_WINDOW_SECONDS - now, 1)
            if waited + sleep_for > TPM_MAX_WAIT_SECONDS:
                raise WindowBudgetExceeded(
                    f"token window still full after {waited:.0f}s (used {sum(e for _, e in self._win)}, need {est})")
            print(f"  [pace] token window full; waiting {sleep_for:.0f}s", flush=True)
            time.sleep(sleep_for)
            waited += sleep_for
        wait = self._last + self.pace - time.time()
        if wait > 0: time.sleep(wait)
        self.budget.take(est)
        self._last = time.time()
        self.state["requests"] = self.state.get("requests", 0) + 1
        resp = self.real.chat.completions.create(**kw)
        used = getattr(getattr(resp, "usage", None), "total_tokens", None)
        if isinstance(used, int): self.budget.add_actual(used, est)
        self._win.append((time.time(), used if isinstance(used, int) else est))
        print(f"  [usage] request {self.state.get('requests')}: {used if isinstance(used, int) else '?'} tokens actual (est {est})", flush=True)
        return resp

class GuardedClient:
    def __init__(self, real, budget, pace, state):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(real, budget, pace, state)

def make_real_client(base_url, key):
    host = (urlparse(base_url).hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise SystemExit(f"Refusing to start: model host '{host}' is not in the allow-list {sorted(ALLOWED_HOSTS)}.")
    if not key: raise SystemExit("HOSTED_API_KEY is not set.")
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=key, max_retries=2, timeout=60)

class App:
    def __init__(self, client, model, access_code=None, cap=200, max_rounds=2, pace=20.0, token_cap=None, budget_path=None, est_tokens_per_request=3000):
        self.client, self.model, self.code = client, model, access_code
        self.budget, self.max_rounds, self.pace = Budget(cap, token_cap, budget_path), max_rounds, pace
        self.est_tpr = est_tokens_per_request   # rough guess used only to refuse a run up front
        self.lock = threading.Lock()
        self.state = {"status": "idle"}

    def start(self, code, rounds, supply):
        if self.code and not hmac.compare_digest(str(code or ''), self.code): return 403, {"error": "access code needed"}
        try: rounds = int(rounds); supply = float(supply)
        except Exception: return 400, {"error": "bad input"}
        if not (1 <= rounds <= self.max_rounds) or supply not in SUPPLIES:
            return 400, {"error": f"rounds 1 to {self.max_rounds}; supply one of {SUPPLIES}"}
        need = rounds * 13
        if self.budget.left() < need: return 429, {"error": f"not enough daily budget left ({self.budget.left()} requests, need {need})"}
        tl = self.budget.tokens_left()
        if tl is not None and tl < need * self.est_tpr:
            return 429, {"error": f"not enough daily token budget left (about {tl}; a run of {need} requests is roughly {need*self.est_tpr}, an estimate)"}
        if not self.lock.acquire(blocking=False): return 409, {"error": "a run is already in progress"}
        self.state = {"status": "running", "rounds_done": 0, "rounds_total": rounds, "supply": supply,
                      "started": time.time(), "requests": 0, "proposals": []}
        threading.Thread(target=self._run, args=(rounds, supply, self.state), daemon=True).start()
        return 200, {"ok": True}

    def _run(self, rounds, supply, state):
        try:
            import agents, negotiation, run_validity
            negotiation.NUM_ROUNDS_PER_SIM = rounds
            negotiation.AVAILABLE_SUPPLY_MAF = supply
            agents.set_active_client(GuardedClient(self.client, self.budget, self.pace, state), self.model)
            def cb(sim_id, rnd, sim_results, agent_states):
                state["rounds_done"] = rnd
                last = sim_results["rounds"][-1] if sim_results.get("rounds") else {}
                state["proposals"] = [{"round": rnd, "agent": p.get("state"),
                    "alloc": p.get("proposed_allocation_maf"), "conservation_pct": p.get("conservation_commitment_pct"),
                    "stance": p.get("shortage_sharing_position"), "replaced": bool(p.get("_parse_error")),
                    "clamped": bool(p.get("_constraint_clamped"))} for p in last.get("proposals", [])]
            result = negotiation.run_simulation(simulation_id=1, on_round_complete=cb)
            v = run_validity.validity({"simulations": [result]})
            state.update(status="done", validity=v, warning=run_validity.warning_text(v),
                         final={k: p.get("proposed_allocation_maf") for k, p in result.get("final_proposals", {}).items()})
        except BudgetExceeded as e:
            state.update(status="stopped", error=str(e))
        except Exception as e:
            state.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
        finally:
            self.lock.release()

PAGE = """<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'><title>Mulholland live run</title>
<style>body{font:16px/1.5 system-ui;max-width:640px;margin:1rem auto;padding:0 1rem}button,select,input{font:inherit;padding:.5rem}
table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{border-bottom:1px solid #ddd;padding:.3rem;text-align:left;font-size:13px;word-wrap:break-word;overflow-wrap:anywhere}th:nth-child(1){width:27%}th:nth-child(2){width:11%}th:nth-child(3){width:11%}th:nth-child(4){width:20%}.w{background:#fff3cd;padding:.7rem}</style>
<h1>Mulholland live run</h1>
<p class=w>13 AI agents negotiate under an <b>assumed</b> water supply. This shows how the AI agents behave under that assumption. It is not a forecast and not what real parties would do. Agent limits are code values, not verified legal limits.</p>
<p>Supply (million acre-feet, assumed): <select id=s><option>9.0<option selected>11.0<option>13.0</select>
Rounds: <select id=r><option>1<option selected>2</select>
<input id=c placeholder="access code" type=password> <button onclick=go()>Run</button></p>
<p id=m>Idle.</p><p id=e role=alert style='color:#b00020;font-weight:bold'></p><div id=o></div><p style='font-size:13px'>Keep this tab open while a run is going: the free host sleeps after 15 minutes without visits, which would stop the run early.</p><p style='font-size:13px'>Flags: <b>replaced</b> = the model reply was unusable and the code substituted a value, so it is not model output. <b>capped</b> = the ask was above the agent's code ceiling and was cut to it.</p>
<script>
async function go(){let b=JSON.stringify({code:c.value,rounds:r.value,supply:s.value});
 e.textContent='';
 try{let x=await fetch('/run',{method:'POST',body:b});let j=await x.json().catch(()=>({}));
 if(!x.ok){e.textContent='Not started: '+(j.error||('server returned '+x.status));return}
 m.textContent='Started. One round takes about 10 to 15 minutes on free-plan pacing.';poll()}
 catch(err){e.textContent='Not started: network error - please try again.'}}
async function poll(){let j=await (await fetch('/status')).json();
 m.textContent=j.status+(j.rounds_total?` - round ${j.rounds_done} of ${j.rounds_total}, ${j.requests} requests`:'')+(j.error?' - '+j.error:'');
 let h='';if(j.status=='stopped')h+='<p class=w>Run stopped early. Only completed rounds are shown and this is not a full run.</p>';if(j.validity)h+=`<p>Reply check: ${j.validity.replies} replies, ${j.validity.replaced_or_defaulted} replaced, ${j.validity.clamped_to_ceiling} capped to code ceiling.</p>`;if(j.warning)h+='<p class=w>'+j.warning+'</p>';
 if(j.proposals&&j.proposals.length){h+='<table><tr><th>Agent<th>MAF<th>Cons %<th>Stance<th>Flags</tr>';
  for(const p of j.proposals)h+=`<tr><td>${p.agent}<td>${p.replaced?'n/a':p.alloc}<td>${p.replaced?'n/a':p.conservation_pct}<td>${p.replaced?'n/a':p.stance}<td>${p.replaced?'replaced':''}${p.clamped?' capped':''}</tr>`;h+='</table>'}
 o.innerHTML=h;if(j.status=='running')setTimeout(poll,5000)}
poll()
</script>"""

def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path == "/": self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/status":
                s = dict(app.state); s["budget_left"] = app.budget.left(); s["tokens_left"] = app.budget.tokens_left(); self._send(200, s)
            elif self.path == "/health": self._send(200, {"ok": True})
            else: self._send(404, {"error": "not found"})
        def do_POST(self):
            if self.path != "/run": return self._send(404, {"error": "not found"})
            n = min(int(self.headers.get("Content-Length") or 0), 2000)
            try: d = json.loads(self.rfile.read(n) or b"{}")
            except Exception: return self._send(400, {"error": "bad json"})
            code, body = app.start(d.get("code"), d.get("rounds"), d.get("supply"))
            self._send(code, body)
    return H

def main():
    if not os.getenv("LIVE_ACCESS_CODE") and os.getenv("ALLOW_NO_CODE") != "1":
        raise SystemExit("Refusing to start without LIVE_ACCESS_CODE. (Set ALLOW_NO_CODE=1 only for a private local run.)")
    base = os.getenv("HOSTED_BASE_URL", "https://api.groq.com/openai/v1")
    client = make_real_client(base, os.getenv("HOSTED_API_KEY"))
    model = os.getenv("HOSTED_MODEL")
    if not model: raise SystemExit("HOSTED_MODEL is not set.")
    app = App(client, model, os.getenv("LIVE_ACCESS_CODE"), int(os.getenv("DAILY_REQUEST_CAP", "200")),
              int(os.getenv("MAX_ROUNDS", "2")), float(os.getenv("PACE_SECONDS", "20")),
              int(os.environ["DAILY_TOKEN_CAP"]) if os.getenv("DAILY_TOKEN_CAP") else None, os.getenv("BUDGET_FILE"))
    ThreadingHTTPServer(("0.0.0.0", int(os.getenv("PORT", "8080"))), make_handler(app)).serve_forever()

if __name__ == "__main__":
    main()
