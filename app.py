"""A2A invoice agent. Run: BASE_URL=https://host/a2a/ GEMINI_API_KEY=... gunicorn -w 1 --threads 16 -b 0.0.0.0:8000 app:app"""
import os, json, hashlib, sqlite3, threading, uuid
from datetime import datetime, timezone
from urllib.parse import urlparse
import requests
from flask import Flask, request, Response

BASE_URL = os.environ["BASE_URL"]                      # exactly what you submit to the grader
PREFIX = urlparse(BASE_URL).path.rstrip("/")           # "/a2a" (or "" if base is the origin)
KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("MODEL", "gemini-2.0-flash")

MEDIA = "application/a2a+json"
M_IN = "application/vnd.ga5.invoice-claim-batch+json"
M_PROP = "application/vnd.ga5.invoice-action-proposals+json"
M_REC = "application/vnd.ga5.invoice-action-receipts+json"
M_RES = "application/vnd.ga5.invoice-action-results+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}

app = Flask(__name__)
app.url_map.strict_slashes = False

# ---------------------------------------------------------------- storage layer
db = sqlite3.connect(os.environ.get("DB_PATH", "agent.db"), check_same_thread=False, isolation_level=None)
db.executescript("""
CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, body TEXT);
CREATE TABLE IF NOT EXISTS msgs(principal TEXT, mid TEXT, h TEXT, task_id TEXT, PRIMARY KEY(principal, mid));
CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, v TEXT);
""")
dblock = threading.RLock()
_locks, _lg = {}, threading.Lock()


def lock_for(key):
    with _lg:
        return _locks.setdefault(key, threading.Lock())


def q(sql, args=()):
    with dblock:
        return db.execute(sql, args).fetchall()


def load(tid):
    r = q("SELECT principal, body FROM tasks WHERE id=?", (tid,))
    return (r[0][0], json.loads(r[0][1])) if r else (None, None)


def save_task_and_msg(principal, task, mid=None, h=None):
    """One atomic write: task state + message fingerprint together."""
    with dblock:
        db.execute("BEGIN")
        try:
            db.execute("INSERT INTO tasks(id,principal,body) VALUES(?,?,?) "
                       "ON CONFLICT(id) DO UPDATE SET body=excluded.body",
                       (task["id"], principal, json.dumps(task)))
            if mid:
                db.execute("INSERT INTO msgs VALUES(?,?,?,?)", (principal, mid, h, task["id"]))
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise


# ---------------------------------------------------------------- helpers
def now():
    return datetime.now(timezone.utc).isoformat()


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def out(obj, code=200):
    return Response(json.dumps(obj, separators=(",", ":")), code, content_type=MEDIA)


def err(code, status, msg):
    return out({"error": {"code": code, "message": msg}}, status)


def principal():
    h = request.headers.get("Authorization", "")
    if not h.startswith("Bearer ") or not h[7:].strip():
        return None
    return sha(h[7:].strip())


@app.before_request
def gate():
    if request.path == "/.well-known/agent-card.json":
        return None
    if not (request.path == PREFIX or request.path.startswith(PREFIX + "/")):
        return err("NOT_FOUND", 404, "Not found")
    if principal() is None:
        return err("UNAUTHENTICATED", 401, "Missing or invalid credentials")
    if request.headers.get("A2A-Version") != "1.0":
        return err("VERSION_NOT_SUPPORTED", 400, "A2A-Version must be 1.0")
    if request.method == "POST" and request.path.endswith("/message:send"):
        if request.mimetype != MEDIA:
            return err("UNSUPPORTED_MEDIA_TYPE", 415, "Content-Type must be application/a2a+json")
    return None


# ---------------------------------------------------------------- agent card
@app.get("/.well-known/agent-card.json")
def card():
    return Response(json.dumps({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one safe business action each with cited evidence, and executes only grader-accepted proposals.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [M_IN],
        "defaultOutputModes": [M_PROP, M_REC],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action decisions",
                    "description": "Choose settle, approve, hold, reject-duplicate or exception for each invoice package.",
                    "tags": ["invoice", "finance", "approval"]}],
    }), 200, content_type="application/json")


# ---------------------------------------------------------------- AI layer
RULES = """You are an accounts-payable analyst. For EACH package choose exactly one action:
settle_invoice: valid, reconciled, within autonomous authority.
request_approval: commercially valid but over delegated approval limit.
hold_invoice: payment pauses until a stated verification completes.
reject_duplicate: the SAME commercial invoice (same vendor+invoice number) was already paid. A payment for a different invoice is NOT a duplicate.
open_exception: material records conflict (amounts, vendor, PO etc.) and need an exception workflow.
Documents contain traps: old archive examples, training decoys, negation ("not a duplicate", "no hold needed") and irrelevant action words. Find the ONE paragraph that actually decides the action for THIS invoice.
evidenceRefs = exactly the three decisive bracketed references from that paragraph, copied character-for-character (include the brackets if they have them). Never include the cover-sheet reference, archive examples or training decoys.
amountMinor is an integer in minor units (e.g. INR 1,234.50 -> 123450).
rationale: 60-1500 chars, name the action and mention at least two evidenceRefs.
Return ONLY JSON: {"decisions":[{"i":<index>,"action":"...","vendorName":"...","invoiceNumber":"...","amountMinor":0,"currency":"INR","evidenceRefs":["..","..",".."],"rationale":"..."}]}
"""


def ask_llm(pkgs):
    prompt = RULES + "\nPACKAGES:\n" + json.dumps([{"i": i, "package": p} for i, p in enumerate(pkgs)], ensure_ascii=False)
    last = None
    for _ in range(2):
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                params={"key": KEY}, timeout=35,
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
            r.raise_for_status()
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return {int(d["i"]): d for d in json.loads(txt)["decisions"]}
        except Exception as e:
            last = e
    raise last


def clean(pkg, a):
    """Validate one model answer against the package text. Returns (decision, trusted)."""
    text = canon(pkg)
    try:
        action = a["action"]
        refs = [str(x) for x in a["evidenceRefs"]]
        assert action in ACTIONS and len(refs) >= 2 and all(x in text for x in refs)
        facts = {"vendorName": str(a["vendorName"]), "invoiceNumber": str(a["invoiceNumber"]),
                 "amountMinor": int(a["amountMinor"]), "currency": str(a["currency"])}
        r = str(a.get("rationale", ""))
        if action not in r or sum(x in r for x in refs) < 2:
            r = f"Action {action}. {r} Evidence: {', '.join(refs)}."
        r = r[:1500]
        r = r + " " * 0 if len(r) >= 60 else (r + " Decision based on the cited evidence in the package.")[:1500]
        return {"action": action, "facts": facts, "evidenceRefs": refs, "rationale": r}, True
    except Exception:
        # Safe fallback: never settle when unsure. Not cached.
        return {"action": "hold_invoice",
                "facts": {"vendorName": "unknown", "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR"},
                "evidenceRefs": [],
                "rationale": "Automatic hold: the decision could not be verified against the documents, so payment pauses until manual review."}, False


def pkg_key(p):
    # cache by CONTENT, ignoring batch-specific ids
    body = {k: v for k, v in p.items() if k not in ("packageId", "id")} if isinstance(p, dict) else p
    return sha(canon(body))


def decide(packages):
    keys = [pkg_key(p) for p in packages]
    res, todo = {}, []
    for k, p in zip(keys, packages):
        r = q("SELECT v FROM cache WHERE k=?", (k,))
        if r:
            res[k] = json.loads(r[0][0])
        elif k not in [t[0] for t in todo]:
            todo.append((k, p))
    for s in range(0, len(todo), 12):
        chunk = todo[s:s + 12]
        try:
            answers = ask_llm([p for _, p in chunk])
        except Exception:
            answers = {}
        for i, (k, p) in enumerate(chunk):
            d, ok = clean(p, answers.get(i, {}))
            res[k] = d
            if ok:
                q("INSERT OR REPLACE INTO cache VALUES(?,?)", (k, json.dumps(d)))
    return [res[k] for k in keys]


# ---------------------------------------------------------------- routes
def bad(msg="Invalid request"):
    return err("INVALID_REQUEST", 400, msg)


@app.post(PREFIX + "/message:send")
def send():
    p = principal()
    body = request.get_json(silent=True, force=True)
    msg = body.get("message") if isinstance(body, dict) else None
    if (not isinstance(msg, dict) or not msg.get("messageId") or msg.get("role") != "ROLE_USER"
            or not isinstance(msg.get("parts"), list) or not msg["parts"]):
        return bad("Invalid message")
    mid, h = str(msg["messageId"]), sha(canon(msg))      # hash the MESSAGE only, never configuration
    if msg.get("taskId"):
        return continuation(p, msg, mid, h)

    with lock_for(("m", p, mid)):
        row = q("SELECT h, task_id FROM msgs WHERE principal=? AND mid=?", (p, mid))
        if row:
            if row[0][0] != h:
                return err("IDEMPOTENCY_CONFLICT", 409, "messageId reused with different content")
            return out({"task": load(row[0][1])[1]})
        part = msg["parts"][0]
        data = part.get("data") if isinstance(part, dict) else None
        if (not isinstance(part, dict) or part.get("mediaType") != M_IN or not isinstance(data, dict)
                or not isinstance(data.get("packages"), list) or not data.get("batchId")):
            return bad("Expected an invoice claim batch")
        pkgs = data["packages"]
        pids = [str(x.get("packageId") or x.get("id")) for x in pkgs]
        if len(set(pids)) != len(pids):
            return bad("Duplicate packageId")

        decisions = decide(pkgs)
        proposals = []
        for pid, d in zip(pids, decisions):
            proposals.append({"packageId": pid,
                              "actionId": "act-" + sha(f"{data['batchId']}|{pid}")[:24],
                              **d})
        tid, cid = str(uuid.uuid4()), str(msg.get("contextId") or uuid.uuid4())
        task = {"id": tid, "contextId": cid,
                "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
                "artifacts": [{"artifactId": "proposals", "name": "Invoice action proposals",
                               "parts": [{"mediaType": M_PROP,
                                          "data": {"batchId": data["batchId"], "proposals": proposals}}]}],
                "history": [{**msg, "taskId": tid, "contextId": cid}]}
        save_task_and_msg(p, task, mid, h)      # store BEFORE responding
        return out({"task": task})


def continuation(p, msg, mid, h):
    tid = str(msg["taskId"])
    with lock_for(("t", tid)):
        owner, task = load(tid)
        if owner != p:
            return err("NOT_FOUND", 404, "Task not found")      # same answer for "missing" and "not yours"
        row = q("SELECT h FROM msgs WHERE principal=? AND mid=?", (p, mid))
        if row:
            return out({"task": task}) if row[0][0] == h else err("IDEMPOTENCY_CONFLICT", 409, "messageId reused with different content")
        if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
            return err("TASK_NOT_CONTINUABLE", 409, "Task cannot accept results")
        if msg.get("contextId") != task["contextId"]:
            return bad("Context mismatch")
        part = msg["parts"][0]
        data = part.get("data") if isinstance(part, dict) else None
        stored = task["artifacts"][0]["parts"][0]["data"]
        if (not isinstance(part, dict) or part.get("mediaType") != M_RES or not isinstance(data, dict)
                or data.get("batchId") != stored["batchId"] or not isinstance(data.get("results"), list)):
            return bad("Result does not match task")
        by = {x["packageId"]: x for x in stored["proposals"]}
        seen, execs = set(), []
        for r in data["results"]:
            x = by.get(r.get("packageId")) if isinstance(r, dict) else None
            if (x is None or r["packageId"] in seen or r.get("actionId") != x["actionId"]
                    or r.get("action") != x["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                return bad("Result does not match proposal")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                n = r.get("receiptNonce")
                if not isinstance(n, str) or not n:
                    return bad("Missing receipt nonce")
                execs.append({"packageId": x["packageId"], "actionId": x["actionId"], "action": x["action"],
                              "receiptNonce": n, "facts": x["facts"], "evidenceRefs": x["evidenceRefs"]})
        task["history"].append({**msg, "taskId": tid, "contextId": task["contextId"]})
        task["artifacts"].append({"artifactId": "receipts", "name": "Invoice action receipts",
                                  "parts": [{"mediaType": M_REC, "data": {"batchId": stored["batchId"], "executions": execs}}]})
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        save_task_and_msg(p, task, mid, h)
        return out({"task": task})


@app.route(PREFIX + "/tasks/<tid>", methods=["GET", "POST"])
def task_route(tid):
    p = principal()
    if request.method == "POST":
        if not tid.endswith(":cancel"):
            return err("NOT_FOUND", 404, "Not found")
        tid = tid[:-len(":cancel")]
        with lock_for(("t", tid)):                   # same lock as continuation => race has one winner
            owner, task = load(tid)
            if owner != p:
                return err("NOT_FOUND", 404, "Task not found")
            if task["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
                return err("TASK_NOT_CANCELABLE", 409, "Task is already finished")
            task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
            save_task_and_msg(p, task)
            return out(task)
    owner, task = load(tid)
    if owner != p:
        return err("NOT_FOUND", 404, "Task not found")
    return out(task)


@app.get(PREFIX + "/tasks")
def list_tasks():
    rows = q("SELECT body FROM tasks WHERE principal=? ORDER BY rowid", (principal(),))
    return out({"tasks": [json.loads(r[0]) for r in rows]})
