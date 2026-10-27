#!/usr/bin/env python3
"""
run_benchmark.py: run the five systems on every test query with one or more LLMs.

Systems
  simple          the message alone, no instructions
  strong          a carefully written empathy system prompt
  oneshot_random  strong-style one-shot with a RANDOM case from the bank
  oneshot_top     the same one-shot prompt with the TOP-1 retrieved case
  framework       the single-call method (profile -> match -> branches -> score -> fuse)

Steps (retrieval is done ONCE, so every LLM sees the same cases)
  1. prepare  python run_benchmark.py prepare --cases kb_all.jsonl --queries queries.jsonl --bench bench/
  2. run      python run_benchmark.py run --cases kb_all.jsonl --queries queries.jsonl --bench bench/ \
                  --model-config models.json --models qwen_local gpt4o --limit 3        # trial first
  3. export   python run_benchmark.py export --queries queries.jsonl --bench bench/

`run` is resumable: finished (model, query, system) triples are skipped, failed ones are retried
next time you run the same command. Needs empathy_rag.py and single_call_prompt.txt in the same
folder. Python standard library only.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import empathy_rag as er  # noqa: E402

SYSTEMS = ["simple", "strong", "oneshot_random", "oneshot_top", "framework"]

STRONG_SYSTEM = """You are a warm, attentive listener. A person is sharing something personal with you. Reply in 3-6 sentences of natural prose.
- First reflect what they said and the feeling behind it in your own words, using specifics from their message (including any physical reaction they mention).
- Match what they need. If they are venting or want to be heard, do not offer advice or solutions. If they ask for advice or help deciding, acknowledge the feeling briefly, then give concrete, practical guidance (at most two suggestions) without deciding for them.
- Do not assume facts they did not state.
- Avoid clichés such as "I understand how you feel", "everything happens for a reason" and "at least...". No lists, headings or bold text.
- Do not suggest therapy unless they ask or there are signs of serious risk.
- Where it fits, end by gently inviting them to say more."""

ONESHOT_SYSTEM = """You are a warm, attentive listener. You will see ONE example of expert guidance on how to respond empathetically in a situation, followed by a different person's message. Learn the STRATEGY from the example and apply it to the new person.

Rules:
- Transfer the principle, not the story. Do not reuse the example's details, names or wording, and never mention the example, experts or books.
- Respond only to what this person actually said. Do not assume facts.
- Do what the expert recommends and nothing on the avoid line.
- Write 3-6 natural sentences. No bullet points, no lists of options."""

ONESHOT_USER = """EXPERT EXAMPLE
Situation: {situation}
The person's need in the example: {need}
What the expert recommends: {recommended}
What to avoid: {avoid}
Guiding principle: {principle}

NEW PERSON
Message: {message}

Write the response to the new person."""


# ---------------------------------------------------------------- helpers
def load_queries(path, split="all", limit=None):
    qs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if split != "all":
        qs = [q for q in qs if q.get("split", "test") == split]
    return qs[:limit] if limit else qs


def kb_md5(path, tiers):
    h = hashlib.md5(Path(path).read_bytes())
    h.update(",".join(sorted(tiers)).encode())
    return h.hexdigest()


def parse_json(t):
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (t or "").strip())
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(s[i: j + 1])
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------- LLM backends
def http_post(url, headers, body, timeout, retries=5):
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and attempt < retries - 1:
                try:
                    wait = float(e.headers.get("Retry-After") or 0)
                except ValueError:
                    wait = 0
                time.sleep(wait or min(60, 2 ** (attempt + 1)))
                continue
            raise RuntimeError(f"HTTP {e.code} from {url}: {detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries - 1:
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            raise RuntimeError(f"network error calling {url}: {e}")


def api_key(cfg):
    env = cfg.get("api_key_env")
    key = os.environ.get(env, "") if env else ""
    if not key:
        raise RuntimeError(f"environment variable {env!r} is not set")
    return key


def call_llm(cfg, system, user, json_mode, max_tokens, timeout):
    """Return (text, tokens_in, tokens_out). temperature=null in the config omits it."""
    b = cfg["backend"]
    temp = cfg.get("temperature", 0.0)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    if b == "ollama":
        opts = {"num_ctx": cfg.get("num_ctx", 16384), "num_predict": max_tokens}
        if temp is not None:
            opts["temperature"] = temp
        body = {"model": cfg["model"], "stream": False, "messages": msgs, "options": opts}
        if json_mode:
            body["format"] = "json"
        r = http_post(cfg.get("host", "http://localhost:11434").rstrip("/") + "/api/chat", {}, body, timeout)
        return r["message"]["content"], r.get("prompt_eval_count"), r.get("eval_count")
    if b in ("openai", "mistral", "openai_compat"):
        default = {"openai": "https://api.openai.com/v1", "mistral": "https://api.mistral.ai/v1"}.get(b)
        base = cfg.get("base_url", default)
        if not base:
            raise RuntimeError("openai_compat backend needs base_url")
        body = {"model": cfg["model"], "messages": msgs}
        body["max_completion_tokens" if b == "openai" else "max_tokens"] = max_tokens
        if temp is not None:
            body["temperature"] = temp
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = http_post(base.rstrip("/") + "/chat/completions",
                      {"Authorization": "Bearer " + api_key(cfg)}, body, timeout)
        u = r.get("usage") or {}
        return r["choices"][0]["message"]["content"], u.get("prompt_tokens"), u.get("completion_tokens")
    if b == "anthropic":
        body = {"model": cfg["model"], "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": user}]}
        if system:
            body["system"] = system
        if temp is not None:
            body["temperature"] = temp
        r = http_post(cfg.get("base_url", "https://api.anthropic.com").rstrip("/") + "/v1/messages",
                      {"x-api-key": api_key(cfg), "anthropic-version": "2023-06-01"}, body, timeout)
        text = "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
        u = r.get("usage") or {}
        return text, u.get("input_tokens"), u.get("output_tokens")
    if b == "google":
        gen = {"maxOutputTokens": max_tokens}
        if temp is not None:
            gen["temperature"] = temp
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {"contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        base = cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
        r = http_post(f"{base.rstrip('/')}/models/{cfg['model']}:generateContent",
                      {"x-goog-api-key": api_key(cfg)}, body, timeout)
        cands = r.get("candidates") or []
        if not cands:
            raise RuntimeError("no candidates returned (blocked?): " + json.dumps(r)[:300])
        text = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", []))
        u = r.get("usageMetadata") or {}
        return text, u.get("promptTokenCount"), u.get("candidatesTokenCount")
    raise RuntimeError(f"unknown backend {b!r}")


# ---------------------------------------------------------------- one job = one (query, system)
def one_shot_user(case, message):
    return ONESHOT_USER.format(
        situation=case["situation_summary"],
        need=", ".join(case.get("underlying_need") or []) or "unclear",
        recommended=case["recommended_response"],
        avoid=case.get("avoid") or "nothing stated",
        principle=case.get("principle") or "",
        message=message)


def do_job(cfg, q, system, ctx):
    t0 = time.time()
    tok = [0, 0]
    mt_text = cfg.get("max_tokens_text", 1500)
    mt_json = cfg.get("max_tokens_json", 4096)

    def gen(sys_p, user, json_mode=False):
        text, ti, to = call_llm(cfg, sys_p, user, json_mode, mt_json if json_mode else mt_text, ctx["timeout"])
        tok[0] += ti or 0
        tok[1] += to or 0
        if not (text or "").strip():
            raise RuntimeError("empty response from model")
        return text

    meta, trace = {}, None
    if system == "simple":
        resp = gen(None, q["text"])
    elif system == "strong":
        resp = gen(STRONG_SYSTEM, q["text"])
    elif system in ("oneshot_random", "oneshot_top"):
        row = ctx["rows"][q["id"]]
        case = ctx["by_id"][row["random" if system == "oneshot_random" else "top1"]]
        resp = gen(ONESHOT_SYSTEM, one_shot_user(case, q["text"]))
        meta = {"case_id": case["case_id"], "book": case["book"]}
    elif system == "framework":
        row = ctx["rows"][q["id"]]
        cands = [ctx["by_id"][cid] for cid in row["candidates"]]
        kid2case = {c["kid"]: c for c in cands}
        sys_p, usr_p = er.build_prompt(ctx["template"], cands, q["text"], ctx["er_args"])
        feedback, attempts, out, issues, weights = "", 0, None, ["no attempt"], {}
        while True:
            attempts += 1
            text = gen(sys_p, usr_p + feedback, json_mode=True)
            out = parse_json(text)
            if not isinstance(out, dict):
                issues, weights = ["output was not valid JSON"], {}
            else:
                issues, weights = er.analyze(out, kid2case, ctx["er_args"])
                if not isinstance(out.get("final_response"), str) or not out["final_response"].strip():
                    issues.append("missing final_response")
            if not issues or attempts > ctx["fw_retries"]:
                break
            feedback = ("\n\nYour previous answer broke these rules: " + "; ".join(issues)
                        + ". Return the corrected JSON object only.")
        resp = out.get("final_response", "") if isinstance(out, dict) and isinstance(out.get("final_response"), str) else ""
        if not resp.strip():
            raise RuntimeError("framework produced no final_response after retries: " + "; ".join(issues))
        sel = [k for k in (out.get("selected_case_ids") or []) if k in kid2case]
        meta = {"attempts": attempts, "issues": issues, "reported_weights": weights,
                "fallback_used": bool(out.get("fallback_used")),
                "risk_flag": (out.get("profile") or {}).get("risk_flag"),
                "selected": [{"kid": k, "case_id": kid2case[k]["case_id"], "book": kid2case[k]["book"]} for k in sel]}
        trace = out
    else:
        raise ValueError(system)
    rec = {"query_id": q["id"], "system": system, "response": resp.strip(), "meta": meta,
           "tokens": tok, "latency_s": round(time.time() - t0, 2), "ts": int(time.time())}
    if trace is not None:
        rec["trace"] = trace
    return rec


# ---------------------------------------------------------------- commands
def cmd_prepare(args):
    tiers = {"case"} | ({"implicit_case"} if args.include_implicit else set())
    kb = er.load_kb(args.cases, tiers)
    if not kb:
        sys.exit("no usable cases in the knowledge bank")
    queries = load_queries(args.queries, args.split, args.limit)
    ep = SimpleNamespace(cases=args.cases, host=args.host, embed_model=args.embed_model,
                         timeout=args.timeout, top_m=args.top_m, max_per_book=args.max_per_book)
    rows = []
    for n, q in enumerate(queries, 1):
        cands = er.prefilter(kb, q["text"], ep)
        top1 = cands[0]
        rng = random.Random(f"{args.seed}-{q['id']}")
        rnd = rng.choice(kb)
        while rnd["case_id"] == top1["case_id"] and len(kb) > 1:
            rnd = rng.choice(kb)
        rows.append({"query_id": q["id"], "candidates": [c["case_id"] for c in cands],
                     "top1": top1["case_id"], "random": rnd["case_id"]})
        print(f"[{n}/{len(queries)}] {q['id']} top1={top1['book'][:30]}", file=sys.stderr)
    bench = Path(args.bench)
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "retrieval.json").write_text(json.dumps({
        "kb_md5": kb_md5(args.cases, tiers), "tiers": sorted(tiers), "cases_file": str(args.cases),
        "embed_model": args.embed_model, "top_m": args.top_m, "max_per_book": args.max_per_book,
        "seed": args.seed, "rows": rows}, indent=1))
    books = {c["book"] for c in kb}
    print(f"prepared {len(rows)} queries; bank has {len(kb)} cases from {len(books)} books", file=sys.stderr)


def run_model(name, cfg, queries, systems, ctx, bench, args):
    if cfg["backend"] != "ollama":
        try:
            api_key(cfg)
        except RuntimeError as e:
            print(f"[{name}] skipped: {e}", file=sys.stderr)
            return
    out_path = Path(bench) / "outputs" / f"{name}.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if (r.get("response") or "").strip():
                    done.add((r["query_id"], r["system"]))
    jobs = [(q, s) for q in queries for s in systems if (q["id"], s) not in done]
    print(f"[{name}] {len(done)} done, {len(jobs)} to run", file=sys.stderr)
    if not jobs:
        return
    lock, counter = threading.Lock(), [0, 0]
    workers = args.workers or cfg.get("workers", 1 if cfg["backend"] == "ollama" else 3)
    ctx = dict(ctx, timeout=cfg.get("timeout", args.timeout))

    def work(job):
        q, s = job
        try:
            rec = do_job(cfg, q, s, ctx)
            rec["model_name"], rec["model"] = name, cfg["model"]
            with lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counter[0] += 1
                print(f"[{name}] {counter[0] + counter[1]}/{len(jobs)} {q['id']} {s} ok ({rec['latency_s']}s)", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - keep the run going, retry next time
            with lock:
                counter[1] += 1
                with open(Path(bench) / "errors.log", "a", encoding="utf-8") as f:
                    f.write(f"{name}\t{q['id']}\t{s}\t{e}\n")
                print(f"[{name}] {counter[0] + counter[1]}/{len(jobs)} {q['id']} {s} FAILED: {str(e)[:150]}", file=sys.stderr)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, jobs))
    print(f"[{name}] finished: {counter[0]} ok, {counter[1]} failed (see errors.log; re-run to retry)", file=sys.stderr)


def cmd_run(args):
    bench = Path(args.bench)
    ret = json.loads((bench / "retrieval.json").read_text())
    tiers = set(ret["tiers"])
    if kb_md5(args.cases, tiers) != ret["kb_md5"]:
        sys.exit("the knowledge-bank file changed since `prepare`; re-run prepare")
    kb = er.load_kb(args.cases, tiers)
    rows = {r["query_id"]: r for r in ret["rows"]}
    queries = [q for q in load_queries(args.queries, args.split, args.limit) if q["id"] in rows]
    if not queries:
        sys.exit("no queries to run (did `prepare` use the same query file and split?)")
    cfgs = json.loads(Path(args.model_config).read_text())
    names = args.models or list(cfgs)
    unknown = [n for n in names if n not in cfgs]
    if unknown:
        sys.exit(f"models not in config: {unknown}")
    systems = SYSTEMS if "all" in args.systems else args.systems
    ctx = {"rows": rows, "by_id": {c["case_id"]: c for c in kb},
           "template": Path(args.prompt_file).read_text(encoding="utf-8"),
           "er_args": SimpleNamespace(k=args.k, shortlist=args.shortlist, temp_w=0.5),
           "fw_retries": args.fw_retries}
    (bench / "outputs").mkdir(parents=True, exist_ok=True)
    for name in names:
        run_model(name, cfgs[name], queries, systems, ctx, bench, args)


def cmd_export(args):
    bench = Path(args.bench)
    qmap = {q["id"]: q for q in load_queries(args.queries)}
    recs = {}
    for p in sorted((bench / "outputs").glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                recs[(r["model_name"], r["query_id"], r["system"])] = r  # last write wins
    long_rows = []
    for (m, qid, s), r in sorted(recs.items()):
        q = qmap.get(qid, {})
        issues = r["meta"].get("issues")
        long_rows.append({"model": m, "query_id": qid, "category": q.get("category"),
                          "wants_advice": q.get("wants_advice"), "difficulty": q.get("difficulty"),
                          "system": s, "words": len(r["response"].split()), "response": r["response"],
                          "framework_issues": "; ".join(issues) if issues else "",
                          "attempts": r["meta"].get("attempts", ""),
                          "source_books": " | ".join(x["book"] for x in r["meta"].get("selected", []))
                          or r["meta"].get("book", "")})
    if not long_rows:
        sys.exit("no outputs found")
    with open(bench / "responses_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        w.writeheader()
        w.writerows(long_rows)
    wide = defaultdict(dict)
    for r in long_rows:
        wide[(r["model"], r["query_id"])][r["system"]] = r["response"]
    with open(bench / "responses_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "query_id", "category", "message"] + SYSTEMS)
        for (m, qid), d in sorted(wide.items()):
            w.writerow([m, qid, qmap.get(qid, {}).get("category"), qmap.get(qid, {}).get("text")]
                       + [d.get(s, "") for s in SYSTEMS])
    stats = defaultdict(list)
    for r in long_rows:
        stats[(r["model"], r["system"])].append(r)
    print(f"{'model':<16}{'system':<16}{'n':>4}{'mean words':>12}{'framework issues':>18}")
    for (m, s), rs in sorted(stats.items()):
        bad = sum(1 for r in rs if r["framework_issues"]) if s == "framework" else ""
        print(f"{m:<16}{s:<16}{len(rs):>4}{sum(r['words'] for r in rs) / len(rs):>12.1f}{bad!s:>18}")
    print(f"\nwrote {bench / 'responses_long.csv'} and {bench / 'responses_wide.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--queries", required=True)
        p.add_argument("--bench", default="bench")
        p.add_argument("--split", choices=["all", "dev", "test"], default="all")
        p.add_argument("--limit", type=int, help="only the first N queries (use 3 for a trial)")

    p = sub.add_parser("prepare")
    common(p)
    p.add_argument("--cases", required=True)
    p.add_argument("--include-implicit", action="store_true")
    p.add_argument("--embed-model", default="nomic-embed-text")
    p.add_argument("--host", default="http://localhost:11434", help="Ollama host used for embeddings")
    p.add_argument("--top-m", type=int, default=30)
    p.add_argument("--max-per-book", type=int, default=6)
    p.add_argument("--seed", type=int, default=13, help="seed for the random-case baseline")
    p.add_argument("--timeout", type=int, default=300)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--cases", required=True)
    p.add_argument("--model-config", default="models.json")
    p.add_argument("--models", nargs="*", help="names from the config (default: all)")
    p.add_argument("--systems", nargs="*", default=["all"], choices=["all"] + SYSTEMS)
    p.add_argument("--prompt-file", default=str(HERE / "single_call_prompt.txt"))
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--shortlist", type=int, default=8)
    p.add_argument("--fw-retries", type=int, default=2, help="re-asks when the framework output breaks the rules")
    p.add_argument("--workers", type=int, help="override threads per model")
    p.add_argument("--timeout", type=int, default=600)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("export")
    p.add_argument("--queries", required=True)
    p.add_argument("--bench", default="bench")
    p.set_defaults(fn=cmd_export)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
