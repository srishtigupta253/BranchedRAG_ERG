#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

SPLIT = "=====USER====="


def post(host, path, body, timeout):
    req = urllib.request.Request(host.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# ------------------------------------------------------------------ knowledge bank
def load_kb(path, tiers):
    cases = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("tier") not in tiers:
            continue
        if not r.get("situation_summary") or not r.get("recommended_response"):
            continue
        cases.append(r)
    for i, c in enumerate(cases, 1):
        c["kid"] = f"K{i:04d}"  # short stable id, saves tokens in the prompt
    return cases


def case_view(c):
    return (f"{c['situation_summary']} Emotions: {', '.join(c.get('person_emotions') or [])}. "
            f"Needs: {', '.join(c.get('underlying_need') or [])}. "
            f"Topics: {', '.join(c.get('topic_tags') or [])}.")


def catalog_entry(c):
    j = lambda x: ", ".join(x) if x else "unclear"
    return (f"[{c['kid']}] book: {c['book']} ({c.get('author') or 'unknown'}) | "
            f"stance: {c.get('empathy_stance') or 'unclear'}\n"
            f"  situation: {c['situation_summary']}\n"
            f"  person needs: {j(c.get('underlying_need'))}; emotions: {j(c.get('person_emotions'))}; "
            f"topics: {j(c.get('topic_tags'))}\n"
            f"  expert recommends: {c['recommended_response']}\n"
            f"  expert avoids: {c.get('avoid') or 'nothing stated'}\n"
            f"  principle: {c.get('principle') or ''}")


# ------------------------------------------------------------------ prefilter
def cosine(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return d / (na * nb) if na and nb else 0.0


def embed(args, texts):
    out = []
    for i in range(0, len(texts), 64):
        res = post(args.host, "/api/embed", {"model": args.embed_model, "input": texts[i:i + 64]},
                   args.timeout)
        out.extend(res["embeddings"])
    return out


def prefilter(cases, query, args):
    cache_path = Path(args.cases).with_name(f"kb_embeddings_{re.sub(r'[^a-z0-9]+', '_', args.embed_model)}.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = []
    for c in cases:
        text = case_view(c)
        h = hashlib.md5(text.encode()).hexdigest()
        if c["case_id"] not in cache or cache[c["case_id"]]["h"] != h:
            todo.append((c, text, h))
    if todo:
        print(f"embedding {len(todo)} cases (cached afterwards)...", file=sys.stderr)
        for (c, text, h), v in zip(todo, embed(args, [t for _, t, _ in todo])):
            cache[c["case_id"]] = {"h": h, "v": v}
        cache_path.write_text(json.dumps(cache))
    qv = embed(args, [query])[0]
    ranked = sorted(cases, key=lambda c: -cosine(qv, cache[c["case_id"]]["v"]))
    out, per_book = [], {}
    for c in ranked:  # cap per book so several authors stay in the candidate pool
        if per_book.get(c["book"], 0) >= args.max_per_book:
            continue
        per_book[c["book"]] = per_book.get(c["book"], 0) + 1
        out.append(c)
        if len(out) >= args.top_m:
            break
    return out


# ------------------------------------------------------------------ prompt + call
def build_prompt(template, cands, query, args):
    system, user = template.split(SPLIT)
    fill = {"{{K}}": str(args.k), "{{SHORTLIST_SIZE}}": str(args.shortlist),
            "{{N_CASES}}": str(len(cands)),
            "{{CATALOG}}": "\n".join(catalog_entry(c) for c in cands),
            "{{USER_MESSAGE}}": query.strip()}
    for k, v in fill.items():
        system, user = system.replace(k, v), user.replace(k, v)
    return system.strip(), user.strip()


def analyze(out, kid2case, args):
    issues = []
    sel = out.get("selected_case_ids") or []
    for k in sel:
        if k not in kid2case:
            issues.append(f"unknown case id {k}")
    books = [kid2case[k]["book"] for k in sel if k in kid2case]
    if len(set(books)) < len(books):
        issues.append("more than one selected case from the same book")
    if len(sel) != args.k and not out.get("fallback_used"):
        issues.append(f"selected {len(sel)} cases, expected {args.k}")
    sc = {s.get("case_id"): s for s in out.get("scores") or []}
    sh = {s.get("case_id"): s for s in out.get("shortlist") or []}
    raw = {}
    for k in sel:
        h, s = sh.get(k, {}), sc.get(k, {})
        rel = (0.3 * num(h.get("situation_match")) + 0.3 * num(h.get("need_match"))
               + 0.25 * num(h.get("emotion_tone_match")) + 0.15 * num(h.get("topic_match")))
        z = (0.4 * rel + 0.3 * num(s.get("need_match")) + 0.3 * num(s.get("strategy_fidelity"))) / args.temp_w
        raw[k] = 0.0 if s.get("avoid_violation") else math.exp(z)
    tot = sum(raw.values()) or 1.0
    return issues, {k: round(v / tot, 3) for k, v in raw.items()}  # for reporting; not fed back


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="cases.jsonl from epub_cases.py merge (may be concatenated over books)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--query")
    g.add_argument("--query-file")
    ap.add_argument("--prompt-file", default=str(Path(__file__).with_name("single_call_prompt.txt")))
    ap.add_argument("--mode", choices=["prefilter", "full"], default="prefilter")
    ap.add_argument("--tiers", default="case", help="comma list: case,implicit_case")
    ap.add_argument("--top-m", type=int, default=30, help="candidates kept after embedding prefilter")
    ap.add_argument("--max-per-book", type=int, default=6)
    ap.add_argument("--k", type=int, default=3, help="branches (cases used)")
    ap.add_argument("--shortlist", type=int, default=8)
    ap.add_argument("--model", help="Ollama chat model")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--max-ctx", type=int, default=32768)
    ap.add_argument("--num-predict", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--temp-w", type=float, default=0.5, help="softmax temperature for reported branch weights")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dump-prompt", help="write the assembled prompt to this file and exit")
    ap.add_argument("--out", default="response.json")
    args = ap.parse_args()

    query = args.query or Path(args.query_file).read_text(encoding="utf-8")
    kb = load_kb(args.cases, set(args.tiers.split(",")))
    if not kb:
        sys.exit("no usable cases (need tier in --tiers, a situation_summary and a recommended_response)")
    cands = kb if args.mode == "full" else prefilter(kb, query, args)
    system, user = build_prompt(Path(args.prompt_file).read_text(encoding="utf-8"), cands, query, args)
    est = (len(system) + len(user)) // 4
    print(f"{len(kb)} cases in bank, {len(cands)} in prompt, ~{est} prompt tokens "
          f"({len({c['book'] for c in cands})} books)", file=sys.stderr)

    if args.dump_prompt:
        Path(args.dump_prompt).write_text(system + "\n\n" + user, encoding="utf-8")
        print(f"prompt written to {args.dump_prompt}", file=sys.stderr)
        return
    if not args.model:
        sys.exit("--model is required unless --dump-prompt is used")
    need_ctx = est + args.num_predict + 500
    if need_ctx > args.max_ctx:
        sys.exit(f"prompt needs ~{need_ctx} tokens of context, above --max-ctx {args.max_ctx}. "
                 f"Lower --top-m, or use a model/--max-ctx that supports more.")
    num_ctx = min(args.max_ctx, ((need_ctx + 1023) // 1024) * 1024)

    res = post(args.host, "/api/chat", {
        "model": args.model, "stream": False, "format": "json",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": args.temperature, "num_ctx": num_ctx, "num_predict": args.num_predict},
    }, args.timeout)
    if res.get("prompt_eval_count", 0) >= 0.95 * num_ctx:
        print("WARNING: prompt may have been truncated; raise --max-ctx or lower --top-m", file=sys.stderr)
    try:
        out = json.loads(res["message"]["content"])
    except json.JSONDecodeError:
        Path(args.out).write_text(res["message"]["content"], encoding="utf-8")
        sys.exit(f"model did not return valid JSON; raw text saved to {args.out}")

    kid2case = {c["kid"]: c for c in cands}
    issues, weights = analyze(out, kid2case, args)
    trace = {
        "user_message": query.strip(), "model": args.model, "mode": args.mode,
        "output": out, "validation_issues": issues, "reported_weights": weights,
        "selected_sources": [{"kid": k, "case_id": kid2case[k]["case_id"], "book": kid2case[k]["book"],
                              "pages": [kid2case[k].get("page_start"), kid2case[k].get("page_end")]}
                             for k in out.get("selected_case_ids") or [] if k in kid2case],
    }
    Path(args.out).write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + (out.get("final_response") or "(no final_response)") + "\n")
    if issues:
        print("validation issues:", "; ".join(issues), file=sys.stderr)
    print(f"full trace saved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
