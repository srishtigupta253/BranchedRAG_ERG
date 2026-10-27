#!/usr/bin/env python3
"""
run_ollama.py: run the case-extraction prompt on every window with a local Ollama model.

  python run_ollama.py --work work/ --model qwen2.5:14b --prompt extraction_prompt.txt
  python run_ollama.py --work work/ --model qwen2.5:14b --limit 3      # quick trial on 3 windows

Reads  work/windows.jsonl   (made by `epub_cases.py window`)
Writes work/responses.jsonl (one line per window: {"window_id", "response", ...})

Resumable: windows already answered with valid JSON are skipped, and failed windows are
not written, so just re-run the same command to retry them. Stdlib only.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

STR_LIST = {"type": "array", "items": {"type": "string"}}
SPAN = {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
SCHEMA = {
    "type": "object",
    "properties": {
        "window_id": {"type": "string"},
        "cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "local_id": {"type": "string"},
                    "type": {"type": "string", "enum": ["case", "implicit_case"]},
                    "situation_span": SPAN,
                    "response_span": SPAN,
                    "commentary_span": SPAN,
                    "continues_from_previous": {"type": "boolean"},
                    "continues_in_next": {"type": "boolean"},
                    "recurring_character": {"type": ["string", "null"]},
                    "situation_summary": {"type": "string"},
                    "person_emotions": STR_LIST,
                    "underlying_need": STR_LIST,
                    "recommended_response": {"type": "string"},
                    "avoid": {"type": ["string", "null"]},
                    "principle": {"type": "string"},
                    "empathy_stance": {"type": "string"},
                    "topic_tags": STR_LIST,
                    "has_situation": {"type": "boolean"},
                    "has_recommendation": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["type", "situation_span", "response_span",
                             "has_situation", "has_recommendation", "confidence"],
            },
        },
        "principle_passages": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["cases"],
}


def chat(args, system, user, seed):
    body = {
        "model": args.model,
        "stream": False,
        "format": "json" if args.no_schema else SCHEMA,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": args.temperature, "num_ctx": args.num_ctx,
                    "num_predict": args.num_predict, "seed": seed},
    }
    req = urllib.request.Request(
        args.host.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        return json.load(r)


def valid(text):
    try:
        d = json.loads(text)
        return isinstance(d, dict) and isinstance(d.get("cases"), list)
    except (json.JSONDecodeError, TypeError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--model", required=True, help="e.g. qwen2.5:14b, llama3.1:8b")
    ap.add_argument("--prompt", default="extraction_prompt.txt")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=8192,
                    help="context window; Ollama's default is smaller and silently truncates")
    ap.add_argument("--num-predict", type=int, default=4096, help="max output tokens")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=None, help="only the first N pending windows")
    ap.add_argument("--no-schema", action="store_true",
                    help='use plain format="json" instead of the JSON schema')
    args = ap.parse_args()

    work = Path(args.work)
    system = Path(args.prompt).read_text(encoding="utf-8")
    windows = [json.loads(l) for l in open(work / "windows.jsonl", encoding="utf-8") if l.strip()]
    out_path = work / "responses.jsonl"

    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if valid(r.get("response")):
                    done.add(r["window_id"])
    pending = [w for w in windows if w["window_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(done)} done, {len(pending)} to run with {args.model}", file=sys.stderr)

    failed = []
    with open(out_path, "a", encoding="utf-8") as out:
        for n, w in enumerate(pending, 1):
            ok = False
            for attempt in range(args.retries):
                try:
                    t0 = time.time()
                    res = chat(args, system, w["user_message"], seed=attempt)
                    text = res["message"]["content"]
                    p_tok = res.get("prompt_eval_count", 0)
                    if p_tok >= 0.95 * args.num_ctx:
                        print(f"  WARNING {w['window_id']}: prompt used {p_tok}/{args.num_ctx} tokens, "
                              f"input may be truncated; raise --num-ctx or shrink windows",
                              file=sys.stderr)
                    if res.get("done_reason") == "length":
                        print(f"  WARNING {w['window_id']}: output hit --num-predict", file=sys.stderr)
                    if valid(text):
                        out.write(json.dumps({
                            "window_id": w["window_id"], "response": text,
                            "model": args.model, "prompt_tokens": p_tok,
                        }, ensure_ascii=False) + "\n")
                        out.flush()
                        ok = True
                        print(f"[{n}/{len(pending)}] {w['window_id']} ok ({time.time() - t0:.0f}s)",
                              file=sys.stderr)
                        break
                    print(f"  {w['window_id']} attempt {attempt + 1}: invalid JSON, retrying",
                          file=sys.stderr)
                except (urllib.error.URLError, TimeoutError, KeyError) as e:
                    print(f"  {w['window_id']} attempt {attempt + 1}: {e}", file=sys.stderr)
                    time.sleep(2 * (attempt + 1))
            if not ok:
                failed.append(w["window_id"])
    if failed:
        print(f"{len(failed)} windows failed: {failed}\nRe-run the same command to retry them.",
              file=sys.stderr)
    else:
        print("all pending windows done", file=sys.stderr)


if __name__ == "__main__":
    main()
