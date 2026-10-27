#!/usr/bin/env python3
"""
epub_cases.py: EPUB -> paragraph IDs -> overlapping windows -> merged case records.

Requires: beautifulsoup4, lxml
"""
import argparse
import json
import posixpath
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import warnings

"""from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning"""
from bs4 import BeautifulSoup

BLOCK = ["p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
         "blockquote", "dd", "dt", "figcaption", "pre", "div"]
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Sections whose (nav or heading) title fully matches are skipped by default.
SKIP_TITLE_RE = re.compile(
    r"^(cover|title page|half title|copyright|copyright page|dedication|epigraph|"
    r"contents|table of contents|acknowledg(e)?ments?|about the author|also by.*|"
    r"praise for.*|notes|endnotes|bibliography|references|index|permissions|"
    r"credits|colophon|other books by.*|other titles|discover more.*|next reads?|"
    r"what.s next.*|further reading)$",
    re.I,
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def norm(s):
    return re.sub(r"\s+", " ", s.replace("\u00a0", " ")).strip()


def local_find(soup, name):
    return soup.find(lambda t: t.name and t.name.split(":")[-1] == name)


def resolve(base_dir, href):
    return posixpath.normpath(posixpath.join(base_dir, unquote(href.split("#")[0])))


def slugify(s, n=30):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:n] or "book"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------
# 1. parse
# ----------------------------------------------------------------------------
def read_toc(z, manifest):
    """Map archive path -> first TOC title that points to it (EPUB3 nav, else NCX)."""
    titles = {}
    for it in manifest.values():
        if "nav" in it["props"].split():
            soup = BeautifulSoup(z.read(it["path"]), "lxml")
            navs = soup.find_all("nav")
            toc = next((n for n in navs if "toc" in str(n.get("epub:type", ""))),
                       navs[0] if navs else None)
            if toc:
                base = posixpath.dirname(it["path"])
                for a in toc.find_all("a", href=True):
                    t = norm(a.get_text())
                    if t:
                        titles.setdefault(resolve(base, a["href"]), t)
    if not titles:
        for it in manifest.values():
            if it["type"] == "application/x-dtbncx+xml":
                soup = BeautifulSoup(z.read(it["path"]), "xml")
                base = posixpath.dirname(it["path"])
                for np_ in soup.find_all("navPoint"):
                    lab, con = np_.find("navLabel"), np_.find("content")
                    if lab and con and con.get("src"):
                        t = norm(lab.get_text())
                        if t:
                            titles.setdefault(resolve(base, con["src"]), t)
    return titles


def is_pagebreak(el):
    return ("pagebreak" in str(el.get("epub:type", ""))
            or el.get("role") == "doc-pagebreak")


def extract_paragraphs(html_bytes, page_state):
    """Leaf block elements in document order -> list of (kind, text, page_label)."""
    soup = BeautifulSoup(html_bytes, "lxml")
    body = soup.body or soup
    for t in body(["script", "style", "nav"]):
        t.decompose()
    for br in body.find_all("br"):
        br.replace_with(" ")
    for pb in body.find_all(is_pagebreak):
        label = pb.get("title") or norm(pb.get_text())
        pb["data-pg"] = label or ""
        pb.clear()  # keep page numbers out of the paragraph text
    out = []
    for el in body.find_all(True):
        if el.has_attr("data-pg"):
            if el["data-pg"]:
                page_state["page"] = el["data-pg"]
            continue
        if el.name not in BLOCK or el.find(BLOCK):
            continue  # only leaf blocks, so nothing is counted twice
        text = norm(el.get_text())
        if len(re.sub(r"\W", "", text)) < 2:
            continue
        kind = "heading" if el.name in HEADINGS else "text"
        out.append((kind, text, page_state["page"]))
    return out


def cmd_parse(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(args.epub)

    container = BeautifulSoup(z.read("META-INF/container.xml"), "xml")
    opf_path = container.find("rootfile")["full-path"]
    opf = BeautifulSoup(z.read(opf_path), "xml")
    opf_dir = posixpath.dirname(opf_path)

    manifest = {}
    for it in opf.find_all("item"):
        manifest[it["id"]] = {
            "path": posixpath.normpath(posixpath.join(opf_dir, unquote(it["href"]))),
            "type": it.get("media-type", ""),
            "props": it.get("properties", ""),
        }
    spine = [r["idref"] for r in opf.find_all("itemref") if r.get("linear", "yes") != "no"]
    toc_titles = read_toc(z, manifest)

    t = local_find(opf, "title")
    a = local_find(opf, "creator")
    title = norm(t.get_text()) if t else Path(args.epub).stem
    author = norm(a.get_text()) if a else "Unknown"
    slug = slugify(title)

    paragraphs, sections = [], []
    page_state = {"page": None}
    for sec_idx, idref in enumerate(spine):
        item = manifest.get(idref)
        if not item or "html" not in item["type"]:
            continue
        try:
            raw = z.read(item["path"])
        except KeyError:
            print(f"warning: {item['path']} missing from archive", file=sys.stderr)
            continue
        paras = extract_paragraphs(raw, page_state)
        words = sum(len(p[1].split()) for p in paras)
        first_heading = next((p[1] for p in paras if p[0] == "heading"), None)
        sec_title = toc_titles.get(item["path"]) or first_heading or Path(item["path"]).stem

        skipped, reason = False, None
        if not paras:
            skipped, reason = True, "empty"
        elif not args.no_skip:
            for cand in (sec_title, first_heading or ""):
                if SKIP_TITLE_RE.match(cand.strip()):
                    skipped, reason = True, f"title matches skip list: {cand!r}"
                    break

        first_idx = len(paragraphs)
        for kind, text, page in paras:
            idx = len(paragraphs)
            paragraphs.append({
                "id": f"P{idx:05d}", "idx": idx, "section_idx": sec_idx,
                "kind": kind, "text": text, "words": len(text.split()),
                "page_label": page,
            })
        sections.append({
            "section_idx": sec_idx, "file": item["path"], "title": sec_title,
            "n_paragraphs": len(paras), "words": words,
            "first_idx": first_idx, "last_idx": len(paragraphs) - 1,
            "skipped": skipped, "skip_reason": reason,
        })

    meta = {"title": title, "author": author, "slug": slug, "source_file": Path(args.epub).name}
    (work / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (work / "sections.json").write_text(json.dumps(sections, indent=2, ensure_ascii=False))
    write_jsonl(work / "paragraphs.jsonl", paragraphs)

    kept = [s for s in sections if not s["skipped"]]
    print(f"{title} by {author}")
    print(f"{len(paragraphs)} paragraphs, {len(sections)} sections "
          f"({len(kept)} kept, {len(sections) - len(kept)} skipped)")
    print(f"Review {work / 'sections.json'}: set \"skipped\" true/false by hand if the "
          f"heuristics got a section wrong, then run `window`.")


# ----------------------------------------------------------------------------
# 2. window
# ----------------------------------------------------------------------------
def make_windows(paras, target, overlap, max_words):
    """Greedy windows over paragraph list. Returns [(start, end_inclusive)]."""
    n, out, start = len(paras), [], 0
    while n:
        end, words = start, 0
        while end < n:
            w = paras[end]["words"]
            if words > 0 and words + w > max_words:
                break
            words += w
            end += 1
            if words >= target:
                break
        last = end - 1
        out.append((start, last))
        if last >= n - 1:
            break
        ow, ns = 0, last + 1
        while ns > start + 1 and ow < overlap:  # step back to cover ~overlap words
            ns -= 1
            ow += paras[ns]["words"]
        start = ns
    return out


def render_paragraph(p):
    prefix = "# " if p["kind"] == "heading" else ""
    return f"[{p['id']}] {prefix}{p['text']}"


def cmd_window(args):
    work = Path(args.work)
    meta = json.loads((work / "meta.json").read_text())
    sections = json.loads((work / "sections.json").read_text())
    paras = read_jsonl(work / "paragraphs.jsonl")

    # Group kept sections; sections shorter than --tiny-words are prepended to the next one.
    groups, pending = [], []
    for s in sections:
        if s["skipped"] or s["n_paragraphs"] == 0:
            continue
        sec_paras = paras[s["first_idx"]: s["last_idx"] + 1]
        if s["words"] < args.tiny_words:
            pending.extend(sec_paras)
            continue
        groups.append((s, pending + sec_paras))
        pending = []
    # Tiny sections with no real section after them are back matter: drop them.
    if pending and not groups:
        groups.append((next(s for s in sections if not s["skipped"]), pending))

    windows, k = [], 0
    for sec, plist in groups:
        prev_end = None
        for (a, b) in make_windows(plist, args.target_words, args.overlap_words, args.max_words):
            k += 1
            wid = f"{meta['slug']}_w{k:04d}"
            chunk = plist[a: b + 1]
            if prev_end is not None and a <= prev_end:
                overlap = f"{plist[a]['id']} to {plist[prev_end]['id']}"
            else:
                overlap = None
            overlap_line = (f"This window overlaps the previous one by paragraphs {overlap}."
                            if overlap else
                            "This is the first window of this chapter; there is no overlap.")
            msg = (f"Book: {meta['title']} by {meta['author']}\n"
                   f"Chapter: {sec['title']}\n"
                   f"Window ID: {wid}\n"
                   f"{overlap_line}\n\n"
                   f"TEXT:\n" + "\n".join(render_paragraph(p) for p in chunk))
            windows.append({
                "window_id": wid, "section_idx": sec["section_idx"],
                "section_title": sec["title"],
                "start_idx": chunk[0]["idx"], "end_idx": chunk[-1]["idx"],
                "first_id": chunk[0]["id"], "last_id": chunk[-1]["id"],
                "overlap_ids": overlap,
                "n_words": sum(p["words"] for p in chunk),
                "user_message": msg,
            })
            prev_end = b
    write_jsonl(work / "windows.jsonl", windows)
    avg = sum(w["n_words"] for w in windows) / max(len(windows), 1)
    print(f"{len(windows)} windows across {len(groups)} chapters (avg {avg:.0f} words/window)")


# ----------------------------------------------------------------------------
# 4. merge
# ----------------------------------------------------------------------------
def parse_response(r):
    if isinstance(r, dict):
        return r
    if not isinstance(r, str):
        return None
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", r.strip())
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


def expand(spans, id2idx, lo, hi, stats):
    """[[start_id, end_id], ...] -> set of paragraph idx, clipped to the window."""
    out = set()
    for sp in spans or []:
        if isinstance(sp, str):
            sp = [sp, sp]
        if not isinstance(sp, (list, tuple)) or not sp:
            stats["invalid_span_refs"] += 1
            continue
        a = id2idx.get(sp[0])
        b = id2idx.get(sp[-1])
        if a is None or b is None:
            stats["invalid_span_refs"] += 1
            continue
        a, b = min(a, b), max(a, b)
        if a < lo or b > hi:
            stats["out_of_window_refs"] += 1
            a, b = max(a, lo), min(b, hi)
            if a > b:
                continue
        out.update(range(a, b + 1))
    return out


def should_merge(a, b, jaccard_thr, gap):
    if a["win"] == b["win"]:
        return False
    A, B = a["all"], b["all"]
    first, second = (a, b) if min(A) <= min(B) else (b, a)
    # continuation: first runs off the end of its window, second picks up where it starts
    if (first["meta"].get("continues_in_next") and second["meta"].get("continues_from_previous")
            and first["section"] == second["section"]
            and min(first["all"]) <= min(second["all"]) <= max(first["all"]) + 1 + gap):
        return True
    inter = len(A & B)
    if not inter:
        return False
    if inter / len(A | B) >= jaccard_thr:
        return True
    small = a if len(A) <= len(B) else b
    truncated = small["meta"].get("continues_from_previous") or small["meta"].get("continues_in_next")
    return bool(truncated) and inter / min(len(A), len(B)) >= 0.8


def union_lists(lists):
    seen, out = set(), []
    for lst in lists:
        for x in lst or []:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def merge_cluster(members):
    primary = max(members, key=lambda c: (len(c["all"]), c["meta"].get("confidence") or 0))
    earliest = min(members, key=lambda c: min(c["all"]))
    latest = max(members, key=lambda c: max(c["all"]))
    meta = dict(primary["meta"])
    meta["continues_from_previous"] = bool(earliest["meta"].get("continues_from_previous"))
    meta["continues_in_next"] = bool(latest["meta"].get("continues_in_next"))
    meta["has_situation"] = any(c["meta"].get("has_situation") for c in members)
    meta["has_recommendation"] = any(c["meta"].get("has_recommendation") for c in members)
    meta["confidence"] = max((c["meta"].get("confidence") or 0) for c in members)
    meta["recurring_character"] = next(
        (c["meta"].get("recurring_character") for c in members
         if c["meta"].get("recurring_character")), None)
    for key in ("person_emotions", "underlying_need", "topic_tags"):
        meta[key] = union_lists(c["meta"].get(key) for c in members)
    notes = union_lists([c["meta"].get("notes")] for c in members)
    meta["notes"] = " | ".join(n for n in notes if n)
    return {
        "meta": meta,
        "sit": set().union(*[c["sit"] for c in members]),
        "resp": set().union(*[c["resp"] for c in members]),
        "com": set().union(*[c["com"] for c in members]),
        "wins": sorted({c["win"] for c in members}),
        "section": primary["section"],
        "types": {c["meta"].get("type") for c in members},
        "n_raw": len(members),
    }


def cmd_merge(args):
    work = Path(args.work)
    book = json.loads((work / "meta.json").read_text())
    paras = read_jsonl(work / "paragraphs.jsonl")
    windows = read_jsonl(work / "windows.jsonl")
    responses = read_jsonl(args.responses)
    sections = {s["section_idx"]: s for s in json.loads((work / "sections.json").read_text())}
    id2idx = {p["id"]: p["idx"] for p in paras}
    wmap = {w["window_id"]: w for w in windows}

    stats = defaultdict(int)
    failed, seen_windows, raw = [], set(), []
    for r in responses:
        w = wmap.get(r.get("window_id"))
        if w is None:
            stats["unknown_window_id"] += 1
            continue
        seen_windows.add(w["window_id"])
        data = parse_response(r.get("response", r.get("output")))
        if not isinstance(data, dict) or not isinstance(data.get("cases", []), list):
            failed.append(w["window_id"])
            continue
        for cs in data.get("cases") or []:
            if not isinstance(cs, dict):
                continue
            lo, hi = w["start_idx"], w["end_idx"]
            sit = expand(cs.get("situation_span"), id2idx, lo, hi, stats)
            resp = expand(cs.get("response_span"), id2idx, lo, hi, stats)
            com = expand(cs.get("commentary_span"), id2idx, lo, hi, stats)
            allp = sit | resp | com
            if not allp:
                stats["cases_without_valid_span"] += 1
                continue
            raw.append({"win": w["window_id"], "section": w["section_idx"], "meta": cs,
                        "sit": sit, "resp": resp, "com": com, "all": allp})
    stats["cases_raw"] = len(raw)

    # union-find over mergeable pairs
    parent = list(range(len(raw)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(raw)):
        for j in range(i + 1, len(raw)):
            if should_merge(raw[i], raw[j], args.jaccard, args.stitch_gap):
                parent[find(i)] = find(j)
    clusters = defaultdict(list)
    for i, c in enumerate(raw):
        clusters[find(i)].append(c)
    merged = [merge_cluster(m) for m in clusters.values()]
    merged.sort(key=lambda c: min(c["sit"] | c["resp"] | c["com"]))

    def text_of(idxs):
        return "\n\n".join(paras[i]["text"] for i in sorted(idxs))

    records, links = [], defaultdict(list)
    for n, c in enumerate(merged, 1):
        allp = c["sit"] | c["resp"] | c["com"]
        m = c["meta"]
        both = bool(m["has_situation"] and m["has_recommendation"])
        if both and m["confidence"] >= args.min_conf:
            tier = "case" if "case" in c["types"] else "implicit_case"
        else:
            tier = "incomplete"
        cid = f"{book['slug']}_c{n:04d}"
        lo, hi = min(allp), max(allp)
        sec = sections.get(c["section"], {})
        rec = {
            "case_id": cid, "tier": tier,
            "book": book["title"], "author": book["author"],
            "section": sec.get("title"), "section_idx": c["section"],
            "first_para_id": paras[lo]["id"], "last_para_id": paras[hi]["id"],
            "page_start": paras[lo]["page_label"], "page_end": paras[hi]["page_label"],
            "type": m.get("type"),
            "situation_text": text_of(c["sit"]),
            "response_text": text_of(c["resp"]),
            "commentary_text": text_of(c["com"]),
            "full_text": text_of(allp),
            "situation_summary": m.get("situation_summary"),
            "person_emotions": m["person_emotions"],
            "underlying_need": m["underlying_need"],
            "recommended_response": m.get("recommended_response"),
            "avoid": m.get("avoid"),
            "principle": m.get("principle"),
            "empathy_stance": m.get("empathy_stance"),
            "topic_tags": m["topic_tags"],
            "recurring_character": m["recurring_character"],
            "confidence": m["confidence"],
            "has_situation": m["has_situation"],
            "has_recommendation": m["has_recommendation"],
            "unresolved_continuation": bool(m["continues_from_previous"] or m["continues_in_next"]),
            "notes": m["notes"],
            "source_windows": c["wins"], "n_raw_merged": c["n_raw"],
        }
        records.append(rec)
        if m["recurring_character"]:
            links[m["recurring_character"].strip().lower()].append(cid)
    for rec in records:
        key = (rec["recurring_character"] or "").strip().lower()
        rec["linked_case_ids"] = [x for x in links.get(key, []) if x != rec["case_id"]] if key else []

    write_jsonl(work / "cases_bookname.jsonl", records)
    (work / "failed_windows.txt").write_text("\n".join(failed))

    per_section = defaultdict(lambda: defaultdict(int))
    for r in records:
        per_section[r["section"]][r["tier"]] += 1
    stats.update({
        "windows_total": len(windows),
        "windows_with_response": len(seen_windows),
        "windows_missing": len(windows) - len(seen_windows),
        "windows_parse_failed": len(failed),
        "cases_merged": len(records),
        "duplicates_removed": len(raw) - len(records),
        "by_tier": {t: sum(1 for r in records if r["tier"] == t)
                    for t in ("case", "implicit_case", "incomplete")},
        "unresolved_continuations": sum(r["unresolved_continuation"] for r in records),
        "per_section": {k: dict(v) for k, v in per_section.items()},
    })
    (work / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "per_section"}, indent=2))
    if failed:
        print(f"{len(failed)} windows failed to parse; IDs in {work / 'failed_windows.txt'}. "
              f"Re-run those windows and append to the responses file.")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse")
    p.add_argument("epub")
    p.add_argument("--work", default="work")
    p.add_argument("--no-skip", action="store_true", help="keep front/back matter too")
    p.set_defaults(fn=cmd_parse)

    w = sub.add_parser("window")
    w.add_argument("--work", default="work")
    w.add_argument("--target-words", type=int, default=1800, help="~4-5 pages")
    w.add_argument("--overlap-words", type=int, default=350, help="~1 page")
    w.add_argument("--max-words", type=int, default=2600, help="hard cap per window")
    w.add_argument("--tiny-words", type=int, default=150,
                   help="sections shorter than this are attached to the next section")
    w.set_defaults(fn=cmd_window)

    m = sub.add_parser("merge")
    m.add_argument("--work", default="work")
    m.add_argument("--responses", required=True)
    m.add_argument("--jaccard", type=float, default=0.5, help="overlap needed to treat two cases as duplicates")
    m.add_argument("--stitch-gap", type=int, default=1, help="max paragraphs between continuation pieces")
    m.add_argument("--min-conf", type=float, default=0.5)
    m.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
