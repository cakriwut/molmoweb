from dataclasses import dataclass, asdict
from typing import Any
import logging
import time
import re
import difflib
import unicodedata
import openai
import math

log = logging.getLogger(__name__)


# ---- LLM plumbing ------------------------------------------------------------
def _client(api_key: str):
    return openai.OpenAI(api_key=api_key)


def get_chat_response(
    prompt: str,
    api_key: str,
    *,
    system_prompt: str | None = None,
    model: str = "gpt-4.1-2025-04-14",
    temperature: float = 0,
    max_tokens: int = 1024,
    patience: int = 6,
    sleep_time: float = 0.6,
) -> str:
    msgs = [{"role": "user", "content": prompt}]
    if system_prompt:
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    cli = _client(api_key)
    for _ in range(patience):
        try:
            r = cli.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            out = r.choices[0].message.content or ""
            out = out.strip()
            if out:
                return out
        except Exception as e:
            log.warning(f"LLM error: {e}")
            time.sleep(sleep_time)
    return ""


# ---- Normalization & heuristics ----------------------------------------------
_WORD_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _strip_quotes(s: str) -> str:
    return (
        s.replace("“", "")
        .replace("”", "")
        .replace("‘", "")
        .replace("’", "")
        .strip("\"' ")
    )


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = _strip_quotes(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[ \t]*\.[ \t]*$", "", s)  # drop trailing period
    return s


def extract_numbers(s: str) -> list[float]:
    if not s:
        return []
    s_norm = normalize_text(s)
    nums: list[float] = []

    # digits, thousand-sep, decimals, percents
    for m in re.finditer(
        r"(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)(?P<pct>\s*%)?", s_norm
    ):
        x = float(m.group("num").replace(",", ""))
        if m.group("pct"):
            x = x / 100.0
        nums.append(x)

    # simple word-numbers
    for w, v in _WORD_NUM.items():
        if re.search(rf"\b{w}\b", s_norm):
            nums.append(float(v))

    # simple time like "1 hour 15 minutes" -> [1, 15]
    # We don't convert to total minutes here; numeric set equality already helps.
    return nums


def token_set(s: str) -> list[str]:
    s = normalize_text(s)
    s = re.sub(r"[^a-z0-9\s\-\+&:/]", " ", s)
    toks = [t for t in re.split(r"[\s,/;]+|(?:\s+and\s+)", s) if t]
    return toks


def list_items(s: str) -> list[str]:
    # for answers like "A; B; C and D" or "A, B, C"
    parts = [
        p.strip()
        for p in re.split(r",|;|\band\b|\u2013|\u2014", normalize_text(s))
        if p.strip()
    ]
    # drop trivial words
    parts = [re.sub(r"^(the|a|an)\s+", "", p) for p in parts]
    return [p for p in parts if p]


def equal_numbers(gold: str, pred: str, *, rtol=1e-2, atol=1e-6) -> bool:
    ng = extract_numbers(gold)
    np_ = extract_numbers(pred)
    if not ng and not np_:
        return False
    if len(ng) != len(np_):
        return False
    for a, b in zip(sorted(ng), sorted(np_)):
        if not (abs(a - b) <= max(atol, rtol * max(1.0, abs(a), abs(b)))):
            return False
    return True


def equal_lists(gold: str, pred: str) -> bool:
    g = set(list_items(gold))
    p = set(list_items(pred))
    if not g or not p:
        return False
    # soft compare via normalized tokens
    g_norm = {" ".join(token_set(x)) for x in g}
    p_norm = {" ".join(token_set(x)) for x in p}
    # quick exact
    if g_norm == p_norm:
        return True
    # allow small symmetric differences if gold says "the same" length
    return False


def short_string_close(gold: str, pred: str, *, thr=0.9) -> bool:
    g = normalize_text(gold)
    p = normalize_text(pred)
    if not g or not p:
        return False
    if g in p or p in g:
        return True
    ratio = difflib.SequenceMatcher(None, g, p).ratio()
    return ratio >= thr


# ---- Main judge ---------------------------------------------------------------
def judge_equivalence(
    question: str, gold_answer: str, pred_answer: str, api_key: str | None
) -> dict:
    q = question or ""
    g = gold_answer or ""
    p = pred_answer or ""
    # 1) Heuristics first
    if equal_numbers(g, p):
        return {"match": True}
    if equal_lists(g, p):
        return {"match": True}
    if short_string_close(g, p):
        return {"match": True}
    # very short gold like "Apply", "English (UK)"
    if len(token_set(g)) <= 3 and g and p and normalize_text(g) == normalize_text(p):
        return {"match": True}

    # 2) LLM fallback (strict, JSON)
    sys = "You are a strict but fair judge for short QA on website screenshots."
    prompt = f"""
Question: {q}

Gold answer: {g}
Model answer: {p}

Decide if the Model answer semantically matches the Gold answer for this question.
Allow minor formatting/casing, punctuation differences, synonyms, and numeral/word equivalence.
If the gold is a list, the model must include the same items (order not required).
If the gold is numeric/percentage/time, allow mathematically equivalent forms.

Output exactly one token: TRUE or FALSE.
"""
    out = get_chat_response(prompt, api_key, system_prompt=sys)
    match = out.strip().upper().startswith("T")
    return {"match": bool(match)}


@dataclass
class JudgeSignal:
    name: str  # e.g., "numbers_equal"
    verdict: bool  # this signal's decision
    weight: float  # confidence weight used for tie-breaks
    details: dict[str, Any]  # small, serializable evidence


def _mk(name, verdict, weight, **details) -> JudgeSignal:
    return JudgeSignal(
        name=name, verdict=bool(verdict), weight=float(weight), details=details
    )


def judge_equivalence(
    question: str, gold_answer: str, pred_answer: str, api_key: str
) -> dict[str, Any]:
    """Run *all* signals (heuristics + LLM) and compute a composite score.

    Returns
    -------
    dict with keys:
      - match: bool                    # final match decision
      - decision: "heuristic"|"llm"|"none"
      - primary_reason: str
      - reasons: list[dict]            # each has name, verdict, weight, details
      - used_llm: bool
      - composite: float               # softmax-weighted trigger score in [0,1]
    """

    q = question or ""
    g = gold_answer or ""
    p = pred_answer or ""

    # Build all signals (keep weights stable and explicit)
    sigs: list[JudgeSignal] = []

    # numeric equivalence
    ng, np_ = extract_numbers(g), extract_numbers(p)
    if ng or np_:
        nums_ok = equal_numbers(g, p)
        sigs.append(_mk("numbers_equal", nums_ok, 0.95, gold_numbers=ng, pred_numbers=np_))

    # list / set equality
    gl, pl = list_items(g), list_items(p)
    if gl or pl:
        gset = {" ".join(token_set(x)) for x in gl}
        pset = {" ".join(token_set(x)) for x in pl}
        lists_ok = (gset == pset) and bool(gset)
        sigs.append(_mk("list_set_equal", lists_ok, 0.90,
                        gold_items=sorted(gset), pred_items=sorted(pset)))

    # short normalized exact for short spans
    geq = (normalize_text(g) == normalize_text(p)) and (len(token_set(g)) <= 8)
    sigs.append(_mk("short_span_exact", geq, 0.90,
                    gold_norm=normalize_text(g), pred_norm=normalize_text(p)))

    # substring containment
    sub_ok = normalize_text(g) in normalize_text(p) or normalize_text(p) in normalize_text(g)
    sigs.append(_mk("substring_close", sub_ok, 0.80))

    # fuzzy ratio >= 0.90
    ratio = difflib.SequenceMatcher(None, normalize_text(g), normalize_text(p)).ratio()
    fuzz_ok = ratio >= 0.90
    sigs.append(_mk("fuzzy_ratio>=0.90", fuzz_ok, 0.75, ratio=ratio))

    # LLM agreement
    sys = "You are a strict but fair judge for short QA on website screenshots."
    prompt = f"""
Question: {q}

Gold answer: {g}
Model answer: {p}

Does the Model answer semantically match the Gold answer? 
Allow minor casing/punctuation, synonyms, and numeral word equivalents.
For lists, require the same items (order not required). For numbers/percents/times, allow equivalent forms.

Output exactly one token: TRUE or FALSE.
"""

    if api_key:
        llm_raw = get_chat_response(prompt, api_key, system_prompt=sys)
        llm_ok = (llm_raw.strip().upper().startswith("T"))
        sigs.append(_mk("llm_agree", llm_ok, 0.70, llm_raw=llm_raw.strip()[:160]))
    else:
        llm_ok = False
        sigs.append(_mk("llm_agree", False, 0.0, llm_raw="(no api key)"))

    # Decide primary: highest-weight heuristic that passed; else LLM if passed; else none
    heur_pass = [s for s in sigs if s.name != "llm_agree" and s.verdict]
    heur_pass.sort(key=lambda s: (-s.weight, s.name))
    if heur_pass:
        decision = "heuristic"
        primary = heur_pass[0].name
        match = True
    elif llm_ok:
        decision = "llm"
        primary = "llm_agree"
        match = True
    else:
        decision = "none"
        primary = "none"
        match = False

    # Composite score: softmax(weights) · 1{triggered}
    ws = [s.weight for s in sigs]
    wmax = max(ws) if ws else 0.0
    exps = [math.exp(w - wmax) for w in ws] if ws else [1.0]
    Z = sum(exps) or 1.0
    probs = [e / Z for e in exps]
    composite = float(sum(p_i * (1.0 if s.verdict else 0.0) for p_i, s in zip(probs, sigs)))

    return {
        "match": match,
        "decision": decision,
        "primary_reason": primary,
        "reasons": [asdict(s) for s in sigs],
        "used_llm": True if api_key else False,
        "composite": composite,
    }


if __name__ == "__main__":
    import argparse, json, os, sys, io, urllib.request, tempfile, time
    from typing import Dict, List

    try:
        from PIL import Image
    except Exception:
        Image = None

    def _open_image(src: str):
        """Return a PIL.Image if possible (supports file path or URL), else None."""
        if Image is None:
            return None
        try:
            if src.startswith("http://") or src.startswith("https://"):
                with urllib.request.urlopen(src, timeout=10) as r:
                    b = r.read()
                return Image.open(io.BytesIO(b))
            else:
                return Image.open(src)
        except Exception as e:
            print(f"[warn] could not open image '{src}': {e}")
            return None

    def _show_image(img, title=None):
        if img is None:
            return
        try:
            if title and hasattr(img, "show"):
                img.show(title=title)
            else:
                img.show()
        except Exception as e:
            print(f"[warn] could not show image: {e}")

    def _load_examples(path: str | None) -> List[Dict]:
        """Examples must have keys: question, answer, and either image or image_url."""
        if not path:
            return [
                {
                    "question": "What’s the listed price?",
                    "answer": "$3.99",
                    "image_url": "https://picsum.photos/seed/qa_price/900/550",
                },
                {
                    "question": "Which cards are accepted?",
                    "answer": "Visa, MasterCard",
                    "image_url": "https://picsum.photos/seed/qa_cards/900/550",
                },
                {
                    "question": "Primary button label?",
                    "answer": "Click Innovation in the main navigation near the top.",
                    "image_url": "https://picsum.photos/seed/qa_btn/900/550",
                },
            ]

        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                if isinstance(data, dict):
                    # allow {"examples":[...]}
                    data = data.get("examples", [])
                return data

    ap = argparse.ArgumentParser(
        description="Interactive test for Screenshot QA judge."
    )
    ap.add_argument(
        "--examples",
        type=str,
        default=None,
        help="JSON or JSONL with fields: question, answer, image|image_url",
    )
    ap.add_argument("--n", type=int, default=None, help="limit #examples")
    ap.add_argument(
        "--no-show", action="store_true", help="do not open image viewer windows"
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="sleep seconds after showing each image",
    )
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "[note] OPENAI_API_KEY not set — LLM fallback will be unavailable; only heuristics can pass.\n"
        )

    examples = _load_examples(args.examples)
    if args.n is not None:
        examples = examples[: max(1, args.n)]

    total = 0
    correct = 0
    reason_primary_counts: Dict[str, int] = {}

    print(
        f"Loaded {len(examples)} example(s). Answer the question after the image preview.\n"
    )

    for i, ex in enumerate(examples, 1):
        q = ex.get("question", "")
        g = ex.get("answer", "")
        src = ex.get("image") or ex.get("image_url") or ""
        print("=" * 80)
        print(f"[{i}] Q: {q}")
        if src:
            print(f"    image: {src}")
            if not args.no_show:
                _show_image(_open_image(src), title=f"Example {i}")
                if args.pause > 0:
                    time.sleep(args.pause)
        else:
            print("    (no image provided)")

        user_pred = input("Your answer: ").strip()
        total += 1

        try:
            res = judge_equivalence(q, g, user_pred, api_key or "")
        except Exception as e:
            print(f"[error] judge_equivalence failed: {e}")
            res = {
                "match": False,
                "reasons": [],
                "decision": "error",
                "primary_reason": "error",
            }

        ok = bool(res.get("match"))
        correct += int(ok)
        primary = res.get("primary_reason", "")
        if primary:
            reason_primary_counts[primary] = reason_primary_counts.get(primary, 0) + 1

        print(
            f" → {'✓ MATCH' if ok else '✗ NO MATCH'} | decision={res.get('decision')} | primary_reason={primary}"
        )
        # compact reason chips
        reasons = res.get("reasons", [])
        if reasons:
            chips = " ".join(
                f"{r.get('name', '?')}{'✅' if r.get('verdict') else '❌'}"
                for r in reasons
            )
            print(f"    reasons: {chips}")
            # show a little evidence for helpful ones
            for r in reasons:
                nm = r.get("name")
                if nm == "numbers_equal":
                    print(
                        f"      - numbers_equal evidence: gold={r['details'].get('gold_numbers')} pred={r['details'].get('pred_numbers')}"
                    )
                if nm == "list_set_equal":
                    print(
                        f"      - list_set_equal evidence: gold={r['details'].get('gold_items')} pred={r['details'].get('pred_items')}"
                    )
                if nm == "fuzzy_ratio>=0.90":
                    print(f"      - fuzzy ratio: {r['details'].get('ratio'):.3f}")
        print(f"    gold: {g}\n")

    acc = (correct / total) if total else 0.0
    print("=" * 80)
    print(f"Done. Accuracy (you vs gold): {correct}/{total} = {acc:.3f}")
    if reason_primary_counts:
        print("Primary reason counts:")
        for k, v in sorted(reason_primary_counts.items(), key=lambda kv: -kv[1]):
            print(f"  - {k}: {v}")
