"""Guardrails for the OKF agents, built on Guardrails AI (https://guardrailsai.com).

Three guards, each a Guardrails `Guard` with a few validators:

    input guard      the router, before anything else      a user's message
    output guard     okf_agent's `check` node               qwen's answer
    untrusted guard  paper_scout, on every arXiv abstract   text we fetched from outside

How a Guard works: you give it validators, each with an `on_fail` action. `guard.validate(text)` runs
all of them on the text and returns a ValidationOutcome:
    .validated_output       the text after every "fix" action was applied
    .validation_summaries   one entry per validator that FAILED (name + reason)
The on_fail actions used here:
    "noop"   only report the failure, change nothing   -> we decide in code (block? log?)
    "fix"    replace the text with the validator's fix  -> e.g. a secret becomes ********
(others: "exception" raises, "filter"/"refrain" return nothing, "reask" asks the LLM again.)

Validators come from two places:
    Hub validators   pip packages, `pip install guardrails-ai-<name>`, import from guardrails_ai.<name>
                     ValidLength, SecretsPresent, DetectJailbreak (ML model, optional)
    Our own          written below with @register_validator: they know OKF things a hub validator
                     can't, like "which concept ids were in the context".

No LLM calls anywhere in here: every check is code or a small CPU model.

Settings (.env):
    OKF_GUARD_MAX_CHARS=2000             longest message the router accepts
    OKF_GUARD_JAILBREAK=1                use the DetectJailbreak model if it's installed (0 = regex only)
    OKF_GUARD_JAILBREAK_THRESHOLD=0.81   lower = stricter
    OKF_GUARD_DEVICE=cpu                 where DetectJailbreak runs (keep the GPU for qwen)
    OKF_GUARD_DROP_UNSOURCED=0           1 = also drop a sentence whose only citations were made up
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field

from guardrails import Guard, OnFailAction
from guardrails.validators import FailResult, PassResult, Validator, register_validator
from guardrails_ai.secrets_present import SecretsPresent
from guardrails_ai.valid_length import ValidLength

MAX_CHARS = int(os.getenv("OKF_GUARD_MAX_CHARS", "2000"))
USE_JAILBREAK_MODEL = os.getenv("OKF_GUARD_JAILBREAK", "1") != "0"
JAILBREAK_THRESHOLD = float(os.getenv("OKF_GUARD_JAILBREAK_THRESHOLD", "0.81"))
GUARD_DEVICE = os.getenv("OKF_GUARD_DEVICE", "cpu")
DROP_UNSOURCED = os.getenv("OKF_GUARD_DROP_UNSOURCED", "0") == "1"

REFUSAL = "The bundle does not contain this."          # okf_agent's own refusal sentence (okf.pipeline)
REMOVED = "[instruction-like text removed]"

# The replies the router sends when it blocks a message. Written by code, so they are always the same.
BLOCKED_REPLIES = {
    "too_long": ("That message is too long for me ({n:,} characters; the limit is {max:,}). "
                 "Could you send just the question?"),
    "empty": "I didn't get a question. Ask me about the papers in the bundle, or ask me to find papers on a topic.",
    "injection": ("I can't follow instructions that try to change how I work. Ask me a question about the "
                  "papers in the bundle, or ask me to find papers on a topic."),
}


# --- patterns -------------------------------------------------------------------------------------
# Prompt injection, the obvious kind. Each pattern needs the INSTRUCTION part ("previous instructions",
# "system prompt"), so "ignore the baseline in Table 2" or "forget GPT-3, what about GPT-4?" pass.
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\s+(all\s+|any\s+)?(of\s+)?(the\s+|your\s+|my\s+|these\s+)?"
    r"(previous|prior|above|earlier|preceding|original|initial|system)\s+(instructions?|prompts?|rules|guidelines|directions)\b",
    r"\b(ignore|disregard|bypass)\s+(all|any|your)\s+(instructions?|rules|guidelines|restrictions|safety)\b",
    r"\b(reveal|show|print|repeat|output|display|tell me|give me|leak|what is|what's|what are)\b[^.\n]{0,20}"
    r"\b(your|the)\s+(system|hidden|initial|original|secret|developer)\s+(prompt|instructions?|message|rules)\b",
    r"\byou are (now|no longer)\b[^.\n]{0,60}\b(unrestricted|unfiltered|jailbroken|free|without (any )?(rules|limits|restrictions)|dan)\b",
    r"\b(act|pretend|behave|roleplay)\b[^.\n]{0,30}\b(as|like|you are)\b[^.\n]{0,30}\b(dan|unrestricted|unfiltered|jailbroken|"
    r"no (rules|restrictions|filters|limits))\b",
    r"\b(developer|god|dan|jailbreak|unrestricted)\s+mode\b",
    r"\b(new|updated|real|actual)\s+instructions?\s*:",
    r"^\s*(system|assistant)\s*:",
    r"<\s*/?\s*(system|instructions?|im_start|im_end)\s*>",
]
INJECTION_RE = [re.compile(p, re.I | re.M) for p in INJECTION_PATTERNS]

# API keys / tokens detect-secrets doesn't know (it knows AWS, GitHub, Slack, private keys, ...).
API_KEY_RE = re.compile(
    r"\b(sk-ant-[A-Za-z0-9_\-]{20,}|sk-(proj-)?[A-Za-z0-9_\-]{20,}|hf_[A-Za-z0-9]{30,}|lsv2_(pt|sk)_[A-Za-z0-9_]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AIza[A-Za-z0-9_\-]{30,}|xox[abprs]-[A-Za-z0-9\-]{10,})\b")

CITATION_RE = re.compile(r"\[([^\[\]]+)\]")                    # same as okf.pipeline.CITATION_RE
NUMBER_RE = re.compile(r"\d+(?:[.,/:\-]\d+)*%?")               # same as report.NUMBER_RE
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[*\-])|\n+")    # where a sentence or line ends


# --- plain helpers (easy to test, and the validators below just call them) -----------------------
QUOTED_RE = re.compile(r"\"[^\"\n]{1,200}\"|“[^”\n]{1,200}”|(?<!\w)'[^'\n]{1,200}'(?!\w)")


def injection_matches(text: str) -> list[str]:
    """The pieces of `text` that look like an instruction to the model. Quoted text doesn't count:
    'What does the "ignore previous instructions" attack do?' MENTIONS an attack, it isn't one."""
    unquoted = QUOTED_RE.sub(" ", text or "")
    return [m.group(0) for r in INJECTION_RE for m in r.finditer(unquoted)]


def remove_instructions(text: str) -> str:
    """Untrusted text with every sentence that contains an instruction replaced by a marker."""
    pieces = SENTENCE_RE.split(text or "")
    out = [REMOVED if injection_matches(p) else p for p in pieces]
    return " ".join(p for p in out if p)


def citation_ids(group: str) -> list[str]:
    """'[a, b; c]' → ['a', 'b', 'c'] (the inside of one [...])."""
    return [part.strip() for part in re.split(r"[,;]", group) if part.strip()]


def looks_like_concept_id(cid: str) -> bool:
    return "/" in cid and " " not in cid                     # papers/gpt-4/..., topics/rlhf


def repair_id(cid: str, allowed_ids: set[str]) -> str | None:
    """A near miss → the real id. qwen sometimes drops a path segment:
    papers/gpt-4-technical-report/3-1-model-mitigations-part-1
    → papers/gpt-4-technical-report/system-card/3-1-model-mitigations-part-1
    Repaired only if exactly ONE real id has the same first and last segment."""
    first, last = cid.split("/")[0], cid.rsplit("/", 1)[-1]
    matches = [a for a in allowed_ids if a.split("/")[0] == first and a.rsplit("/", 1)[-1] == last]
    return matches[0] if len(matches) == 1 else None


def strip_fake_citations(text: str, allowed_ids: set[str], drop_sentences: bool = DROP_UNSOURCED
                         ) -> tuple[str, list[str], dict, int]:
    """Citations that look like concept ids but were not in the context: repaired if they're a near miss,
    removed otherwise. With drop_sentences, a sentence whose ONLY citations were fake is dropped too.
    Returns (new text, removed ids, {repaired id: real id}, number of sentences dropped)."""
    removed: list[str] = []
    repaired: dict = {}

    def fix(cid: str) -> str | None:
        if cid in allowed_ids or not looks_like_concept_id(cid):
            return cid
        real = repair_id(cid, allowed_ids)
        if real:
            repaired[cid] = real
        elif cid not in removed:
            removed.append(cid)
        return real

    def fix_group(m: re.Match) -> str:
        ids = citation_ids(m.group(1))
        if not any(looks_like_concept_id(c) for c in ids):
            return m.group(0)                                  # not a citation: [1], [sic], [Table 2]
        keep = [k for k in (fix(c) for c in ids) if k]
        return f"[{', '.join(dict.fromkeys(keep))}]" if keep else ""

    dropped = 0
    out = []
    for piece in re.split(r"(?<=[.!?])(\s+)", text):           # keep the whitespace between sentences
        new_piece = CITATION_RE.sub(fix_group, piece)
        cited_before = any(looks_like_concept_id(c) for g in CITATION_RE.findall(piece) for c in citation_ids(g))
        cited_after = any(looks_like_concept_id(c) for g in CITATION_RE.findall(new_piece) for c in citation_ids(g))
        if drop_sentences and cited_before and not cited_after:   # it cited, but nothing it cited was real
            dropped += 1
            continue
        out.append(new_piece)
    new = re.sub(r"[ \t]+([.,;:!?])", r"\1", "".join(out))      # "62.0% ." → "62.0%."
    new = re.sub(r"[ \t]{2,}", " ", new).strip()
    return new, removed, repaired, dropped


def unsupported_numbers(text: str, context: str) -> list[str]:
    """Numbers in the answer that appear nowhere in the context (citations and 1-digit numbers ignored:
    "[papers/x/8-4-...]" and list numbering aren't claims)."""
    plain = CITATION_RE.sub(" ", text)
    out = []
    for m in NUMBER_RE.finditer(plain):
        n = m.group(0)
        if m.start() and (plain[m.start() - 1].isalpha() or plain[m.start() - 1] == "-"):
            continue                                           # part of a name: GPT-3.5, Llama-2, Q4
        core = n.rstrip("%").rstrip(".,")
        if len(core.replace(".", "").replace(",", "")) <= 1:
            continue
        if core not in context and n not in out:
            out.append(n)
    return out


# --- our own validators ---------------------------------------------------------------------------
# A validator = a class with validate(value, metadata) -> PassResult | FailResult(error_message, fix_value).
# @register_validator gives it a name; `metadata` is whatever we pass to guard.validate(..., metadata={...}).
@register_validator(name="okf/injection_phrases", data_type="string")
class InjectionPhrases(Validator):
    """Fails if the text contains an instruction aimed at the model. Fix: remove those sentences."""

    def validate(self, value, metadata):
        found = injection_matches(value)
        if not found:
            return PassResult()
        return FailResult(error_message=f"instruction-like text: {found[0]!r}", fix_value=remove_instructions(value))


@register_validator(name="okf/api_keys", data_type="string")
class ApiKeys(Validator):
    """Fails if the text contains an API key. Fix: replace it with ********."""

    def validate(self, value, metadata):
        if not API_KEY_RE.search(value):
            return PassResult()
        return FailResult(error_message="API key in the text", fix_value=API_KEY_RE.sub("********", value))


@register_validator(name="okf/citations_in_context", data_type="string")
class CitationsInContext(Validator):
    """Fails if the answer cites a concept id that wasn't in the context (metadata["allowed_ids"]).
    Fix: repair near misses, remove the rest (and with OKF_GUARD_DROP_UNSOURCED=1, drop sentences
    that only had made-up citations)."""

    def validate(self, value, metadata):
        new, removed, repaired, dropped = strip_fake_citations(value, set(metadata.get("allowed_ids", [])))
        if not removed and not repaired:
            return PassResult()
        return FailResult(error_message=f"made-up citations: removed {removed}, repaired {list(repaired)}, "
                                        f"{dropped} sentence(s) dropped",
                          fix_value=new or REFUSAL)


@register_validator(name="okf/numbers_in_context", data_type="string")
class NumbersInContext(Validator):
    """Fails if the answer has a number that's nowhere in the context (metadata["context"]). No fix: a
    regex can't tell a garbled number from a correct paraphrase ("about a third"), so we only report it."""

    def validate(self, value, metadata):
        bad = unsupported_numbers(value, metadata.get("context", ""))
        if not bad:
            return PassResult()
        return FailResult(error_message=f"numbers not in the context: {', '.join(bad)}")


# --- building the guards (once, on first use: DetectJailbreak loads models) ------------------------
def _guard(name: str, *validators) -> Guard:
    # NOTE: pass ALL validators to ONE .use() call. In guardrails-ai 0.11, .use(a).use(b) keeps only b.
    g = Guard(name=name).use(*validators)
    g.configure(allow_metrics_collection=False)        # Guardrails sends anonymous usage metrics by default
    return g


class _Guards:
    def __init__(self):
        self._lock = threading.Lock()
        self.input = self.output = self.untrusted = None
        self.jailbreak = None                                   # the DetectJailbreak validator, if on
        self.jailbreak_status = "not loaded yet"

    def load(self) -> "_Guards":
        with self._lock:
            if self.input is not None:
                return self
            validators = [
                ValidLength(min=1, max=MAX_CHARS, on_fail=OnFailAction.NOOP),   # hub: block in code, don't truncate
                InjectionPhrases(on_fail=OnFailAction.NOOP),                    # ours: block in code
                SecretsPresent(on_fail=OnFailAction.FIX),                       # hub: AWS/GitHub/... keys → ****
                ApiKeys(on_fail=OnFailAction.FIX),                              # ours: sk-, hf_, lsv2_ ... → ****
            ]
            jailbreak = _load_jailbreak()
            if isinstance(jailbreak, Validator):
                validators.append(jailbreak)
                self.jailbreak = jailbreak
                self.jailbreak_status = f"on (threshold {JAILBREAK_THRESHOLD}, {GUARD_DEVICE})"
            else:
                self.jailbreak_status = jailbreak                               # why it's off
            self.input = _guard("okf_input", *validators)
            self.output = _guard("okf_output", CitationsInContext(on_fail=OnFailAction.FIX),
                                 NumbersInContext(on_fail=OnFailAction.NOOP))
            self.untrusted = _guard("okf_untrusted", InjectionPhrases(on_fail=OnFailAction.FIX))
            return self


def _load_jailbreak():
    """The DetectJailbreak validator, or a string saying why it isn't used."""
    if not USE_JAILBREAK_MODEL:
        return "off (OKF_GUARD_JAILBREAK=0)"
    try:
        from guardrails_ai.detect_jailbreak import DetectJailbreak
    except ImportError:
        return "off (not installed: pip install guardrails-ai-detect-jailbreak)"
    try:
        return DetectJailbreak(threshold=JAILBREAK_THRESHOLD, device=GUARD_DEVICE, use_local=True,
                               on_fail=OnFailAction.NOOP)
    except Exception as exc:                          # no internet for the first download, ...
        print(f"guards: DetectJailbreak didn't load, using the regex only: {type(exc).__name__}: {exc}")
        return f"off (failed to load: {type(exc).__name__})"


GUARDS = _Guards()


def guard_status() -> dict:
    GUARDS.load()
    return {"max_chars": MAX_CHARS, "jailbreak_model": GUARDS.jailbreak_status,
            "drop_unsourced_sentences": DROP_UNSOURCED}


def jailbreak_scores(texts: list[str]) -> list[dict]:
    """DetectJailbreak's three sub-scores per text (for choosing a threshold). [] if the model is off.
    known_attack: similarity to known jailbreak prompts; saturation_attack: prompts stuffed with
    junk; other_attack: a fine-tuned classifier. The validator fails a text if the MAX > threshold."""
    jb = GUARDS.load().jailbreak
    return jb.predict_jailbreak(list(texts), reduction_function=None) if jb else []


# --- what the agents call ------------------------------------------------------------------------
@dataclass
class InputCheck:
    text: str                                  # the message to use (secrets replaced by ********)
    blocked: str | None = None                 # None, "too_long", "empty" or "injection"
    reply: str | None = None                   # the reply to send instead, if blocked
    failed: dict = field(default_factory=dict)  # {validator: reason} for everything that failed
    ms: float = 0.0


def check_input(text: str) -> InputCheck:
    t0 = time.perf_counter()
    text = text or ""
    if not text.strip():
        return InputCheck(text=text, blocked="empty", reply=BLOCKED_REPLIES["empty"])
    outcome = GUARDS.load().input.validate(text)
    failed = {s.validator_name: s.failure_reason for s in outcome.validation_summaries}
    blocked = None
    if "ValidLength" in failed:
        blocked = "too_long"
    elif "InjectionPhrases" in failed or "DetectJailbreak" in failed:
        blocked = "injection"
    reply = BLOCKED_REPLIES[blocked].format(n=len(text), max=MAX_CHARS) if blocked else None
    return InputCheck(text=(outcome.validated_output or text).strip(), blocked=blocked, reply=reply, failed=failed,
                      ms=round((time.perf_counter() - t0) * 1000, 1))


@dataclass
class OutputCheck:
    text: str                                          # the answer to show
    removed_citations: list[str] = field(default_factory=list)
    repaired_citations: dict = field(default_factory=dict)      # {made-up id: real id}
    dropped_sentences: int = 0
    unsupported_numbers: list[str] = field(default_factory=list)
    failed: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"removed_citations": self.removed_citations, "repaired_citations": self.repaired_citations,
                "dropped_sentences": self.dropped_sentences,
                "unsupported_numbers": self.unsupported_numbers}


def check_answer(answer: str, allowed_ids: list[str], context: str) -> OutputCheck:
    """okf_agent's answer: made-up citations removed, unsupported numbers reported."""
    if not answer or answer.strip().startswith(REFUSAL):
        return OutputCheck(text=answer)
    outcome = GUARDS.load().output.validate(answer, metadata={"allowed_ids": list(allowed_ids), "context": context})
    failed = {s.validator_name: s.failure_reason for s in outcome.validation_summaries}
    # The Guard did the fixing; the helpers tell us WHAT was fixed (cheap: plain regex).
    _, removed, repaired, dropped = strip_fake_citations(answer, set(allowed_ids))
    numbers = unsupported_numbers(answer, context) if "NumbersInContext" in failed else []
    return OutputCheck(text=(outcome.validated_output or REFUSAL).strip(), removed_citations=removed,
                       repaired_citations=repaired, dropped_sentences=dropped, unsupported_numbers=numbers,
                       failed=failed)


def clean_untrusted(text: str) -> tuple[str, bool]:
    """Text from outside (an arXiv abstract): sentences that give the model instructions removed.
    Returns (text, whether anything was removed)."""
    if not text or not injection_matches(text):              # fast path: almost every abstract
        return text, False
    outcome = GUARDS.load().untrusted.validate(text)
    return outcome.validated_output or text, True


__all__ = ["check_input", "check_answer", "clean_untrusted", "guard_status", "jailbreak_scores", "InputCheck", "OutputCheck",
           "BLOCKED_REPLIES", "injection_matches", "strip_fake_citations", "unsupported_numbers"]
