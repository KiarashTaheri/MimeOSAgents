"""Deterministic safety layer around the model: the agent explains data, it never gives medical advice.

Three checks, all plain code so they work the same on any model and can't be talked out of:

1. `check_question`  runs BEFORE the model. Dosing, medication, diagnosis, symptom and prompt-injection
                     requests get a fixed, pre-written reply; the model is never called for them.
2. `check_answer`    runs AFTER the model. If an answer still contains a dose, a treatment instruction or
                     a diagnosis, it is replaced with a safe reply.
3. `glucose_alerts`  looks at the data itself. An urgent latest reading, or 14-day numbers outside the
                     consensus targets, adds a note telling the user to contact a healthcare professional.

Rules are deliberately conservative: a harmless question may get a referral instead of an answer,
but a dosing question must never reach the model.
"""

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel

from . import cgm

BlockReason = Literal["symptoms", "dosing", "medication", "diagnosis", "prompt_injection", "unsafe_answer"]

# ------------------------------------------------------------------------------------------- replies

_DATA_OFFER = "I can still show you what your CGM data says, for example your time in range, lows or highs."

REPLIES: dict[BlockReason, str] = {
    "symptoms": (
        "I can't assess symptoms or tell you how to treat them. If you feel unwell, follow the plan your care "
        "team gave you. If symptoms are severe (confusion, fainting, a seizure, chest pain or trouble "
        "breathing), call your local emergency number now. Otherwise, please contact a healthcare "
        "professional as soon as you can."
    ),
    "dosing": (
        "I can't give advice on insulin or medication doses. Dosing depends on your full medical history, so "
        f"please ask your doctor, diabetes educator or care team. {_DATA_OFFER}"
    ),
    "medication": (
        "I can't recommend starting, stopping or changing any medication. Please talk to your doctor or "
        f"pharmacist about that. {_DATA_OFFER}"
    ),
    "diagnosis": (
        "I can't diagnose conditions. A diagnosis needs a healthcare professional and lab tests such as an "
        "HbA1c. If you're concerned, please book an appointment with your doctor; a summary of your CGM data "
        "could be useful to bring along."
    ),
    "prompt_injection": (
        "I can only help you understand your own CGM data, and I can't change how I work or give medical "
        f"advice. {_DATA_OFFER}"
    ),
    "unsafe_answer": (
        "I can't help with that because it would mean giving medical advice. Please talk to your doctor or "
        f"care team. {_DATA_OFFER}"
    ),
}

# ------------------------------------------------------------------------------------------- patterns

_TREATMENT = (
    r"insulin|units?|iu|dose|dosage|dosing|bolus|basal|correction|carb ratio|"
    r"medication|medicine|meds?|pills?|tablets?|drugs?|injections?|shots?"
)
_MEDICATIONS = (
    r"metformin|ozempic|wegovy|semaglutide|mounjaro|zepbound|tirzepatide|trulicity|dulaglutide|victoza|"
    r"liraglutide|glp-?1|jardiance|empagliflozin|farxiga|dapagliflozin|sglt-?2|januvia|sitagliptin|"
    r"glipizide|gliclazide|glimepiride|glyburide|sulfonylureas?|pioglitazone|lantus|levemir|tresiba|toujeo|"
    r"basaglar|humalog|novolog|novorapid|fiasp|lyumjev|apidra|glucagon|baqsimi|statins?|aspirin"
)
_CHANGE_VERBS = (
    r"take|inject|give|use|dose|bolus|increase|decrease|raise|lower|reduce|adjust|change|skip|stop|double|start|switch"
)

_QUESTION_PATTERNS: list[tuple[BlockReason, re.Pattern[str]]] = [
    ("symptoms", re.compile(
        r"\b(shak(y|ing)|trembl\w*|sweat(y|ing)|dizz(y|iness)|light-?headed|confus\w*|faint\w*|pass(ed|ing)? out|"
        r"seizures?|unconscious|chest pain|can'?t breathe|short(ness)? of breath|vomit\w*|throwing up|"
        r"ketones?|dka|blurr?(y|ed) vision|palpitations?|heart (is )?racing|feel (very |really )?(sick|unwell|ill))\b"
    )),
    ("dosing", re.compile(
        rf"\b(how (many|much)|what|which|right|correct|enough)\b.{{0,40}}\b({_TREATMENT})\b"
        rf"|\b(should|can|could|do|must) i\b.{{0,40}}\b({_CHANGE_VERBS})\b.{{0,40}}\b({_TREATMENT}|{_MEDICATIONS})\b"
        r"|\b(insulin|correction) (to carb )?(ratio|factor)\b"
        # Words that only come up when asking about treatment block on their own.
        r"|\b(bolus|basal|dose|doses|dosage|dosing|units?|iu|correction (dose|shot))\b"
    )),
    ("medication", re.compile(
        rf"\b({_MEDICATIONS})\b|\b(prescribe|prescription|medication|medicine|meds)\b"
    )),
    ("diagnosis", re.compile(
        r"\bdiagnos\w*|\b(do|could|might|would) i (have|get)\b|\bam i (pre-?)?diabetic\b"
        r"|\b(is|are) (this|it|that|these|my \w+) (a sign of|signs of|symptoms of)\b"
        r"|\b(is|does) (this|it|that) mean i (have|am)\b|\bwhat (condition|disease|illness)\b"
    )),
    ("prompt_injection", re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(instructions?|rules?|prompt|guidelines?|above|previous)\b"
        r"|\byou are now\b|\bpretend (to be|you are)\b|\b(act|role-?play) as\b|\bsystem prompt\b"
        r"|\bdeveloper mode\b|\bjailbreak\b|\bno (restrictions|rules|filters)\b"
    )),
]

# Applied to the model's answer. A referral ("ask your doctor about adjusting your insulin") is allowed.
_ANSWER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d+(\.\d+)?\s*(units?|iu)\b"),  # a dose amount
    re.compile(rf"\b({_CHANGE_VERBS})\b.{{0,40}}\b({_TREATMENT}|{_MEDICATIONS})\b"),  # a treatment instruction
    re.compile(rf"\b({_MEDICATIONS})\b"),  # naming a specific drug
    re.compile(r"\b(you (have|are|likely have|may have|might have)|this (is|indicates|suggests|means))\b.{0,20}"
               r"\b(pre-?diabet\w*|diabet\w*|hypoglycemi\w*|hyperglycemi\w*|insulin resistan\w*|gastroparesis)\b"),
    re.compile(r"\bi (diagnose|prescribe|recommend (taking|starting|stopping|increasing|decreasing))\b"),
]
_REFERRAL_CONTEXT = re.compile(
    r"\b(doctor|provider|clinician|care team|professional|pharmacist|educator|physician|nurse)\b"
)
_REFERRAL_WINDOW_CHARS = 80


def _normalize(text: str) -> str:
    """Lower-case, unify look-alike characters and whitespace so 'Ｉｎｓｕｌｉｎ' or 'how  many' still match."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text)


def check_question(question: str) -> BlockReason | None:
    """Return why a question must not reach the model, or None if it's a data question."""
    text = _normalize(question)
    for reason, pattern in _QUESTION_PATTERNS:  # ordered: symptoms first, they are the most urgent
        if pattern.search(text):
            return reason
    return None


def check_answer(answer: str) -> bool:
    """True if the answer contains medical advice (a dose, a treatment instruction or a diagnosis)."""
    text = _normalize(answer)
    for pattern in _ANSWER_PATTERNS:
        for match in pattern.finditer(text):
            preceding = text[max(0, match.start() - _REFERRAL_WINDOW_CHARS) : match.start()]
            if not _REFERRAL_CONTEXT.search(preceding):
                return True
    return False


# ------------------------------------------------------------------------------------------- alerts

# Consensus targets for most adults with type 1 or type 2 diabetes (Battelino et al., 2019), over 14 days.
ALERT_WINDOW_DAYS = 14
TARGET_MAX_VERY_LOW_PCT = 1.0  # time below 3.0 mmol/L
TARGET_MAX_LOW_PCT = 4.0  # time below 3.9 mmol/L (includes below 3.0)
TARGET_MIN_IN_RANGE_PCT = 70.0
TARGET_MAX_HIGH_PCT = 25.0  # time above 10.0 mmol/L (includes above 13.9)
TARGET_MAX_VERY_HIGH_PCT = 5.0  # time above 13.9 mmol/L


class SafetyAlert(BaseModel):
    level: Literal["urgent", "advisory"]
    code: str
    message: str


def glucose_alerts(readings: list[cgm.GlucoseReading]) -> list[SafetyAlert]:
    """Alerts that come from the data, independent of the question and of the model."""
    alerts: list[SafetyAlert] = []
    latest = readings[-1]
    when = f"{latest.timestamp:%a %H:%M}"
    if latest.mmol_l < cgm.VERY_LOW:
        alerts.append(SafetyAlert(level="urgent", code="latest_very_low", message=(
            f"Your most recent reading ({latest.mmol_l} mmol/L, {when}) is very low. Follow your care plan for "
            "a low now. If you feel confused, faint or very unwell, call your local emergency number."
        )))
    elif latest.mmol_l < cgm.LOW:
        alerts.append(SafetyAlert(level="urgent", code="latest_low", message=(
            f"Your most recent reading ({latest.mmol_l} mmol/L, {when}) is below range. Follow your care plan "
            "for a low, and contact a healthcare professional if it doesn't come back up or you feel unwell."
        )))
    elif latest.mmol_l > cgm.VERY_HIGH:
        alerts.append(SafetyAlert(level="urgent", code="latest_very_high", message=(
            f"Your most recent reading ({latest.mmol_l} mmol/L, {when}) is very high. Follow your care plan, "
            "and contact a healthcare professional if it stays high or you feel unwell."
        )))

    overall = cgm.time_in_range(readings, ALERT_WINDOW_DAYS).overall
    below_low = round(overall.very_low_pct + overall.low_pct, cgm.OUTPUT_DECIMALS)
    above_high = round(overall.high_pct + overall.very_high_pct, cgm.OUTPUT_DECIMALS)
    missed: list[str] = []
    if overall.very_low_pct > TARGET_MAX_VERY_LOW_PCT:
        missed.append(f"time below 3.0 mmol/L is {overall.very_low_pct}% (target under {TARGET_MAX_VERY_LOW_PCT:g}%)")
    if below_low > TARGET_MAX_LOW_PCT:
        missed.append(f"time below 3.9 mmol/L is {below_low}% (target under {TARGET_MAX_LOW_PCT:g}%)")
    if overall.in_range_pct < TARGET_MIN_IN_RANGE_PCT:
        missed.append(f"time in range is {overall.in_range_pct}% (target over {TARGET_MIN_IN_RANGE_PCT:g}%)")
    if above_high > TARGET_MAX_HIGH_PCT:
        missed.append(f"time above 10.0 mmol/L is {above_high}% (target under {TARGET_MAX_HIGH_PCT:g}%)")
    if overall.very_high_pct > TARGET_MAX_VERY_HIGH_PCT:
        missed.append(
            f"time above 13.9 mmol/L is {overall.very_high_pct}% (target under {TARGET_MAX_VERY_HIGH_PCT:g}%)"
        )
    if missed:
        alerts.append(SafetyAlert(level="advisory", code="outside_targets", message=(
            f"Over the last {ALERT_WINDOW_DAYS} days, some of your numbers are outside the recommended targets: "
            + "; ".join(missed)
            + ". Please review these with your healthcare provider."
        )))
    return alerts
