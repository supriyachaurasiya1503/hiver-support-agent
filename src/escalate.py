"""
Escalation decision: auto-handle vs. escalate to a human, with a stated
reason (required by the assignment).

DESIGN CHOICE (see decision_log.md): this is a rule-based layer, not a
learned classifier. With only ~35-40 true escalation examples even in a
234-example golden set, training a separate model would badly overfit;
rules built from reading actual failure patterns are more auditable and
directly editable by a support lead without retraining anything, which
matters more than a marginal accuracy gain at this data volume.

REAL-DATA UPDATE: patterns below were expanded after reviewing actual
AppleSupport messages (not just imagined phrasing) — see
decision_log.md item 19. Real users say "I'll sue" (often sarcastically
— we cannot distinguish sarcasm from genuine intent with regex, and say
so explicitly as a limitation), "unacceptable," "ridiculous," "ruined
my phone," and reference fraud/double-charges more than a generic
"lawyer" mention.

Triggers, in priority order (first match wins, reason is the trigger name):
  1. explicit_human_request   — "talk to a human", "not a bot", etc.
  2. legal_or_regulatory      — lawyer, sue, consumer protection, BBB, fraud
  3. repeated_contact         — "third time", "again", "still not fixed",
     OR (with thread history available) a customer with several prior
     unresolved contacts — see decide_escalation_with_history()
  4. high_severity_sentiment  — strong negative language + churn signal
  5. low_classifier_confidence— intent prediction confidence below threshold
  6. sensitive_intent_with_financial_ask — billing/refund requests over
     a $ threshold mentioned in text
  7. phishing_scam_report intent — ALWAYS escalates (security-sensitive
     category added after reviewing real data; an auto-agent should
     never be the one confirming or denying "is this a scam" — that
     needs a human/security team regardless of confidence)
If none match: auto_handle.
"""
import re

HUMAN_REQUEST_PATTERNS = [r"\bhuman\b", r"\breal person\b", r"\bnot a bot\b", r"\bactual person\b"]
LEGAL_PATTERNS = [
    r"\blawyer\b", r"\bsue\b", r"\blawsuit\b", r"consumer protection", r"\bbbb\b", r"\bftc\b",
    r"\bfraud(ulent)?\b", r"legal action",
]
REPEATED_CONTACT_PATTERNS = [
    r"third time", r"again\b.*contact", r"still not fixed", r"keep(s)? happening",
    r"already (told|said|reported)", r"no (progress|response)", r"hung up on",
]
HIGH_SEVERITY_PATTERNS = [
    r"furious", r"leave apple", r"switching to", r"never buying", r"absolutely done",
    r"worst (support|experience|company)", r"terrible experience", r"unacceptable", r"ridiculous", r"ruined my",
]
# money detection: handles $, £, €, "twice"/"double" charged phrasing
MONEY_PATTERN = re.compile(r"[\$£€]\s?(\d+(\.\d{1,2})?)")
DOUBLE_CHARGE_PATTERN = re.compile(r"charged? (twice|double)", re.IGNORECASE)

CONFIDENCE_THRESHOLD = 0.55  # see eval/threshold_sweep.py for how this was checked


def _matches_any(text, patterns):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def decide_escalation(text: str, intent: str, intent_confidence: float, money_threshold: float = 20.0):
    if _matches_any(text, HUMAN_REQUEST_PATTERNS):
        return {"escalate": True, "reason": "explicit_human_request",
                "detail": "Customer explicitly asked for a human agent."}
    if _matches_any(text, LEGAL_PATTERNS):
        return {"escalate": True, "reason": "legal_or_regulatory",
                "detail": "Message references legal action, fraud, or a regulatory body. "
                          "NOTE: regex cannot distinguish a genuine threat from sarcasm "
                          "(\"I'll sue you lol\") — false positives on sarcastic phrasing "
                          "are an accepted tradeoff given the cost of missing a real one."}
    if intent == "phishing_scam_report":
        return {"escalate": True, "reason": "security_sensitive_intent",
                "detail": "Phishing/scam verification should always be confirmed by a human "
                          "or security team, regardless of classifier confidence."}
    if _matches_any(text, REPEATED_CONTACT_PATTERNS):
        return {"escalate": True, "reason": "repeated_contact",
                "detail": "Customer indicates this is a repeat/unresolved contact."}
    if _matches_any(text, HIGH_SEVERITY_PATTERNS):
        return {"escalate": True, "reason": "high_severity_sentiment",
                "detail": "Strong negative sentiment with churn signal detected."}
    if intent_confidence < CONFIDENCE_THRESHOLD:
        return {"escalate": True, "reason": "low_classifier_confidence",
                "detail": f"Intent classifier confidence {intent_confidence:.2f} is below "
                          f"threshold {CONFIDENCE_THRESHOLD} — safer to have a human confirm intent."}
    if intent == "billing_subscription_refund":
        money_matches = [float(m.group(1)) for m in MONEY_PATTERN.finditer(text)]
        if any(m >= money_threshold for m in money_matches) or DOUBLE_CHARGE_PATTERN.search(text):
            return {"escalate": True, "reason": "sensitive_intent_with_financial_ask",
                    "detail": f"Billing intent with a stated amount >= {money_threshold} or a "
                              f"double-charge claim — refund authority above this is out of "
                              f"scope for the auto-agent."}
    return {"escalate": False, "reason": "auto_handle",
            "detail": "No escalation trigger matched; within auto-agent's scope."}


def decide_escalation_with_history(text: str, intent: str, intent_confidence: float,
                                    author_prior_contact_count: int = 0, repeat_threshold: int = 2,
                                    money_threshold: float = 20.0):
    """Same as decide_escalation, but also escalates on THREAD-LEVEL repeat
    contact (a customer who has messaged AppleSupport multiple times
    before), not just single-message keyword phrasing like "third time" —
    real customers rarely narrate their own contact count in the message
    itself (kaami #6: no multi-turn/history signal existed before this).
    """
    base = decide_escalation(text, intent, intent_confidence, money_threshold)
    if base["escalate"]:
        return base
    if author_prior_contact_count >= repeat_threshold:
        return {"escalate": True, "reason": "repeated_contact_history",
                "detail": f"Customer has contacted AppleSupport {author_prior_contact_count} "
                          f"times before (thread history), independent of this message's wording."}
    return base


if __name__ == "__main__":
    tests = [
        ("i want to speak to an actual human being, not a bot", "general_inquiry", 0.9),
        ("battery drains so fast, help please", "battery_performance", 0.95),
        ("this is the third time contacting you about this", "device_hardware_issue", 0.88),
        ("got charged $29.99 for something i never bought", "billing_subscription_refund", 0.9),
        ("is this email from apple or a phishing scam", "phishing_scam_report", 0.9),
        ("charged twice for my subscription this month", "billing_subscription_refund", 0.9),
    ]
    for text, intent, conf in tests:
        print(decide_escalation(text, intent, conf), "<-", text[:50])
