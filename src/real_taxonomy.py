"""
Intent taxonomy, RE-DERIVED from the real dataset (kaami #4 in the
original review: the old taxonomy was designed against synthetic
templates without ever looking at real text).

METHOD: we ran a keyword-frequency scan over all 73,571 real
AppleSupport customer messages (see decision_log.md item 16) before
naming a single intent. Two categories the synthetic version completely
missed showed up clearly in real data and are added here:
  - phishing_scam_report (~200+ messages: "is this a scam", "phishing
    email", "fake text message") — a real, distinct, and notably
    SECURITY-sensitive category that should route differently
  - photos_media_sync (~1,300+ messages: lost photos, iCloud Photos not
    syncing) — big enough and different enough from generic iCloud
    login issues to deserve its own bucket, not be folded into
    account_security

This module provides a KEYWORD-RULE weak labeler, used for two
different purposes with two different honesty bars:
  1. Bootstrapping SILVER labels for the large training pool (~73k
     messages) — weak supervision, never claimed to be human-verified.
  2. A first pass on golden-set candidates, which are then spot-checked
     by hand (see eval/build_golden_set_real.py for the verified
     subset and how disagreements were found) — NOT the same claim as
     "all 200 individually hand-labeled from scratch," which would be
     a different (stronger, slower) methodology. We are explicit about
     which one we did.
"""
import re

INTENTS = [
    "software_update_issue",
    "battery_performance",
    "device_hardware_issue",
    "account_security",
    "warranty_repair_request",
    "billing_subscription_refund",
    "photos_media_sync",
    "phishing_scam_report",
    "general_inquiry",  # catch-all: how-to, feature questions, sales/upgrade program, misc
]

# Ordered rules — first match wins. Order matters: phishing/scam and
# security checks are checked before generic account/software rules so
# a message like "is this email from Apple legit or a phishing scam"
# doesn't get swallowed by a generic "email/account" rule.
RULES = [
    ("phishing_scam_report", [
        r"\bphish", r"\bscam\b", r"\bfake (email|text|message)\b", r"is this (a scam|legit|real)",
        r"\bfraud", r"suspicious (email|text|link)",
    ]),
    ("account_security", [
        r"apple\s?id", r"\bicloud\b.*(login|password|sign in|locked|disabled)",
        r"\bpassword\b", r"two.?factor", r"\b2fa\b", r"can'?t sign in", r"account (disabled|locked|hacked)",
    ]),
    ("photos_media_sync", [
        r"\bphotos?\b.*(lost|missing|disappear|sync|deleted|gone)", r"lost (all )?(my )?photos",
        r"icloud photos?", r"photo library", r"\bsync(ing)?\b.*(photo|video)",
    ]),
    ("battery_performance", [
        r"\bbattery\b", r"won'?t (hold a )?charge", r"drain(s|ing)? (so )?fast", r"battery health",
    ]),
    ("device_hardware_issue", [
        r"won'?t turn on", r"black screen", r"screen (is )?(cracked|broken|black|frozen)",
        r"stuck (on|at)", r"unresponsive", r"dropped (it|my)", r"water damage", r"won'?t (respond|charge)",
        r"keeps? (rebooting|restarting|crashing|freezing)",
    ]),
    ("software_update_issue", [
        r"\bios\s?\d", r"\bupdate\b", r"software update", r"\bbug\b", r"after (the |updating to )?(ios|update)",
    ]),
    ("warranty_repair_request", [
        r"\bwarranty\b", r"applecare", r"\brepair\b", r"genius bar", r"service (center|provider)",
        r"replace(ment)? (device|phone|screen)",
    ]),
    ("billing_subscription_refund", [
        r"\brefund\b", r"\bcharged?\b", r"\bbilling\b", r"subscription", r"\binvoice\b",
        r"apple music.*(charge|bill)", r"icloud storage.*(charge|bill|pay)",
    ]),
]

GENERAL_FALLBACK = "general_inquiry"


def weak_label_intent(text: str) -> str:
    text = text.lower()
    for intent, patterns in RULES:
        if any(re.search(p, text) for p in patterns):
            return intent
    return GENERAL_FALLBACK


def clean_real_text(text: str) -> str:
    """Minimal cleaning for real Twitter noise (kaami #12: no cleaning
    step existed for real-data artifacts before)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://t\.co/\w+", "", text)  # strip t.co links (not human-readable content)
    text = re.sub(r"@\d{4,}", "", text)  # strip numeric anonymized handles like @115858
    text = text.replace("\\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/real_apple_pairs.csv")
    df["customer_text"] = df["customer_text"].apply(clean_real_text)
    df["brand_text"] = df["brand_text"].apply(clean_real_text)
    df = df[df["customer_text"].str.len() > 3].reset_index(drop=True)
    df["weak_intent"] = df["customer_text"].apply(weak_label_intent)
    print(df["weak_intent"].value_counts())
    df.to_csv("data/real_apple_pairs_labeled.csv", index=False)
