"""
Escalation evaluation on REAL data.

HONESTY NOTE (important): unlike the synthetic version, we do NOT have
an independent "should this have escalated" human label for real
threads — nobody annotated the real dataset for this, and we (the
assignment author) are not a real AppleSupport agent who can say what
actually should have happened next. Computing precision/recall against
a label WE generated from the same rules being tested would be circular
and worthless.

Instead we did what a real team would do first, before collecting
proper labels: a MANUAL AUDIT.
  1. Read all 35 messages the rule currently flags as escalate (100% of
     positives) and judged whether each is a reasonable escalation call.
  2. Searched the auto-handle bucket for likely misses using a broader,
     looser keyword net than the actual rules use (words like "cancel,"
     "hate," "terrible," "mad," "stupid," "useless") and read all matches.
This gives real, if informal, evidence of precision (from step 1) and a
lower-bound sense of recall gaps (from step 2) — not a certified
precision/recall number. See report.md Section 3 for the actual findings
and what we changed as a result (added "terrible experience" pattern).
"""
import pandas as pd
from src.escalate import decide_escalation


def audit_positives(golden_csv="eval/golden_set_real_with_predictions.csv"):
    df = pd.read_csv(golden_csv)
    decisions = df.apply(
        lambda r: decide_escalation(r["customer_text"], r["predicted_intent"], r["confidence"]), axis=1
    )
    df["rule_escalate"] = decisions.apply(lambda d: d["escalate"])
    df["rule_reason"] = decisions.apply(lambda d: d["reason"])

    positives = df[df["rule_escalate"]]
    print(f"=== ESCALATION AUDIT: {len(positives)}/{len(df)} messages flagged ({len(positives)/len(df):.1%}) ===")
    print("\nBy reason:")
    print(positives["rule_reason"].value_counts())

    print(f"\nAll {len(positives)} flagged messages (manually reviewed — see report.md for verdict):")
    for _, r in positives.iterrows():
        print(f"  [{r['rule_reason']:28s}] {r['customer_text'][:100]}")

    df.to_csv("eval/escalation_audit_real.csv", index=False)
    return df


def find_likely_misses(audited_csv="eval/escalation_audit_real.csv"):
    df = pd.read_csv(audited_csv)
    auto = df[~df["rule_escalate"]]
    loose_net = r"cancel|delete my account|never buy|hate|awful|terrible|angry|\bmad\b|disgust|refuse|stupid|garbage|useless"
    suspicious = auto[auto["customer_text"].str.contains(loose_net, case=False, regex=True, na=False)]
    print(f"\n=== LIKELY-MISS SCAN: {len(suspicious)} auto-handled messages matched a broader distress "
          f"keyword net not in the actual rules ===")
    for _, r in suspicious.iterrows():
        print(f"  [{r['intent']}] {r['customer_text'][:100]}")
    return suspicious


if __name__ == "__main__":
    audit_positives()
    find_likely_misses()
