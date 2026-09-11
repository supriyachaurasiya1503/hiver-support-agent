"""
Evaluates the escalation decision layer against the golden set's human
escalate labels. Uses the golden set's classifier confidence (already
computed by src/classify.py into golden_set_with_predictions.csv) so
this is measuring the FULL pipeline's escalation call, not the rule
logic in isolation.
"""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.escalate import decide_escalation, CONFIDENCE_THRESHOLD


def evaluate_escalation(golden_csv="eval/golden_set_with_predictions.csv"):
    df = pd.read_csv(golden_csv)
    decisions = df.apply(
        lambda r: decide_escalation(r["text"], r["predicted_intent"], r["confidence"]),
        axis=1,
    )
    df["decided_escalate"] = decisions.apply(lambda d: d["escalate"])
    df["decided_reason"] = decisions.apply(lambda d: d["reason"])

    y_true = df["escalate"].astype(bool)
    y_pred = df["decided_escalate"].astype(bool)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"=== ESCALATION DECISION: golden set (n={len(df)}, threshold={CONFIDENCE_THRESHOLD}) ===")
    print(f"Precision: {precision:.3f}  (of what we escalated, how much SHOULD have been escalated)")
    print(f"Recall:    {recall:.3f}  (of what SHOULD have been escalated, how much we caught)")
    print(f"F1:        {f1:.3f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")

    missed = df[(y_true) & (~y_pred)]
    print(f"\nFALSE NEGATIVES (should have escalated, didn't) — n={len(missed)}, this is the costly failure mode:")
    for _, r in missed.iterrows():
        print(f"  - [{r['intent']}] {r['text'][:90]}")

    false_pos = df[(~y_true) & (y_pred)]
    print(f"\nFALSE POSITIVES (escalated unnecessarily) — n={len(false_pos)}:")
    for _, r in false_pos.iterrows():
        print(f"  - [{r['decided_reason']}] {r['text'][:90]}")

    df.to_csv("eval/escalation_eval.csv", index=False)
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    evaluate_escalation()
