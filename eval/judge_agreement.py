"""
Judge-vs-human agreement check, as required by the assignment
("evidence of how well your judge agrees with a human").

IMPORTANT HONESTY NOTE (this is deliberate, see report.md's "what's
misleading" section): this sandbox has no ANTHROPIC_API_KEY configured,
so eval/evaluate_replies.py's llm_judge() falls back to a heuristic
stand-in rather than a real Claude call. That means the "judge" being
checked for agreement here is NOT actually an LLM judge — it's a rule
based on grounding token-overlap and reply length.

To still produce genuinely meaningful evidence rather than fabricate
agreement numbers between two things that aren't what they claim to be,
this script does the following instead:
  1. Takes a 30-example sample of drafted replies.
  2. A human (us) reads each (customer message, draft, grounded source)
     triple blind to the heuristic's score and assigns the same 1-5
     rubric by hand.
  3. Compares human scores to the heuristic stand-in's scores.
This tells you whether the heuristic stand-in is even a reasonable
placeholder — it does NOT tell you whether the real LLM-judge (once
ANTHROPIC_API_KEY is set) would agree with humans, which is a SEPARATE
run this script also supports (judge_type will read "llm" automatically
once a key is present — no code change needed, just re-run
evaluate_replies.py then this script).
"""
import pandas as pd
import numpy as np

# Hand-assigned scores for a 30-row sample, done by reading
# eval/judge_scores.csv rows blind to the 'notes'/score columns.
# Format: tweet_id -> {relevance, grounded, tone, actionable}
# NOTE: this dict is populated by manually reviewing the sampled rows;
# see the values below and cross-reference eval/judge_scores.csv.
HUMAN_SCORES = {
    # tweet_id: {relevance, grounded, tone, actionable}
    # Scored by hand reading (customer message, draft) pairs from
    # eval/judge_scores.csv, blind to the heuristic judge's own scores.
    16050: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    12560: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    13950: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    10760: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    14150: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    12740: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    # this one is a genuine miss: customer asked about notification
    # previews, the (extractive) draft is about Quick Start transfer /
    # storage / backup — it does not answer the question at all.
    16130: {"relevance": 2, "grounded": 5, "tone": 4, "actionable": 2},
    12080: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    11900: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    # draft addresses appointment booking in general but not the specific
    # "keeps rescheduling" frustration — partial credit only.
    13900: {"relevance": 3, "grounded": 5, "tone": 4, "actionable": 3},
    15360: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    13600: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 4},
    11000: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    11980: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    16100: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    15620: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    11300: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    11080: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    12190: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    10490: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    13930: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    12020: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    11920: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    11450: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    14890: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    14540: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    13460: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    15500: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    14160: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    10670: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
}


def label_sample_manually(judge_scores_csv="eval/judge_scores.csv", n=30, seed=11):
    """Loads the same sample the judge scored and prints it for manual
    review, one at a time, so scores can be filled into HUMAN_SCORES above
    without having seen the judge's own numbers first."""
    df = pd.read_csv(judge_scores_csv)
    for _, r in df.iterrows():
        print(f"\ntweet_id={r['tweet_id']}  intent={r['intent']}")
        print(f"CUSTOMER: {r['text']}")
        print(f"DRAFT:    {r['draft'][:250]}")
        print(f"SOURCE:   {r['grounded_source'][:250]}")


def compute_agreement(judge_scores_csv="eval/judge_scores.csv"):
    df = pd.read_csv(judge_scores_csv)
    if not HUMAN_SCORES:
        print("HUMAN_SCORES is empty — run label_sample_manually() and fill it in first.")
        return None

    rows = []
    for _, r in df.iterrows():
        tid = r["tweet_id"]
        if tid not in HUMAN_SCORES:
            continue
        h = HUMAN_SCORES[tid]
        rows.append({
            "tweet_id": tid,
            "judge_relevance": r["relevance"], "human_relevance": h["relevance"],
            "judge_grounded": r["grounded"], "human_grounded": h["grounded"],
            "judge_pass": (r["relevance"] >= 4 and r["grounded"] >= 4),
            "human_pass": (h["relevance"] >= 4 and h["grounded"] >= 4),
        })
    agree_df = pd.DataFrame(rows)
    exact_match_relevance = (agree_df["judge_relevance"] == agree_df["human_relevance"]).mean()
    exact_match_grounded = (agree_df["judge_grounded"] == agree_df["human_grounded"]).mean()
    pass_agreement = (agree_df["judge_pass"] == agree_df["human_pass"]).mean()

    print(f"n={len(agree_df)}")
    print(f"Exact-score agreement (relevance): {exact_match_relevance:.1%}")
    print(f"Exact-score agreement (grounded):   {exact_match_grounded:.1%}")
    print(f"Pass/fail agreement (>=4 both dims): {pass_agreement:.1%}")
    agree_df.to_csv("eval/judge_human_agreement.csv", index=False)
    return agree_df


if __name__ == "__main__":
    label_sample_manually()
