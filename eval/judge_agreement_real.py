"""
Judge-vs-human agreement on REAL data (40-example sample from
eval/judge_scores_real.csv), same honesty framing as
eval/judge_agreement.py: no ANTHROPIC_API_KEY here, so the "judge"
being checked is the heuristic stand-in, not a real LLM judge.

This is scored independently from the synthetic-data version — same
1-5 rubric (relevance, grounded, tone, actionable), read blind to the
heuristic's own output, on the actual real drafts in
eval/judge_scores_real.csv.
"""
import pandas as pd

# key = tweet_id, value = {relevance, grounded, tone, actionable}
# "grounded" is 5 for every row here because the extractive fallback
# makes draft == source verbatim by construction — see
# eval/evaluate_replies_real.py's docstring on why that specific
# dimension can't discriminate anything on this path. relevance and
# actionable are where real disagreement with the heuristic shows up.
HUMAN_SCORES_REAL = {
    2041503: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    1445561: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    1886836: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    2001105: {"relevance": 3, "grounded": 5, "tone": 4, "actionable": 2},
    1466875: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    361421:  {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    1335172: {"relevance": 1, "grounded": 5, "tone": 4, "actionable": 1},  # off-topic retrieval: playlists
    2057835: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    613165:  {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    2942177: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 4},
    1425144: {"relevance": 2, "grounded": 5, "tone": 4, "actionable": 2},  # message itself too vague to parse
    1150998: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    1747992: {"relevance": 2, "grounded": 5, "tone": 4, "actionable": 2},  # message lacks context (no image visible)
    1765086: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    2014990: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    2422484: {"relevance": 2, "grounded": 5, "tone": 4, "actionable": 2},  # doesn't address the scam/legit question
    647336:  {"relevance": 3, "grounded": 5, "tone": 4, "actionable": 2},
    1573642: {"relevance": 3, "grounded": 5, "tone": 4, "actionable": 3},
    2056532: {"relevance": 3, "grounded": 5, "tone": 4, "actionable": 2},
    335782:  {"relevance": 2, "grounded": 5, "tone": 4, "actionable": 2},
    1335339: {"relevance": 3, "grounded": 5, "tone": 4, "actionable": 3},
    956411:  {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    1173184: {"relevance": 3, "grounded": 5, "tone": 4, "actionable": 2},
    2799471: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 5},
    1305636: {"relevance": 1, "grounded": 5, "tone": 4, "actionable": 1},  # off-topic: Visual Voicemail vs Audio Books
    2067477: {"relevance": 2, "grounded": 5, "tone": 4, "actionable": 2},  # itunes desktop vs phone issue
    1456632: {"relevance": 3, "grounded": 5, "tone": 4, "actionable": 2},
    1905441: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    588105:  {"relevance": 3, "grounded": 5, "tone": 4, "actionable": 2},
    2942169: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    2238940: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 4},
    2321180: {"relevance": 1, "grounded": 5, "tone": 4, "actionable": 1},  # off-topic: calendar entries
    2295869: {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 4},
    1758941: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    1246461: {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    1700017: {"relevance": 3, "grounded": 5, "tone": 4, "actionable": 2},
    334923:  {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 3},
    352632:  {"relevance": 5, "grounded": 5, "tone": 4, "actionable": 4},
    1828303: {"relevance": 1, "grounded": 5, "tone": 4, "actionable": 1},  # off-topic: Online Sales team
    556081:  {"relevance": 4, "grounded": 5, "tone": 4, "actionable": 2},
}


def compute_agreement_real(judge_scores_csv="eval/judge_scores_real.csv"):
    df = pd.read_csv(judge_scores_csv)
    rows = []
    for _, r in df.iterrows():
        tid = r["tweet_id"]
        if tid not in HUMAN_SCORES_REAL:
            continue
        h = HUMAN_SCORES_REAL[tid]
        rows.append({
            "tweet_id": tid,
            "judge_relevance": r["relevance"], "human_relevance": h["relevance"],
            "judge_pass": (r["relevance"] >= 4 and r["grounded"] >= 4),
            "human_pass": (h["relevance"] >= 4 and h["grounded"] >= 4),
        })
    agree_df = pd.DataFrame(rows)
    exact = (agree_df["judge_relevance"] == agree_df["human_relevance"]).mean()
    pass_agreement = (agree_df["judge_pass"] == agree_df["human_pass"]).mean()
    human_pass_rate = agree_df["human_pass"].mean()
    judge_pass_rate = agree_df["judge_pass"].mean()

    print(f"n={len(agree_df)}")
    print(f"Exact-score agreement (relevance): {exact:.1%}")
    print(f"Pass/fail agreement: {pass_agreement:.1%}")
    print(f"Human pass rate: {human_pass_rate:.1%}  |  Heuristic judge pass rate: {judge_pass_rate:.1%}")
    agree_df.to_csv("eval/judge_human_agreement_real.csv", index=False)
    return agree_df


if __name__ == "__main__":
    compute_agreement_real()
