"""Reply quality evaluation on REAL data — see eval/evaluate_replies.py
for the original synthetic-data version.

METRIC FIX (found while running this on real data): grounding_overlap
as originally defined (token overlap between the DRAFT and its
retrieved SOURCE) is trivially 1.0 whenever the extractive fallback is
used, because the fallback literally returns the source verbatim by
construction — comparing a string to itself. This was a flaw in the
metric's design, not something real data happened to fix. We replace it
here with something that actually varies: retrieval_relevance, the
token overlap between the NEW customer message and the RETRIEVED
historical customer message it's being grounded on — this measures
whether BM25 found a topically similar past case, which is the thing
that can actually go wrong and does vary across examples.
"""
import pandas as pd
from src.draft_reply import GroundedRetriever, draft_reply
from src.ingest import load_brand_pairs


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(str(a).lower().split()), set(str(b).lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def run_real_reply_eval(golden_csv="eval/golden_set_real_with_predictions.csv",
                         data_csv="data/real_twcs.csv"):
    golden = pd.read_csv(golden_csv)
    pairs = load_brand_pairs(data_csv, "AppleSupport", top_level_only=True)
    # LEAKAGE FIX: exclude the golden set's own tweets from the retrieval
    # corpus. Without this, a golden example's own historical tweet is
    # sitting in the corpus and retrieves itself with relevance=1.0 —
    # this was silently inflating retrieval_relevance to ~0.97 on the
    # first run of this script, the same kind of mistake
    # src/classify_real.py already guards against for training data.
    golden_ids = set(golden["customer_tweet_id"].astype(str))
    pairs = pairs[~pairs["customer_tweet_id"].astype(str).isin(golden_ids)].reset_index(drop=True)
    retriever = GroundedRetriever(pairs)

    rows = []
    for _, r in golden.iterrows():
        result = draft_reply(r["customer_text"], retriever, k=3)
        top = result["grounding"][0] if result["grounding"] else {"customer_text": "", "brand_text": "", "score": 0.0}
        retrieval_relevance = token_overlap(r["customer_text"], top["customer_text"])
        rows.append({
            "tweet_id": r["customer_tweet_id"],
            "text": r["customer_text"],
            "intent": r["intent"],
            "draft": result["draft"],
            "method": result["method"],
            "bm25_score": round(top["score"], 3),
            "retrieval_relevance": round(retrieval_relevance, 3),
            "retrieved_customer_text": top["customer_text"],
            "grounded_source": top["brand_text"],
        })
    df = pd.DataFrame(rows)
    df.to_csv("eval/reply_drafts_real.csv", index=False)
    print(f"Drafted {len(df)} replies on real data")
    print(f"Mean retrieval_relevance (customer msg <-> retrieved past customer msg): {df['retrieval_relevance'].mean():.3f}")
    print(f"Median retrieval_relevance: {df['retrieval_relevance'].median():.3f}")
    print(f"Near-zero retrieval_relevance (< 0.05, likely bad match): {(df['retrieval_relevance'] < 0.05).sum()} / {len(df)}")
    print("\nWorst 5 retrievals (lowest relevance — likely off-topic grounding):")
    worst = df.nsmallest(5, "retrieval_relevance")
    for _, r in worst.iterrows():
        print(f"\nCUSTOMER:  {r['text'][:110]}")
        print(f"RETRIEVED: {r['retrieved_customer_text'][:110]}  (relevance={r['retrieval_relevance']})")
        print(f"DRAFT:     {r['draft'][:150]}")
    return df


if __name__ == "__main__":
    run_real_reply_eval()
