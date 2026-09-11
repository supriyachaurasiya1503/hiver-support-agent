"""
Loads the raw twcs-schema CSV (real Kaggle file OR our schema-identical
synthetic sample) and reconstructs (customer_message, brand_resolution)
pairs for one brand.

Real dataset note: the real file is ~3M rows across dozens of brands and
does not fit comfortably in-memory as naive pandas on a laptop; we
stream-filter by author_id/in_response_to before building thread pairs.
Here on the sample it's small enough to just load directly, but the
filter-first code path is written to also work on the full file.
"""
import pandas as pd


def load_brand_pairs(csv_path: str, brand: str = "AppleSupport", top_level_only: bool = True) -> pd.DataFrame:
    """Returns a DataFrame with one row per (customer message -> brand
    reply) pair for the given brand: columns customer_text, brand_text,
    customer_tweet_id, brand_tweet_id.

    top_level_only=True (default, and the important fix — see
    decision_log.md item 17): restricts to customer messages that are
    THREAD-INITIATING (in_response_to_tweet_id is empty), not every
    customer turn in a thread. Without this, a customer's mid-thread
    "thanks!" or "okay" gets paired with whatever the brand says next
    and treated as if it were a fresh incoming issue to classify — on
    the real dataset this was roughly half of all naively-paired
    messages and was the single largest source of "general_inquiry"
    catch-all noise before this fix.
    """
    # Stream-filter: only load rows that are either inbound-to-brand or
    # authored-by-brand. On the real 3M-row file this keeps memory bounded
    # instead of loading everything.
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=200_000, dtype=str):
        mask = (chunk["author_id"] == brand) | (
            chunk["text"].str.contains(f"@{brand}", na=False)
        )
        chunks.append(chunk[mask])
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    brand_replies = df[df["author_id"] == brand].set_index("tweet_id")
    customer_msgs = df[df["author_id"] != brand]

    if top_level_only:
        customer_msgs = customer_msgs[
            customer_msgs["in_response_to_tweet_id"].isna()
            | (customer_msgs["in_response_to_tweet_id"] == "")
        ]

    pairs = []
    for _, row in customer_msgs.iterrows():
        resp_id = row.get("response_tweet_id")
        if pd.isna(resp_id) or resp_id == "":
            continue
        # response_tweet_id can be a comma-separated list in the real data
        first_resp = str(resp_id).split(",")[0].strip()
        if first_resp in brand_replies.index:
            brand_row = brand_replies.loc[first_resp]
            pairs.append({
                "customer_tweet_id": row["tweet_id"],
                "customer_text": clean_text(row["text"], brand),
                "brand_tweet_id": first_resp,
                "brand_text": clean_text(brand_row["text"], brand),
            })
    return pd.DataFrame(pairs)


def clean_text(text: str, brand: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace(f"@{brand}", "").strip()
    # strip leading @handle (reply-to mention) e.g. "@cust_123 message..."
    parts = text.split(" ", 1)
    if parts and parts[0].startswith("@") and len(parts) > 1:
        text = parts[1]
    return text.strip()


if __name__ == "__main__":
    df = load_brand_pairs("data/sample_twcs.csv", "AppleSupport")
    print(f"Loaded {len(df)} customer->brand pairs")
    print(df.head(3).to_string())
