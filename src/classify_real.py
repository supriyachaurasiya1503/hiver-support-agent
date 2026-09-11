"""
Intent classifier trained on REAL AppleSupport data.

IMPORTANT — training labels are WEAK (rule-based, src/real_taxonomy.py),
not hand-labeled, because hand-labeling all ~45k training examples isn't
feasible. This is standard weak supervision, but it means the training
signal has an estimated ~82% agreement with a human reader (measured on
a 34-example spot check, see eval/build_golden_set_real.py) — the
classifier is learning to approximate the WEAK RULES, which is a lower
ceiling than learning true human intent. The 234-example GOLDEN set is
independently spot-check-corrected and is what evaluation numbers below
are measured against.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


def load_training_pool(labeled_csv: str, golden_csv: str) -> pd.DataFrame:
    pool = pd.read_csv(labeled_csv)
    golden = pd.read_csv(golden_csv)
    pool = pool[~pool["customer_tweet_id"].isin(golden["customer_tweet_id"])]
    pool = pool[pool["customer_text"].str.len().between(8, 300)]
    return pool.rename(columns={"weak_intent": "intent"})


def train_classifier(pool: pd.DataFrame):
    X_train, X_val, y_train, y_val = train_test_split(
        pool["customer_text"], pool["intent"], test_size=0.1, random_state=0, stratify=pool["intent"]
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=8000, sublinear_tf=True)
    Xt = vectorizer.fit_transform(X_train)
    Xv = vectorizer.transform(X_val)

    clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")
    clf.fit(Xt, y_train)

    val_preds = clf.predict(Xv)
    print("--- internal validation split (held out from weak-labeled training pool, NOT the golden set) ---")
    print(classification_report(y_val, val_preds, zero_division=0))

    joblib.dump({"vectorizer": vectorizer, "clf": clf}, "src/intent_model_real.joblib")
    return vectorizer, clf


def predict_intent(texts, vectorizer=None, clf=None):
    if vectorizer is None or clf is None:
        bundle = joblib.load("src/intent_model_real.joblib")
        vectorizer, clf = bundle["vectorizer"], bundle["clf"]
    X = vectorizer.transform(texts)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    confidences = probs.max(axis=1)
    return preds, confidences


def evaluate_on_golden(golden_csv: str):
    golden = pd.read_csv(golden_csv)
    bundle = joblib.load("src/intent_model_real.joblib")
    vectorizer, clf = bundle["vectorizer"], bundle["clf"]

    majority_class = golden["intent"].mode()[0]
    trivial_preds = [majority_class] * len(golden)
    trivial_acc = accuracy_score(golden["intent"], trivial_preds)
    trivial_f1 = f1_score(golden["intent"], trivial_preds, average="macro", zero_division=0)

    preds, confidences = predict_intent(golden["customer_text"], vectorizer, clf)
    acc = accuracy_score(golden["intent"], preds)
    macro_f1 = f1_score(golden["intent"], preds, average="macro", zero_division=0)

    print(f"\n=== INTENT CLASSIFICATION ON REAL DATA: golden set (n={len(golden)}) ===")
    print(f"Trivial baseline (always predict '{majority_class}'): acc={trivial_acc:.3f}, macro_f1={trivial_f1:.3f}")
    print(f"TF-IDF + LogReg (ours), weak-label trained:            acc={acc:.3f}, macro_f1={macro_f1:.3f}")
    print(f"Mean prediction confidence: {confidences.mean():.3f}")
    print("\nPer-class report:")
    print(classification_report(golden["intent"], preds, zero_division=0))

    golden["predicted_intent"] = preds
    golden["confidence"] = confidences
    golden.to_csv("eval/golden_set_real_with_predictions.csv", index=False)

    return {"trivial_acc": trivial_acc, "trivial_f1": trivial_f1, "model_acc": acc, "model_f1": macro_f1}


if __name__ == "__main__":
    pool = load_training_pool("data/real_apple_pairs_labeled.csv", "eval/golden_set_real.csv")
    print(f"Training pool size (golden set excluded, weak labels): {len(pool)}")
    print(pool["intent"].value_counts())
    train_classifier(pool)
    evaluate_on_golden("eval/golden_set_real.csv")
