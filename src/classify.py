"""
Intent classification, with two baselines as required by the assignment:
  - trivial baseline: always predict the majority class
  - simple baseline: TF-IDF + Logistic Regression (this doubles as our
    actual production classifier too — see decision_log.md for why we
    didn't reach for an LLM call per-message here: latency + cost per
    classification on high-volume support traffic, and a linear model
    on n-grams is already >90% on this task, see eval results).

We train on threads OUTSIDE the golden set (a held-out training pool),
never on the golden set itself, to keep the eval honest.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


def load_training_pool(labels_csv: str, golden_csv: str) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    golden = pd.read_csv(golden_csv)
    # exclude anything in the golden set from training — no leakage
    pool = labels[~labels["tweet_id"].isin(golden["tweet_id"])]
    return pool


def train_classifier(pool: pd.DataFrame):
    X_train, X_val, y_train, y_val = train_test_split(
        pool["text"], pool["intent"], test_size=0.15, random_state=0, stratify=pool["intent"]
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True)
    Xt = vectorizer.fit_transform(X_train)
    Xv = vectorizer.transform(X_val)

    clf = LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")
    clf.fit(Xt, y_train)

    val_preds = clf.predict(Xv)
    print("--- internal validation split (not the golden set) ---")
    print(classification_report(y_val, val_preds))

    joblib.dump({"vectorizer": vectorizer, "clf": clf}, "src/intent_model.joblib")
    return vectorizer, clf


def predict_intent(texts, vectorizer=None, clf=None):
    if vectorizer is None or clf is None:
        bundle = joblib.load("src/intent_model.joblib")
        vectorizer, clf = bundle["vectorizer"], bundle["clf"]
    X = vectorizer.transform(texts)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    confidences = probs.max(axis=1)
    return preds, confidences


def trivial_baseline_predict(texts, majority_class):
    return [majority_class] * len(texts)


def evaluate_on_golden(golden_csv: str):
    golden = pd.read_csv(golden_csv)
    bundle = joblib.load("src/intent_model.joblib")
    vectorizer, clf = bundle["vectorizer"], bundle["clf"]

    # --- trivial baseline ---
    majority_class = golden["intent"].mode()[0]
    trivial_preds = trivial_baseline_predict(golden["text"], majority_class)
    trivial_acc = accuracy_score(golden["intent"], trivial_preds)
    trivial_f1 = f1_score(golden["intent"], trivial_preds, average="macro")

    # --- our classifier ---
    preds, confidences = predict_intent(golden["text"], vectorizer, clf)
    acc = accuracy_score(golden["intent"], preds)
    macro_f1 = f1_score(golden["intent"], preds, average="macro")

    print(f"\n=== INTENT CLASSIFICATION: golden set (n={len(golden)}) ===")
    print(f"Trivial baseline (always predict '{majority_class}'): acc={trivial_acc:.3f}, macro_f1={trivial_f1:.3f}")
    print(f"TF-IDF + LogReg (ours):                                acc={acc:.3f}, macro_f1={macro_f1:.3f}")
    print(f"Mean prediction confidence: {confidences.mean():.3f}")
    print("\nPer-class report:")
    print(classification_report(golden["intent"], preds))

    golden["predicted_intent"] = preds
    golden["confidence"] = confidences
    golden.to_csv("eval/golden_set_with_predictions.csv", index=False)

    return {
        "trivial_acc": trivial_acc, "trivial_f1": trivial_f1,
        "model_acc": acc, "model_f1": macro_f1,
    }


if __name__ == "__main__":
    pool = load_training_pool("data/full_synthetic_labels.csv", "eval/golden_set.csv")
    print(f"Training pool size (golden set excluded): {len(pool)}")
    train_classifier(pool)
    evaluate_on_golden("eval/golden_set.csv")
