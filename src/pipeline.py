"""End-to-end agent: one customer message in, full decision out.

Defaults to the REAL-data-trained model/pipeline (see report.md — this
is the one that matters). Set data_csv/model_path explicitly to use the
synthetic fallback pipeline instead.
"""
import joblib
from src.ingest import load_brand_pairs
from src.classify_real import predict_intent
from src.draft_reply import GroundedRetriever, draft_reply
from src.escalate import decide_escalation

BRAND = "AppleSupport"


class SupportAgent:
    def __init__(self, data_csv="data/real_twcs.csv", model_path="src/intent_model_real.joblib"):
        bundle = joblib.load(model_path)
        self.vectorizer, self.clf = bundle["vectorizer"], bundle["clf"]
        pairs = load_brand_pairs(data_csv, BRAND, top_level_only=True)
        self.retriever = GroundedRetriever(pairs)

    def handle(self, customer_text: str):
        [intent], [confidence] = predict_intent([customer_text], self.vectorizer, self.clf)
        esc = decide_escalation(customer_text, intent, confidence)
        result = {
            "customer_text": customer_text,
            "intent": intent,
            "intent_confidence": round(float(confidence), 3),
            "escalate": esc["escalate"],
            "escalate_reason": esc["reason"],
            "escalate_detail": esc["detail"],
        }
        if not esc["escalate"]:
            draft = draft_reply(customer_text, self.retriever, k=3)
            result["draft_reply"] = draft["draft"]
            result["draft_method"] = draft["method"]
            result["grounded_on"] = [g["customer_text"] for g in draft["grounding"]]
        else:
            result["draft_reply"] = None
            result["draft_method"] = None
            result["grounded_on"] = None
        return result


if __name__ == "__main__":
    agent = SupportAgent()
    examples = [
        "my iphone won't turn on after i dropped it, screen is black",
        "is this email from apple legit or a phishing scam?",
        "how do i back up my ipad before selling it",
        "charged twice for icloud storage, need this fixed NOW, third time asking",
    ]
    for msg in examples:
        r = agent.handle(msg)
        print("=" * 80)
        print("IN:", msg)
        print(f"intent={r['intent']} (conf={r['intent_confidence']})  escalate={r['escalate']} ({r['escalate_reason']})")
        if r["draft_reply"]:
            print("DRAFT:", r["draft_reply"][:200])
