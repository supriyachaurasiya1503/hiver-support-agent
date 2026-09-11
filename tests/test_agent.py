"""
Automated tests (kaami #11 from the review: no test suite existed
before). Focused on the parts that are easy to silently break:
escalation rule logic, thread-pairing correctness, and the weak
labeler's ordering/precedence.

Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.escalate import decide_escalation
from src.real_taxonomy import weak_label_intent, clean_real_text
from src.ingest import clean_text


class TestEscalation:
    def test_explicit_human_request_escalates(self):
        d = decide_escalation("i want to talk to an actual human", "general_inquiry", 0.9)
        assert d["escalate"] is True
        assert d["reason"] == "explicit_human_request"

    def test_legal_threat_escalates(self):
        d = decide_escalation("fix this or i will sue you", "device_hardware_issue", 0.9)
        assert d["escalate"] is True
        assert d["reason"] == "legal_or_regulatory"

    def test_phishing_intent_always_escalates_even_high_confidence(self):
        d = decide_escalation("routine message", "phishing_scam_report", 0.99)
        assert d["escalate"] is True
        assert d["reason"] == "security_sensitive_intent"

    def test_low_confidence_escalates(self):
        d = decide_escalation("some ambiguous message", "general_inquiry", 0.2)
        assert d["escalate"] is True
        assert d["reason"] == "low_classifier_confidence"

    def test_high_confidence_routine_message_auto_handles(self):
        d = decide_escalation("how do i turn off notifications", "general_inquiry", 0.95)
        assert d["escalate"] is False
        assert d["reason"] == "auto_handle"

    def test_large_billing_amount_escalates(self):
        d = decide_escalation("why was i charged $150 for this", "billing_subscription_refund", 0.9)
        assert d["escalate"] is True
        assert d["reason"] == "sensitive_intent_with_financial_ask"

    def test_small_billing_amount_does_not_escalate_on_amount_alone(self):
        d = decide_escalation("why was i charged $2.99 for this app", "billing_subscription_refund", 0.9)
        # should NOT hit the money-threshold trigger (below $20), and
        # nothing else in this message should trigger escalation either
        assert d["reason"] != "sensitive_intent_with_financial_ask"

    def test_double_charge_escalates_regardless_of_amount(self):
        d = decide_escalation("i was charged twice for my subscription", "billing_subscription_refund", 0.9)
        assert d["escalate"] is True

    def test_priority_order_human_request_beats_low_confidence(self):
        # message would also trigger low_classifier_confidence, but
        # explicit_human_request must win since it's checked first
        d = decide_escalation("i need a human, not a bot", "general_inquiry", 0.1)
        assert d["reason"] == "explicit_human_request"


class TestWeakLabeler:
    def test_phishing_checked_before_account_security(self):
        # contains "apple id" (account_security pattern) AND phishing
        # language — phishing must win per the documented rule order
        label = weak_label_intent("is this apple id email a phishing scam")
        assert label == "phishing_scam_report"

    def test_battery_keyword_detected(self):
        assert weak_label_intent("my battery drains so fast") == "battery_performance"

    def test_unmatched_text_falls_back_to_general_inquiry(self):
        assert weak_label_intent("hello how are you today") == "general_inquiry"

    def test_photos_sync_detected(self):
        label = weak_label_intent("i lost all my photos after the update")
        assert label == "photos_media_sync"


class TestTextCleaning:
    def test_clean_real_text_strips_tco_links(self):
        cleaned = clean_real_text("check this out https://t.co/abc123 thanks")
        assert "t.co" not in cleaned

    def test_clean_real_text_strips_numeric_handles(self):
        cleaned = clean_real_text("@115858 hello there")
        assert "@115858" not in cleaned

    def test_ingest_clean_text_strips_brand_mention(self):
        cleaned = clean_text("@AppleSupport my phone is broken", "AppleSupport")
        assert "@AppleSupport" not in cleaned
        assert "phone is broken" in cleaned


class TestIngestThreading:
    def test_top_level_filter_excludes_replies(self, tmp_path):
        # build a tiny synthetic twcs-schema CSV with one thread-initiating
        # customer message and one mid-thread customer reply
        csv_content = (
            "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
            '1,cust_1,True,Mon Jan 01 00:00:00 +0000 2024,"@Brand my phone is broken",2,\n'
            '2,Brand,False,Mon Jan 01 00:01:00 +0000 2024,"@cust_1 sorry to hear, DM us",3,1\n'
            '3,cust_1,True,Mon Jan 01 00:02:00 +0000 2024,"@Brand thanks!",4,2\n'
            '4,Brand,False,Mon Jan 01 00:03:00 +0000 2024,"@cust_1 no problem",,3\n'
        )
        p = tmp_path / "tiny_twcs.csv"
        p.write_text(csv_content)

        from src.ingest import load_brand_pairs
        top_level = load_brand_pairs(str(p), "Brand", top_level_only=True)
        all_pairs = load_brand_pairs(str(p), "Brand", top_level_only=False)

        assert len(top_level) == 1
        assert "phone is broken" in top_level.iloc[0]["customer_text"]
        assert len(all_pairs) == 2  # includes the "thanks!" mid-thread reply
