"""
Generates a synthetic sample dataset shaped exactly like the real
'Customer Support on Twitter' Kaggle dataset (same columns), for one
brand: AppleSupport.

WHY SYNTHETIC: this sandbox has no network access to kaggle.com or
huggingface.co, so the real ~3M row CSV can't be downloaded here. This
script produces data with the same schema, same noisy-tweet style, and
the same *kinds* of intents/resolutions you'd actually see for
AppleSupport in the real dataset, so the whole pipeline below is
provably runnable end-to-end. Swapping in the real file is a one-line
change — see README.

Real dataset columns (Kaggle thoughtvector/customer-support-on-twitter):
tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id
"""
import csv
import random
from datetime import datetime, timedelta

random.seed(7)

BRAND = "AppleSupport"

# 7 intents we define from looking at what AppleSupport actually handles
# on Twitter: device issues, software, account/iCloud, billing, warranty/
# repair, general how-to, and connectivity. This taxonomy IS a decision
# we made (see decision_log.md) — it's coarser than Apple's internal
# categories on purpose, because it's what's separable from tweet text alone.
INTENTS = {
    "battery_performance": {
        "templates": [
            "my {device} battery is dying so fast after the update, its unusable now",
            "@AppleSupport battery drains from 100 to 20 in like 2 hours on my {device}, help",
            "is anyone else's {device} battery health tanking after ios {ver}? mine dropped to {pct}%",
            "{device} won't hold a charge anymore, had it {age} only",
            "battery percentage jumps around randomly on my {device}, one sec its 40 next its 12",
        ],
        "resolution": "Thanks for reaching out. Battery health issues after an update are usually tied to background app refresh re-indexing after the install, which settles in 24-48h. If it's still bad after that, an official diagnostic can check battery health % in Settings > Battery > Battery Health, and we can look at repair/replacement options if it's below 80%. DM us your device serial so we can look into your specific case.",
    },
    "software_update_issue": {
        "templates": [
            "ios {ver} update bricked my {device}, stuck on apple logo",
            "cant update to ios {ver}, keeps saying 'unable to verify update' @AppleSupport",
            "after updating {device} to {ver} apps keep crashing constantly",
            "update has been 'preparing' for 3 hours now on my {device}, is this normal",
            "lost all my photos after the {ver} update??? this is insane",
        ],
        "resolution": "Sorry for the trouble — this sounds like the update got interrupted. First try a force restart (hold volume down + side button until the Apple logo appears). If it's stuck on the logo for over 15 minutes, connect to a computer and restore via Finder/iTunes in recovery mode — this should NOT erase data if it's a failed update rather than a factory reset. If you're on macOS/Windows and comfortable with these steps, DM us and we'll walk through it live.",
    },
    "device_not_turning_on": {
        "templates": [
            "my {device} won't turn on at all, screen is completely black",
            "{device} died and now wont charge or turn on, tried different cables",
            "dropped my {device} in water and now it's not responding, please help",
            "screen went black mid call and {device} hasn't turned back on since",
        ],
        "resolution": "That's frustrating, let's troubleshoot. Please try: 1) Connect to a wall charger (not a computer) for 30 min minimum, then attempt a force restart. 2) Check the charging port for lint/debris. If there's no response after that, or if liquid was involved, this typically needs a hands-on look at an Apple Store or Authorized Service Provider — liquid damage isn't something we can diagnose over DM. Let us know your location and we can find the nearest one.",
    },
    "account_login_icloud": {
        "templates": [
            "locked out of my apple id, it says my account is disabled",
            "cant sign into icloud on my {device}, keeps rejecting my password even after reset",
            "two factor code never arrives when i try to log into my apple id",
            "someone else is signed into my icloud and i cant remove them from my {device}",
        ],
        "resolution": "We understand how urgent account access is. For a disabled Apple ID, you'll need to go through account recovery at iforgot.apple.com — this can take anywhere from a few hours to several days depending on your account's security settings, and we can't expedite it over social media for security reasons. If you're not receiving 2FA codes, check that the trusted phone number on file is still active. DM us if you want us to check for any account restriction notices on our end.",
    },
    "warranty_repair_request": {
        "templates": [
            "how much would it cost to fix a cracked screen on my {device}, still under warranty?",
            "is my {device} still covered under applecare, bought it {age}",
            "want to book a repair for my {device}, nearest store is always fully booked",
            "genius bar keeps rescheduling my repair appointment, so annoying",
        ],
        "resolution": "Coverage depends on your AppleCare status — you can check this anytime at checkcoverage.apple.com with your serial number. Screen repairs for devices out of AppleCare+ are billed at the out-of-warranty rate, which varies by model. For booking, the Support app or apple.com/apple-store often shows slots that don't appear on the walk-in calendar. DM your device serial and zip code and we'll help find the earliest available appointment.",
    },
    "billing_subscription": {
        "templates": [
            "got charged twice for apple music this month, need a refund",
            "cancelled my icloud storage plan but still being charged @AppleSupport",
            "why was i charged {amt} for an app i never downloaded",
            "trying to get a refund for an accidental in-app purchase my kid made",
        ],
        "resolution": "We're sorry for the billing trouble. Most subscription and purchase issues can be resolved directly at reportaproblem.apple.com, where you can flag a specific charge for refund review — most requests are processed within 48 hours. Duplicate charges are often a pending authorization that drops off automatically in 1-3 business days rather than a real double-charge, but if it's still there after that, use the link above and we'll escalate from our end too.",
    },
    "general_howto": {
        "templates": [
            "how do i transfer photos from my old {device} to my new one",
            "is there a way to see how much storage im using on {device}",
            "how do you turn off those annoying notification previews on {device}",
            "whats the best way to back up my {device} before i sell it",
        ],
        "resolution": "Great question! Quick Start makes device-to-device transfer easiest — just place your old and new device near each other and follow the prompts. For storage, check Settings > General > iPhone Storage for a full breakdown by app. Before selling, back up via iCloud or your computer, then use Settings > General > Transfer or Reset > Erase All Content and Settings. Happy to walk through any of these in more detail if you get stuck.",
    },
}

DEVICES = ["iPhone 13", "iPhone 14", "iPhone 15", "iPad Pro", "iPad Air", "MacBook Air", "Apple Watch", "iPhone SE"]
VERSIONS = ["17.2", "17.4", "17.5.1", "18.0", "18.1"]
AGES = ["6 months ago", "a year ago", "3 weeks ago", "2 years ago", "last month"]

# A handful of messages that SHOULD trip escalation regardless of intent —
# legal threats, self-harm-adjacent distress framing, explicit "human"
# requests, repeated unresolved contact. These are injected across intents.
ESCALATION_TRIGGERS = [
    " this is the third time im contacting you about this, absolutely done",
    " my lawyer will be in touch if this isn't resolved today",
    " i want to speak to an actual human being, not a bot",
    " i'm going to leave apple over this, genuinely furious",
    " reported this to consumer protection already",
]


def make_thread(thread_id, intent_name, brand_author_ids, cust_id_start):
    spec = INTENTS[intent_name]
    template = random.choice(spec["templates"])
    text = template.format(
        device=random.choice(DEVICES),
        ver=random.choice(VERSIONS),
        pct=random.randint(60, 85),
        age=random.choice(AGES),
        amt=f"${random.choice([2.99, 9.99, 14.99, 29.99])}",
    )
    escalate_flag = random.random() < 0.18
    if escalate_flag:
        text += random.choice(ESCALATION_TRIGGERS)

    t0 = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 300), minutes=random.randint(0, 1000))
    cust_author = f"cust_{cust_id_start}"
    brand_author = "AppleSupport"

    inbound_id = thread_id * 10
    outbound_id = inbound_id + 1

    rows = [
        {
            "tweet_id": inbound_id,
            "author_id": cust_author,
            "inbound": "True",
            "created_at": t0.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "text": f"@{brand_author} {text}",
            "response_tweet_id": str(outbound_id),
            "in_response_to_tweet_id": "",
        },
        {
            "tweet_id": outbound_id,
            "author_id": brand_author,
            "inbound": "False",
            "created_at": (t0 + timedelta(minutes=random.randint(5, 240))).strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "text": f"@{cust_author} {spec['resolution']}",
            "response_tweet_id": "",
            "in_response_to_tweet_id": str(inbound_id),
        },
    ]
    return rows, intent_name, escalate_flag, text


def main():
    all_rows = []
    labels = []  # (tweet_id, text, intent, escalate) for building golden set later
    thread_id = 1000
    n_per_intent = 90  # 7 intents * 90 = 630 threads = 1260 tweets, "sub-sample" scale
    for intent_name in INTENTS:
        for i in range(n_per_intent):
            rows, intent, escalate, raw_text = make_thread(thread_id, intent_name, [], thread_id)
            all_rows.extend(rows)
            labels.append({
                "tweet_id": rows[0]["tweet_id"],
                "text": rows[0]["text"],
                "intent": intent,
                "escalate": escalate,
            })
            thread_id += 1

    random.shuffle(all_rows)

    with open("data/sample_twcs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tweet_id", "author_id", "inbound", "created_at", "text",
                                           "response_tweet_id", "in_response_to_tweet_id"])
        w.writeheader()
        w.writerows(all_rows)

    with open("data/full_synthetic_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tweet_id", "text", "intent", "escalate"])
        w.writeheader()
        w.writerows(labels)

    print(f"Wrote {len(all_rows)} tweets ({len(labels)} threads) to data/sample_twcs.csv")
    print(f"Wrote ground-truth generator labels (NOT the golden set) to data/full_synthetic_labels.csv")


if __name__ == "__main__":
    main()
