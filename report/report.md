# Report — AppleSupport AI Support Agent (v2: real dataset)

## 0. What changed since v1

v1 of this report ran entirely on synthetic data because this sandbox
had no path to Kaggle. The user then uploaded the real
`thoughtvector/customer-support-on-twitter` archive
(`archive.zip` → `twcs/twcs.csv`, ~3M rows), and this version re-does
the entire pipeline against it. Every number below is real. Where v1
flagged something as "likely inflated" or "untested," this version
either fixes it or reports the honest real number — see the
"kaami → fix" table at the end of Section 4.

Banking77 (the assignment's optional secondary dataset) was **not**
used — `huggingface.co` is still unreachable from this sandbox, and it
wasn't uploaded. Everything here uses only the primary dataset, which
the assignment says is sufficient.

## 1. Problem framing (updated)

Same brand (AppleSupport) and same priorities as v1 (never confidently
auto-answer something risky; ground replies in real history; make
escalation reasons legible). **The intent taxonomy changed** after
actually looking at real data — see Section 2's taxonomy note. We also
made one structural fix that mattered more than any taxonomy change:

**Thread-pairing fix.** The naive approach (pair any customer message
that has a `response_tweet_id` pointing at the brand) captures **every
customer turn in a thread**, including "thanks!" and "okay" replies
mid-conversation — these aren't incoming issues to classify, they're
conversational filler that happens to sit next to a brand reply.
Filtering to only **thread-initiating** messages
(`in_response_to_tweet_id` empty) dropped the candidate pool from
97,153 to 45,366 real customer→brand pairs and measurably reduced
catch-all mislabeling (Section 3). This is now the default in
`src/ingest.py` (`top_level_only=True`).

## 2. Intent taxonomy — re-derived from real data

We ran a keyword-frequency scan over all 73,571 real customer messages
before naming a single intent (see `src/real_taxonomy.py`). Two
categories the earlier synthetic version completely missed showed up
clearly and are added:
- **`phishing_scam_report`** (~360 messages after cleaning: "is this a
  scam," "phishing email," "fake text") — real, distinct, and
  **security-sensitive** enough that we route it to *always* escalate
  (Section 5), not just classify it.
- **`photos_media_sync`** (~245 messages: lost photos, iCloud Photos
  not syncing) — big and distinct enough to deserve its own bucket
  instead of being folded into generic account issues.

Final 9 intents: `software_update_issue`, `battery_performance`,
`device_hardware_issue`, `account_security`, `warranty_repair_request`,
`billing_subscription_refund`, `photos_media_sync`,
`phishing_scam_report`, `general_inquiry` (catch-all).

**`general_inquiry` is ~59% of real traffic even after the
thread-pairing fix.** This is not a bug to hide — real Twitter support
traffic is genuinely long-tailed (app-specific bugs, sarcasm, one-off
feature questions, non-English text, messages missing context because
they originally replied to an image). A keyword-rule taxonomy will
always leave a large uncategorized remainder; a real production system
would need either a learned classifier trained on enough hand-labeled
long-tail examples to subdivide this further, or an acceptance that
"general inquiry → human-reviewed or LLM-generated-from-scratch reply"
is itself a legitimate route, not a failure.

## 3. Results

### Intent classification (n=234 real golden set)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Trivial baseline (always predict majority class) | 12.0% | 2.4% |
| **TF-IDF + Logistic Regression, weak-label trained (ours)** | **92.7%** | **92.8%** |

This is **lower than v1's 95.9%**, as predicted — real lexical variety
costs ~3 points versus the templated synthetic version. Per-class
detail: `general_inquiry` precision is only 0.69 (recall 1.00) — the
classifier over-predicts the catch-all bucket, meaning some
genuine-but-rare-category messages (e.g. `warranty_repair_request`,
recall 0.77) get swallowed into "general" rather than mis-routed
somewhere actively wrong. That's a relatively safe failure direction
(a swallowed warranty question just doesn't get a great targeted
reply, vs. actively wrong info), but it is real information loss.

**Training labels are weak (rule-based), not hand-labeled** — we
measured ~82% agreement between the weak labeler and a human reading
34 spot-checked examples (`eval/build_golden_set_real.py`), found and
corrected 6 real mislabels in that process (e.g., "why my charge goes
from 16% to 9%" was bucketed as billing because "charge" is an
overloaded word — actually battery). The classifier is trained on
labels with a real, measured error rate, not ground truth; 92.7% on
the (separately, more carefully corrected) golden set is a genuine
result, but the training signal underneath it is imperfect by
construction.

### Reply drafting (n=234 real golden set)

The `grounding_overlap` metric from v1 was **found to be broken** while
running this: it's trivially 1.0 whenever the extractive fallback is
used, because that fallback returns the retrieved source verbatim —
comparing a string to itself. Replaced with `retrieval_relevance`:
token overlap between the **new customer message** and the
**retrieved historical customer message**, which measures whether BM25
found a genuinely similar past case.

| Metric | Value |
|---|---|
| Mean retrieval_relevance | 0.225 |
| Median retrieval_relevance | 0.206 |
| Near-zero (<0.05) retrievals — likely bad matches | 4 / 234 (1.7%) |

A second bug was caught and fixed in the same script: the retrieval
corpus initially still included the golden set's **own** tweets, so
many examples retrieved themselves (mean relevance was a meaningless
0.974 before the fix). After excluding golden-set tweet IDs from the
corpus, the number above is real.

**Judge-vs-human agreement — the most important finding in this
report.** We hand-scored a 40-example sample of real drafts (blind to
the heuristic judge's output) on the same relevance/grounded/tone/
actionable rubric as v1:

| Metric | v1 (synthetic) | v2 (real data) |
|---|---|---|
| Exact-score agreement (relevance) | 40.0% | 40.0% |
| Pass/fail agreement (relevance≥4 AND grounded≥4) | 93.3% | **55.0%** |
| Heuristic judge's own pass rate | 100% | 100% |
| **Human's actual pass rate** | 93.3% | **55.0%** |

On synthetic data, the heuristic judge's blindness to real topical
relevance was mostly hidden because nearly every retrieval was
correct by construction (limited templates). On real, messy data, **the
extractive-fallback agent is actually only right about 55% of the
time by a human's judgment — while the heuristic judge insists it's
100% correct, every time, because it only measures "did the draft
equal its source" (trivially yes).** This is exactly why an actual
LLM-as-judge (not a heuristic stand-in) matters, and it's the clearest
evidence in this whole project that the synthetic-data results in v1
could not be trusted as a preview of real performance.

Concrete real failures a human catches instantly and the heuristic
cannot: a message asking "can I click next when creating this apple
id" retrieved and returned a reply about **calendar entries**; a
message about Instagram/Snapchat crashing retrieved a reply directing
the customer to the **Online Sales team**; a message asking about
missing Audio Books retrieved a reply about **Visual Voicemail**. All
three would ship as-is under the current extractive fallback and are
wrong. See `eval/judge_human_agreement_real.csv` for the full 40.

### Escalation decision (n=234 real golden set)

We do **not** have an independent "should this have escalated" human
label for real threads (nobody annotated the dataset for this, and we
aren't a real support agent who knows what should have happened next)
— computing precision/recall against a label generated by the same
rule being tested would be circular. Instead:

- **Manual audit of all 35 flagged escalations** (100% of positives):
  every one was judged a reasonable call on reading — mostly
  `phishing_scam_report` intent (always escalates by design, 18 of 35),
  low classifier confidence on genuinely ambiguous messages, and a few
  billing/fraud disputes.
- **Broader-net scan of the auto-handle bucket** for likely misses,
  using a looser keyword list than the actual rules ("cancel," "hate,"
  "terrible," "mad," "stupid," "useless"): 13 matches, of which most
  were reasonably left auto-handled (mild frustration, not real risk),
  but this scan is exactly how we found and added the
  `"terrible experience"` trigger phrase that was previously missing.
- Real escalation-trigger phrases are genuinely rare: ~0.3% of raw
  messages (149/45,366) match ANY of our human-request/legal/
  repeated-contact/severity patterns before the phishing-intent rule is
  even applied. Real customers rarely narrate their own anger the way
  synthetic-generator templates did.

This is real progress over v1 (which never built a rule-vs.-ground-truth
comparison at all for real data), but it's honest to call this a
**manual audit**, not a certified precision/recall number — see
Section 4, item 5.

## 4. Failure analysis — updated top failure modes

1. **Retrieval quality has a real, measurable long tail.** ~1.7% of
   real golden examples get a near-zero-relevance retrieval, and the
   human-judge sample shows the true failure rate on "is this reply
   actually about the right thing" is closer to 45% (100% − 55% human
   pass rate) once you include on-topic-intent-but-wrong-specific-issue
   retrievals like the calendar/Sales-team/Voicemail examples above.
   The extractive fallback is not a safe default for production; it
   needs the LLM-grounded generation path (still untested — item 3).

2. **`general_inquiry` catch-all absorbs ~59% of real traffic even
   after the thread-pairing fix.** This is a taxonomy/data-coverage
   limitation, not a code bug — see Section 2.

3. **The LLM-grounded draft path and the real LLM-as-judge are still
   untested** — no `ANTHROPIC_API_KEY` in this sandbox. Given the 55%
   human pass rate on the extractive-fallback path, this is now the
   single highest-priority gap: we have direct evidence the fallback
   isn't good enough, but no evidence yet on whether the intended
   system (generative, grounded) actually does better.

4. **Weak training labels cap classifier reliability below what 92.7%
   accuracy implies.** The classifier is scored against a carefully
   corrected golden set, but trained on labels with a measured ~18%
   disagreement rate with a human reader (82% agreement, Section 3).
   Some of its "errors" against the golden set may actually be
   correctly learning a wrong pattern from noisy training data, not
   independent model failure.

5. **No certified escalation precision/recall exists, real or
   synthetic.** v1's "100% recall, 94.9% precision" was measured
   against labels generated by the same synthetic generator whose
   trigger-phrase list the rules were built to catch — closer to a
   unit test than an evaluation. v2's manual audit (Section 3) is more
   honest about real-world rarity of trigger phrases but is still not
   a number you could put confidence intervals on. A real next step:
   get a second person to independently label 100-200 real messages
   for "should this escalate" without seeing the rule's output, then
   compute real precision/recall.

6. **Escalation rules cannot distinguish sarcasm from genuine intent**
   ("I'll sue you lol" vs. an actual legal threat) — confirmed on real
   data (`eval/evaluate_escalation_real.py`'s audit found several
   `sue` matches that were clearly hyperbolic). We accept this
   trade-off explicitly: over-escalating a sarcastic complaint costs a
   few minutes of human review; under-escalating a real legal threat
   costs much more.

7. **Money-amount detection still can't parse "twenty dollars," mixed
   currencies beyond $/£/€ symbols, or amounts split across a thread.**
   Improved from v1 (now handles £/€ and "charged twice" phrasing) but
   not exhaustive.

8. **Multi-turn / thread-history escalation signal
   (`decide_escalation_with_history`) is implemented but not
   evaluated** — we don't have a clean way to compute "how many times
   has this author_id contacted AppleSupport before" at evaluation time
   without re-deriving it from the full 3M-row file per author, which
   we didn't do in this pass. This closes part of kaami #6 (the
   function exists and is tested for logic) but not all of it (no
   real-world evaluation of its impact on recall).

### Kaami → fix summary

| Original kaami | Status |
|---|---|
| #1 Real dataset never used | **Fixed** — real 3M-row Kaggle file loaded, 45,366 real thread-initiating pairs used throughout |
| #2 LLM-grounded generation never run | **Still open** — no API key available |
| #3 LLM-as-judge was a heuristic stand-in | **Still open** (same reason), but its unreliability is now directly demonstrated (item 3 above), not just assumed |
| #4 95.9% intent accuracy inflated by templates | **Fixed / confirmed** — real number is 92.7%, taxonomy re-derived from real keyword frequencies |
| #5 Escalation rules keyword-based / real coverage unverified | **Improved** — expanded patterns from real data, added phishing-always-escalates rule, manual audit done (not a certified P/R, see item 5 above) |
| #6 No multi-turn context | **Partially fixed** — `decide_escalation_with_history()` implemented and unit-tested; not evaluated end-to-end on real thread histories |
| #7 `general_howto` too coarse, off-topic replies | **Improved** — taxonomy split into 9 real-derived buckets incl. `photos_media_sync` and `phishing_scam_report`; `general_inquiry` catch-all is smaller but still large (Section 2) |
| #8 Money regex too narrow | **Improved** — handles £/€ symbols and "charged twice" phrasing now |
| #9 Confidence threshold untuned | **Attempted, honestly inconclusive** — we don't have ground truth to tune against on real data either (see `eval/evaluate_escalation_real.py` docstring); still 0.55, unchanged |
| #10 Golden set circular (from our own generator) | **Fixed** — golden set built from real threads, spot-check-corrected by hand against real text |
| #11 No automated tests | **Fixed** — `tests/test_agent.py`, 17 tests, `pytest tests/` |
| #12 No cleaning for real-data artifacts | **Fixed** — `src/real_taxonomy.py`'s `clean_real_text()` strips t.co links and anonymized numeric handles |

Two additional bugs were found and fixed *while doing this real-data
pass* that weren't on the original kaami list: the `grounding_overlap`
metric being trivially 1.0 by construction (Section 3), and retrieval
corpus leakage (golden-set tweets retrieving themselves, Section 3).
Both are the kind of mistake that's invisible on clean synthetic data
and only surfaces once you run against something real — which is
itself the strongest argument for why item #1 mattered more than any
other single fix on this list.

## 5. What we'd do next with one more week

1. **Get an API key wired in** and actually run the LLM-grounded draft
   path + real LLM-as-judge — this is now the clear top priority given
   the 55% human pass rate finding.
2. **Independent escalation ground truth**: have a second person
   label 150-200 real messages for "should this escalate," blind to
   the rule's output, then compute real precision/recall.
3. **Evaluate `decide_escalation_with_history()`** by actually
   computing per-author prior-contact counts from the full 3M-row file
   and checking whether it changes recall on the manual-audit misses.
4. **Subdivide `general_inquiry`** using topic clustering (e.g. LDA or
   embedding clusters) over its ~27k real messages to find sub-buckets
   worth splitting out, rather than guessing more keyword rules.
5. **Re-run the weak-label training pool through the same 34-example-
   style spot-check at larger scale** (100-200 examples) to get a
   tighter estimate than 82% agreement, and consider a
   semi-supervised bootstrap (train on weak labels, use high-confidence
   predictions to relabel low-confidence weak labels, iterate).
6. **Add Banking77** if it becomes reachable, purely for the intent
   taxonomy comparison the assignment mentions as optional — not
   integrated in this pass since huggingface.co isn't reachable and it
   wasn't uploaded.
