# Report — AppleSupport AI Support Agent

An AI support agent for `@AppleSupport` that classifies inbound Twitter
messages into intents, drafts a reply grounded in how the brand has
historically resolved similar issues, and decides auto-handle vs.
escalate-to-human with a stated reason. This report explains what was
built, exactly how, what the results are, and — deliberately — what is
still broken or untested.

---

## 1. Brand and data chosen, and why

**Brand: `@AppleSupport`.** Picked over other brands in the dataset
(e.g. retail or airline accounts) because its Twitter traffic has
clearly separable technical/account/billing issue types, which makes a
small intent taxonomy meaningful, and because its resolution threads
tend to be detailed rather than generic "DM us" brush-offs — better
raw material for grounding replies.

**Dataset: Kaggle `thoughtvector/customer-support-on-twitter`**
(the real ~3M-row `twcs.csv`, provided by the user). Columns:
`tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id`.

An earlier pass of this project was built on a synthetic dataset
(same schema, generator-created) because the sandbox originally had no
network path to Kaggle. Once the real file was provided, the entire
pipeline was rebuilt and re-evaluated against it — synthetic-data code
is kept in the repo only for reference/fast smoke-testing, and is
**not** what the numbers in this report describe unless stated
otherwise.

---

## 2. Step 1 — Turning raw tweets into (customer message, brand
   resolution) pairs

**What the raw data looks like:** every tweet — customer or brand — is
one row. A customer's tweet and the brand's reply are two separate
rows linked by `response_tweet_id` / `in_response_to_tweet_id`. To
build training/grounding data, these had to be reconstructed into
pairs: *this customer message* → *this brand reply*.

**How (`src/ingest.py`):**
1. Stream the 3M-row file in chunks of 200,000 rows (not loaded fully
   into memory at once) and keep only rows where the author is
   `AppleSupport` or the text mentions `@AppleSupport`.
2. Index all `AppleSupport`-authored rows by `tweet_id`.
3. For every customer row, look up its `response_tweet_id` in that
   index — if it matches an AppleSupport tweet, that's a resolved pair.
4. Clean each side of the pair: strip the `@AppleSupport` mention and
   any leading `@handle` (the reply-to mention) from the text.

**A real problem found and fixed here:** the first version of this
pairing captured **every** customer message in a thread that happens
to have a brand reply after it — including a customer's own mid-thread
"thanks!" or "okay" reply, which is not a fresh incoming issue at all,
just conversational filler. Fix: restrict to only **thread-initiating**
messages (`in_response_to_tweet_id` is empty — meaning the customer
tweet is not itself a reply to something else). This dropped the
candidate pool from 97,153 to **45,366** genuine customer→brand pairs
and is now the default (`top_level_only=True`).

---

## 3. Step 2 — Designing the intent taxonomy from real data

The assignment asks for "a small set of intents you define from the
data" — so this had to come from actually reading real messages, not
from guessing.

**How:** ran a keyword-frequency scan across all 73,571 real customer
messages (before the thread-filter fix) — counted how often words like
`battery`, `update`, `icloud`, `warranty`, `refund`, `scam`, `photos`,
etc. appeared. This surfaced two categories that would have been
missed by intuition alone:
- **`phishing_scam_report`** — hundreds of messages like "is this a
  scam," "is this legit," "phishing email" — a real, distinct, and
  **security-sensitive** category.
- **`photos_media_sync`** — hundreds of messages about lost photos or
  iCloud Photos not syncing — big and specific enough to deserve its
  own bucket instead of being folded into generic account issues.

**Final taxonomy (9 intents):**
`software_update_issue`, `battery_performance`,
`device_hardware_issue`, `account_security`,
`warranty_repair_request`, `billing_subscription_refund`,
`photos_media_sync`, `phishing_scam_report`, and `general_inquiry`
(catch-all for everything else — how-to questions, feature requests,
sales/upgrade-program questions, and genuinely miscellaneous traffic).

**Honest limitation:** even after the thread-filter fix, `general_inquiry`
absorbs **~59% of real traffic**. This is not hidden — real Twitter
support traffic is genuinely long-tailed (app-specific bugs, sarcasm,
non-English text, messages replying to an image with no other context).
A 9-bucket keyword taxonomy will always leave a large uncategorized
remainder; narrowing this further would need either much more
hand-labeled data or a learned (not rule-based) sub-classifier.

---

## 4. Step 3 — Labeling data at scale (weak supervision) + a genuine
   golden set

There is no pre-existing "intent" label in the raw data, and hand-labeling
all 45,366 training examples wasn't feasible. Two different, honestly
distinct methods were used for two different purposes:

### 4a. Weak (rule-based) labels for the ~45k training pool

`src/real_taxonomy.py` implements an ordered set of regex rules (e.g.
if the text matches a phishing-related phrase, label
`phishing_scam_report`; else if it matches a battery-related phrase,
label `battery_performance`; etc. — first match wins, checked in a
specific priority order so e.g. "is this apple id email a scam"
correctly goes to `phishing_scam_report` and not `account_security`).
Applied to all 45,366 messages to bootstrap training labels. These are
called **weak** labels on purpose — they were never claimed to be human
-verified at this scale.

### 4b. A genuine, spot-check-corrected golden evaluation set (234 examples)

1. **Stratified sample:** ~26 examples per intent bucket (using the
   weak labels), so rare real categories (e.g. `phishing_scam_report`,
   `photos_media_sync` — each under 1% of raw volume) get real
   evaluation coverage instead of being drowned out.
2. **Manual spot-check:** read 34 of these 234 examples by hand, blind
   to what the weak rule had labeled them, and independently judged the
   correct intent. Found **6 real mislabels** — e.g. "why my charge
   goes from 16% to 9% in 1 minute" had been bucketed as
   `billing_subscription_refund` because "charge" is an overloaded word
   (it meant *battery* charge, not a monetary charge) — corrected to
   `battery_performance`.
3. **Result: ~82% agreement** between the weak labeler and a human
   reader on this spot-check sample. This number is stated explicitly
   in the report rather than hidden, because it caps how much the
   downstream classifier's accuracy can be trusted (see Section 6).

**What this deliberately is NOT claimed to be:** all 234 golden
examples were not individually re-read from scratch — that would be a
stronger, slower methodology. The report states precisely which one
was actually done.

---

## 5. Step 4 — Intent classifier

**Model: TF-IDF (unigrams + bigrams) + Logistic Regression**
(`src/classify_real.py`), not a per-message LLM call. Reasoning: at
real support-traffic volume, an LLM call for every incoming message
adds meaningful latency and cost for a task a linear model already
handles well; LLM calls are reserved for the reply-*drafting* step,
where they earn their cost.

**How it was trained:**
1. Excluded the 234 golden-set examples from the training pool (no
   leakage).
2. Held out 10% of the remaining ~44,788 weakly-labeled examples as an
   internal validation split.
3. Trained `TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=8000)`
   → `LogisticRegression(C=3.0, class_weight="balanced")` (class
   weighting matters because `general_inquiry` dominates the raw
   counts).

**Results on the golden set (n=234):**

| Model | Accuracy | Macro F1 |
|---|---|---|
| Trivial baseline (always predict majority class) | 12.0% | 2.4% |
| TF-IDF + Logistic Regression (ours) | **92.7%** | **92.8%** |

Per-class detail worth noting: `general_inquiry` precision is only
0.69 (recall 1.00) — the model over-predicts the catch-all bucket,
meaning some genuine `warranty_repair_request` messages (recall 0.77)
get swallowed into "general" rather than actively mis-routed elsewhere.
That's a relatively safe failure direction, but it is real information
loss.

---

## 6. Step 5 — Grounded reply drafting

**Approach:** BM25 keyword-retrieval over historical
(customer message → brand resolution) pairs, then draft a reply
grounded in the top-3 retrieved cases. Two execution paths, both
implemented in `src/draft_reply.py`:
- **LLM-grounded generation** (the intended system): calls the
  Anthropic API with the retrieved cases as the *only* allowed source
  of policy/steps, so the model can't invent commitments. This path
  requires `ANTHROPIC_API_KEY`, which was **not available** in the
  build/test environment, so it has **never actually been run**.
- **Extractive fallback** (what was actually tested): reuses the
  single best-matching historical resolution verbatim. Clearly labeled
  as a fallback in every output row, never presented as the real system.

**How retrieval quality was measured, and two bugs found along the way:**
1. First metric attempt (`grounding_overlap`) compared the *draft* to
   its *source* — but for the extractive fallback, draft **is** the
   source (verbatim), so this metric was trivially 1.0 every time,
   regardless of how good or bad the actual match was. This was a flaw
   in the metric's design, caught by noticing the number never moved.
2. Replaced with `retrieval_relevance`: overlap between the **new
   customer message** and the **retrieved historical customer
   message** — this actually varies and reflects real retrieval quality.
3. Second bug: the retrieval corpus initially still contained the
   golden set's *own* tweets, so many golden examples were retrieving
   themselves (relevance artificially ≈0.97). Fixed by excluding
   golden-set tweet IDs from the retrieval corpus — same class of
   leakage bug already guarded against on the classifier's training
   side, just missed here initially.

**Real result after both fixes:** mean `retrieval_relevance` = 0.225,
median 0.206; only 1.7% of examples got a near-zero (likely bad) match.

---

## 7. Step 6 — Evaluating reply quality: automated metrics + LLM-as-judge

**Rubric (`eval/evaluate_replies_real.py`, `eval/judge_agreement_real.py`):**
each draft scored 1-5 on four dimensions — relevance, grounded, tone,
actionable.

**No API key means the "judge" is a heuristic stand-in**, not a real
LLM judge — clearly labeled as such (`judge_type` column in every
output). The heuristic scores based only on token overlap between
draft and source, which (as above) is close to meaningless for the
extractive-fallback path.

**Judge-vs-human agreement — the single most important finding in this
whole project:** 40 real drafts were hand-scored (blind to the
heuristic's own scores) on the same rubric.

| Metric | Value |
|---|---|
| Exact-score agreement (relevance) | 40.0% |
| Pass/fail agreement (relevance≥4 AND grounded≥4) | 55.0% |
| Heuristic judge's own pass rate | 100% |
| **Human's actual pass rate** | **55.0%** |

In plain terms: **the heuristic judge insists every reply is correct,
100% of the time — because it only checks "is the draft literally the
retrieved source" (always yes by construction). A human reading the
same 40 replies judged only 55% of them as actually addressing the
customer's message correctly.** Concrete examples a human catches
instantly that the heuristic completely misses: a question about
"how do I click next when creating this Apple ID" retrieved and
returned a reply about **calendar entries**; a message about
Instagram/Snapchat crashing retrieved a reply pointing to the
**Online Sales team**; a question about missing Audio Books retrieved
a reply about **Visual Voicemail**. All three would ship as-is under
the current fallback and are simply wrong.

This is the strongest evidence in the whole project for why a real
LLM-as-judge (not a heuristic) — and a real LLM-generation path instead
of the extractive fallback — matter, and it's exactly the kind of thing
that's invisible on clean synthetic data and only shows up once you
test against something real.

---

## 8. Step 7 — Escalation decision, with stated reasons

**Design: rule-based, not a learned classifier**
(`src/escalate.py`). With only ~35 true escalation examples even in
the 234-example golden set, a trained model would badly overfit; rules
built from reading actual escalation-worthy real messages are
auditable and directly editable by a support lead without retraining
anything.

**Rules, checked in this priority order (first match wins):**
1. **Explicit human request** — "talk to a human," "not a bot."
2. **Legal/regulatory** — "lawyer," "sue," "fraud," "consumer protection."
   *(Real-data finding: people genuinely do say "I'll sue you" — often
   sarcastically. Regex cannot tell sarcasm from a real threat; this is
   accepted as an explicit trade-off, not silently ignored — better to
   over-escalate a joke than miss a real threat.)*
3. **`phishing_scam_report` intent → always escalates**, regardless of
   classifier confidence. This is the one rule shaped differently from
   all the others: confidence about the *topic* (this is about a scam)
   says nothing about confidence in *is it actually a scam*, and a
   security-sensitive category should never be auto-answered.
4. **Repeated contact** — "third time," "no progress," "hung up on."
5. **High-severity sentiment** — "furious," "worst experience,"
   "unacceptable," "ruined my."
6. **Low classifier confidence** (below 0.55) — safer to have a human
   confirm intent than guess.
7. **Billing + large/disputed amount** — refund authority above a
   threshold (or an explicit "charged twice" claim) is treated as out
   of scope for auto-handling.

**How this was evaluated — and why NOT with precision/recall:** there
is no independent "should this have escalated" ground truth for real
threads (nobody annotated the dataset for this, and this project's
author is not a real AppleSupport agent who knows what should have
happened next). Computing precision/recall against a label generated
by the same rule being tested would be circular and meaningless.
Instead, a **manual audit** was done:
- Read **all 35** messages the rules currently flag (100% of positives)
  — every one judged a reasonable call on reading.
- Scanned the auto-handle bucket with a *broader, looser* keyword net
  than the actual rules use ("cancel," "hate," "terrible," "mad,"
  "stupid," "useless") to look for likely misses — found 13 candidates,
  most reasonably left auto-handled, but this scan is exactly how the
  `"terrible experience"` trigger phrase (previously missing) was found
  and added.
- Real escalation-trigger phrases are genuinely rare: only ~0.3% of
  raw messages match any pattern — real customers rarely narrate their
  own anger the way a synthetic generator's templates do.

---

## 9. Automated tests

`tests/test_agent.py` — 17 pytest tests covering: every escalation
rule (including priority ordering — e.g. an explicit human request
must win over a low-confidence trigger even when both would fire), the
weak labeler's rule ordering (phishing checked before generic account
patterns), text-cleaning functions, and the thread-pairing fix (a
constructed mini-thread confirms mid-thread replies are correctly
excluded when `top_level_only=True`). All 17 pass
(`python -m pytest tests/ -v`).

---

## 10. Top failure modes (ranked by importance)

1. **Retrieval/draft quality has a real, large gap** — only 55% human
   pass rate on the current extractive-fallback path (Section 7). This
   is the single most important number in this report.
2. **`general_inquiry` catch-all absorbs ~59% of real traffic** even
   after the thread-pairing fix — a taxonomy/data-coverage limit, not
   a code bug (Section 3).
3. **The intended LLM-generation and real LLM-judge paths have never
   been run** — no API key available in the build environment. Given
   finding #1, this is the top priority to test next.
4. **Weak training labels cap classifier reliability** — ~18%
   measured disagreement rate with a human reader on the training
   signal (Section 4b); some of the classifier's "errors" against the
   golden set may be correctly learning a wrong pattern from noisy data.
5. **No certified escalation precision/recall** — only a manual audit
   exists (Section 8), because no independent ground truth is available
   for real threads.
6. **Escalation rules cannot distinguish sarcasm from genuine legal
   threats** — accepted trade-off, not fixed.
7. **Money-amount detection is still incomplete** — handles $/£/€
   symbols and "charged twice" phrasing, but not spelled-out amounts
   ("twenty dollars") or amounts split across multiple messages in a
   thread.
8. **Multi-turn/thread-history escalation signal is implemented but
   not evaluated** — `decide_escalation_with_history()` exists and is
   unit-tested, but computing real per-customer prior-contact counts
   would require scanning the full 3M-row file by author, which wasn't
   done in this pass.

---

## 11. What one more week would go toward, in priority order

1. Get an API key wired in and actually run the LLM-grounded draft
   path + real LLM-as-judge — directly tests whether the intended
   system fixes the 55%-pass-rate problem found in the fallback.
2. Get independent escalation ground truth (a second person labeling
   150-200 real messages blind to the rule's output) to compute a real
   precision/recall instead of a manual audit.
3. Evaluate `decide_escalation_with_history()` against real per-author
   contact counts.
4. Subdivide `general_inquiry` using topic clustering over its ~27k
   real messages, rather than guessing more keyword rules.
5. Re-run the weak-label spot-check at a larger scale (100-200
   examples instead of 34) for a tighter agreement estimate, and
   consider a semi-supervised bootstrap to improve the training labels
   themselves.

---

## 12. What was explicitly not built, and why

- No sentiment/emotion model beyond keyword-based severity triggers —
  overkill for a rule that mainly needs to catch a handful of phrase
  patterns.
- No multi-turn dialogue state tracking beyond the (untested) history
  signal above — each message is currently scored independently.
- No ungrounded "free generation" baseline for replies — the
  assignment specifically asks for replies grounded in historical
  resolutions, so retrieval-then-draft is treated as the actual target
  system, with the extractive fallback as the comparison baseline.
- Banking77 (the assignment's optional secondary dataset) was not
  used — `huggingface.co` was unreachable from the build environment
  and the file was not separately provided.
