# Decision Log

Non-obvious decisions made while building this, and why.

1. **Built against synthetic data shaped like the real Kaggle schema,
   rather than a toy dataset with a different structure.** This sandbox
   can't reach kaggle.com/huggingface.co, so the real file was
   unavailable. Rather than build against arbitrary made-up data, the
   generator (`data/generate_sample_data.py`) outputs the exact same
   columns (`tweet_id, author_id, inbound, created_at, text,
   response_tweet_id, in_response_to_tweet_id`) so `src/ingest.py`'s
   stream-filter-then-thread logic is the same code path that would run
   on the real file — swapping the CSV path is the only change needed.

2. **Chose AppleSupport over a retail/airline brand.** AppleSupport has
   clearly delineated technical issue types (battery, update, device
   dead, account, warranty, billing) that map cleanly to a small intent
   taxonomy, and (in the real dataset) long, detailed resolution
   threads that are good grounding material for retrieval — better
   than short "DM us" brush-off replies common on some retail brand
   accounts.

3. **7 coarse intents, not Banking77-style fine-grained (77) intents.**
   The assignment says "a small set of intents you define from the
   data" — for a single brand's Twitter traffic, 6-9 topical buckets is
   what's actually separable from tweet text alone; more granularity
   would mostly be discriminating on phrasing, not real topic
   difference, and would require far more labeled data than 150-250
   golden examples can support.

4. **TF-IDF + Logistic Regression as the actual classifier, not an LLM
   call per message.** At real support-traffic volume, calling an LLM
   to classify every inbound message adds latency and cost per message
   for a task a linear model already does at >90%+ on n-grams (see
   report.md — with the caveat that this number is inflated by
   synthetic template repetition). LLM calls are reserved for the
   generative drafting step, where they earn their cost.

5. **Rule-based (not learned) escalation layer.** With ~35-37 true
   escalation examples in a 196-example golden set, training a
   dedicated classifier would badly overfit. Rules built from reading
   actual escalation-worthy messages are auditable, and a support lead
   can add/edit a trigger phrase without retraining anything — that
   matters more than a marginal accuracy gain at this data volume.

6. **Retrieval filtered/reasoned about per predicted intent, not global
   retrieval over all historical resolutions.** Grounding a battery
   complaint's draft on a billing resolution because of surface-level
   lexical overlap would be a worse failure than the intent classifier
   being slightly wrong — retrieval should search within the topic
   bucket first.

7. **Extractive fallback when no `ANTHROPIC_API_KEY` is present, rather
   than skipping the reply-drafting step entirely or faking an LLM
   response.** The harness needs to run end-to-end to be gradeable;
   the fallback is clearly labeled in every output row (`method` /
   `judge_type` columns) so it's never mistaken for the real generative
   path.

8. **Money-threshold rule ($20) for billing escalation**, rather than
   escalating all billing messages or none. Small billing questions
   ("why was I charged $2.99") are usually a quick account-standing
   lookup an auto-agent can point to `reportaproblem.apple.com` for;
   larger disputed amounts carry more reputational/financial risk to
   get wrong and should have a human confirm before any reply implies a
   refund process.

9. **Confidence threshold (0.55) for "escalate on low classifier
   confidence," picked by inspection, not tuned.** Flagged explicitly
   as untuned in report.md — we didn't have a large enough confidently-
   ambiguous example set to do a real threshold sweep, and picking one
   is safer than not having the rule at all.

10. **Judge-vs-human agreement measured as pass/fail (≥4 both dims)
    AND exact-score, not just one.** Exact-score agreement is the
    stricter, more informative number (and it exposed a real
    limitation — 40% — that pass/fail (93%) would have hidden). We
    report both rather than picking whichever looks better.

11. **Golden set sampling is stratified by intent + escalation-topped-up,
    not a uniform random sample of all threads.** A uniform sample would
    just reproduce the generator's class balance (which is
    intentionally uniform-per-intent) and under-represent the ~18%
    minority escalation class — see `eval/build_golden_set.py`'s
    docstring for the full sampling rationale.

12. **~4% deliberate label noise injected into the golden set.** A
    "hand-labeled" set drawn straight from generator metadata with zero
    disagreement would be an artificially clean eval — real human
    labelers disagree with themselves and each other on genuinely
    ambiguous cases. This keeps the intent-classification headline
    number from reading as more trustworthy than it should.

13. **Excluded escalated messages from reply-drafting evaluation
    entirely** (`eval/evaluate_replies.py` skips rows where
    `escalate=True`), rather than drafting-then-discarding a reply for
    them. An agent that decides to escalate shouldn't also be scored on
    reply quality for a message it correctly decided not to answer.

14. **Chose to build `eval/evaluate_escalation.py` mid-build after
    initially leaving it as a known gap in a draft of report.md.**
    Precision/recall/F1 against golden labels is a 20-line script and
    much stronger evidence than "spot-checked on 4 examples" — worth
    the extra half hour rather than shipping the weaker version.

15. **Repo has no single "notebook" — everything is importable modules
    (`src/`, `eval/`) run via `python -m`.** Keeps the README's
    reproduce-in-under-15-minutes path to a short list of commands
    rather than "open this notebook and run all cells," and makes each
    stage (ingest, classify, draft, escalate, evaluate) independently
    testable, which is how this was actually built and debugged.

## Real-data pass (after the user uploaded the real Kaggle archive)

16. **Re-derived the intent taxonomy from a keyword-frequency scan of
    real data, rather than reusing the synthetic taxonomy.** Two
    categories (`phishing_scam_report`, `photos_media_sync`) only
    became obvious once we actually looked at what real AppleSupport
    customers write — a taxonomy designed before touching real data is
    a taxonomy designed against our own assumptions, not the data.

17. **Restricted thread-pairing to thread-initiating messages only
    (`top_level_only=True`, now the default).** The naive pairing
    (any customer message with a `response_tweet_id`) captured
    mid-thread filler like "thanks!" as if it were a fresh incoming
    issue. This single fix dropped the candidate pool from 97,153 to
    45,366 and was a bigger single improvement to label quality than
    any taxonomy or rule change.

18. **Golden-set labeling used spot-check-corrected weak labels, not
    full independent hand-labeling of all 234 examples.** Given real
    time constraints, we read 34 examples blind to the weak label,
    corrected 6 real mislabels (~82% agreement), and applied those
    specific corrections — rather than either (a) trusting all 234
    weak labels uncritically, or (b) claiming a stronger methodology
    ("all 234 hand-labeled from scratch") than we actually did. Stating
    the real methodology precisely matters more than making the number
    look more rigorous than it is.

19. **Expanded escalation trigger patterns using a manual audit of
    real data, not just reasoning about what triggers "should" exist.**
    Found and added `"terrible experience"` and fraud/double-charge
    patterns this way; also discovered that real "I'll sue you"
    mentions are often sarcastic, which we can't distinguish with
    regex — documented as an accepted, explicit trade-off rather than
    quietly leaving the false-positive risk unstated.

20. **Made `phishing_scam_report` intent always escalate, regardless of
    classifier confidence.** This is a genuinely different rule shape
    from every other trigger (which are either text-pattern or
    confidence-based) — a deliberate choice that security-sensitive
    content should never be auto-answered even when the classifier is
    very confident about the intent, because confidence about the
    *topic* says nothing about confidence in *is this actually a scam*.

21. **Replaced the `grounding_overlap` metric after discovering it was
    trivially 1.0 by construction** (it compared the extractive
    fallback's draft to its own source — literally the same string).
    Replaced with `retrieval_relevance` (customer message vs. retrieved
    past customer message), which actually varies and is informative.
    Left the broken original in v1's `eval/evaluate_replies.py`
    untouched with a note, rather than silently deleting evidence of
    the mistake — the report documents both the bug and the fix.

22. **Excluded the golden set's own tweets from the retrieval corpus**
    after finding they were being retrieved by themselves (mean
    relevance was a meaningless 0.974 before this fix, same class of
    leakage bug `src/classify_real.py` already guarded against for
    training data, just missed on the retrieval side initially).

23. **Used a manual audit (not a fabricated precision/recall) for
    escalation evaluation on real data**, because no independent
    "should this have escalated" ground truth exists for real threads
    and computing metrics against a label generated by the same rule
    being tested would be circular and worthless — see
    `eval/evaluate_escalation_real.py`'s docstring for the full
    reasoning. A weaker, honest result beats a stronger, meaningless one.

24. **Ran the exact same judge-vs-human agreement methodology on real
    data as was used on synthetic data (v1), rather than a different,
    easier check**, specifically so the two numbers (93.3% vs. 55.0%
    pass agreement) would be directly comparable and make the
    synthetic-data overstatement visible rather than obscured by a
    methodology change.
