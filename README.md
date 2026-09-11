# Hiver SDE Intern Assignment — AppleSupport AI Support Agent

An AI support agent for `@AppleSupport` that classifies inbound Twitter
messages into intents, drafts a reply grounded in how the brand has
historically resolved similar issues, and decides auto-handle vs.
escalate-to-human with a stated reason.

**Read `report/report.md` first** — it documents what's real, what's
still a known gap, and (in its Section 4 table) exactly which
limitations from an earlier review were fixed vs. still open.

This repo has **two parallel pipelines**:
- `*_real.py` files: run on the actual Kaggle
  `thoughtvector/customer-support-on-twitter` dataset (real ~3M-row
  file). **This is the one that matters** — read the report.
- Files without `_real`: an earlier pass built on synthetic data before
  the real dataset was available, kept for reference and because it's
  still a faster smoke-test of the pipeline logic.

## What's in this repo

```
data/
  generate_sample_data.py          # generates synthetic sample (fallback pipeline)
  real_taxonomy.py → see src/      # (taxonomy lives in src/, not data/)
src/
  ingest.py             # loads twcs-schema CSV, reconstructs thread pairs (top_level_only fix)
  real_taxonomy.py       # intent taxonomy + weak labeler, derived from real keyword frequencies
  classify.py / classify_real.py   # intent classifier: synthetic / real pipelines
  draft_reply.py         # BM25 retrieval + grounded reply drafting (shared by both pipelines)
  escalate.py             # rule-based escalation decision with stated reasons (real-data-tuned)
  pipeline.py             # ties it together: SupportAgent.handle(message) -> full decision
eval/
  build_golden_set.py / build_golden_set_real.py   # golden set: synthetic / real
  evaluate_replies.py / evaluate_replies_real.py   # reply quality eval: synthetic / real
  evaluate_escalation.py / evaluate_escalation_real.py  # escalation eval: synthetic (P/R) / real (audit)
  judge_agreement.py / judge_agreement_real.py     # judge-vs-human agreement: synthetic / real
tests/
  test_agent.py           # pytest suite: escalation rules, taxonomy, ingest threading (17 tests)
report/report.md          # v2: real-data results, "kaami → fix" table, next steps
decision_log.md           # 24 non-obvious decisions and why
```

## Reproduce the REAL-DATA results (should take < 15 minutes, dataset excluded)

The real dataset (`twcs.csv`, ~500MB) isn't included in this repo —
download it from Kaggle (`thoughtvector/customer-support-on-twitter`)
first.

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn rank_bm25 joblib pytest

# 2. Place the real dataset at data/real_twcs.csv (twcs.csv from Kaggle, renamed)

# 3. Build the real customer<->brand pairs and run the weak labeler
python3 -c "
from src.ingest import load_brand_pairs
load_brand_pairs('data/real_twcs.csv', 'AppleSupport', top_level_only=True).to_csv('data/real_apple_pairs.csv', index=False)
"
python3 -m src.real_taxonomy

# 4. Build the real golden evaluation set (234 examples, weak-labeled +
#    spot-check corrected — see eval/build_golden_set_real.py docstring)
python3 -m eval.build_golden_set_real
python3 -c "from eval.build_golden_set_real import apply_spot_check; apply_spot_check()"

# 5. Train the intent classifier (weak-label trained) and evaluate vs. trivial baseline
python3 -m src.classify_real

# 6. Run the real reply-drafting eval (retrieval relevance + judge)
python3 -m eval.evaluate_replies_real

# 7. Run the escalation manual audit
python3 -m eval.evaluate_escalation_real

# 8. Run the test suite
python3 -m pytest tests/ -v

# 9. Try the full end-to-end agent on real-shaped example messages
python3 -m src.pipeline
```

Expected headline numbers (see `report/report.md` Section 3-4 for full
context and honest caveats on each):
- Intent classification: 92.7% accuracy vs. 12.0% trivial baseline
- Escalation: 35/234 flagged, 100% manually audited as reasonable calls
  (not a certified precision/recall — see report)
- Reply quality: only **55%** human-judged pass rate on the current
  extractive-fallback drafting path — this is the most important
  number in the whole report, see Section 3

## Reproduce the SYNTHETIC-DATA results (no download needed, ~5 min)

Kept as a fast, self-contained smoke test of the pipeline logic. **Do
not treat these numbers as representative of real performance** — see
report.md Section 0 and the "kaami → fix" table for why.

```bash
python3 data/generate_sample_data.py
python3 eval/build_golden_set.py
python3 -m src.classify
python3 -m eval.evaluate_replies
python3 -m eval.evaluate_escalation
python3 -m src.pipeline
```

## Using the LLM-grounded draft path and real LLM-as-judge

Both `src/draft_reply.py` and `eval/judge_agreement*.py` will
automatically use the Anthropic API instead of their documented
fallbacks the moment `ANTHROPIC_API_KEY` is set — no code changes
needed:

```bash
export ANTHROPIC_API_KEY=sk-...
python3 -m eval.evaluate_replies_real
```

This is the single highest-priority next step per the report — we have
direct evidence (55% human pass rate) that the current extractive
fallback isn't good enough, but haven't yet been able to test whether
the intended generative path does better.

## What's genuinely NOT done

Stated plainly rather than left to be discovered:
- No real LLM generation or real LLM-as-judge has ever actually run in
  this environment (no API key available here).
- Escalation has no certified precision/recall — real threads have no
  independent "should this escalate" ground truth to measure against.
- Banking77 (optional secondary dataset) was not used —
  `huggingface.co` wasn't reachable from the build environment used
  for this project, and the file wasn't separately obtained.
- `decide_escalation_with_history()` (multi-turn signal) is implemented
  and unit-tested but not evaluated end-to-end against real thread
  histories.

## Rules followed / borrowing disclosure

- No external code was copied from another repo or Stack Overflow
  answer; standard library usage (pandas, scikit-learn, rank_bm25) is
  used per each library's documented API.
- AI coding assistance was used throughout, per the assignment's
  explicit permission — happy to walk through and modify any part of
  this live.
