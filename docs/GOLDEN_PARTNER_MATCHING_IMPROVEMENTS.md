### Improving golden partner matching (sales pitch quality)

Today, partner matching is primarily:
- an LLM prompt that sees the target company attributes + a list of partner summaries
- and returns `matched_partner_name` + `match_score` + rationale.

This works, but it can drift:
- wrong partner chosen → pitch sounds “for a different company”
- LLM may overfit to superficial keywords

---

### What we want (behavioral goals)

- **More deterministic**: similar input → similar match.
- **Fewer “obviously wrong” matches**: guardrails + thresholds.
- **Cheaper**: don’t send 47 partners every time; pre-filter first.
- **Auditable**: store why a partner was chosen (both heuristic + LLM rationale).

---

### Proposal: hybrid matching (fast filter → LLM rerank)

#### Stage 1 (cheap, deterministic): heuristic pre-filter
Build a short-list of partners (e.g. top 8–12) using non-LLM signals:
- Industry keyword overlap
- Product/service overlap
- Exclude partners with obvious mismatches (B2C vs B2B, geography, etc.)

Implementation options:
- TF-IDF cosine similarity over partner descriptions + extracted attributes
- Weighted keyword scoring

This stage yields:
- `candidate_partner_names` (ranked)
- `candidate_partner_scores` (numbers)

#### Stage 2 (LLM, small prompt): rerank + explanation
Send only the short-list to the LLM and require:
- choose the best partner,
- choose a match strength (High/Med/Low),
- cite top 3 shared features.

---

### Guardrails (prevent wrong pitches)

Add explicit rules:
- If match_score is Low / Unknown → **do not generate a pitch**, or generate a “generic” pitch template.
- Require at least N shared features.
- Optionally require the company summary mentions B2B if you’re targeting B2B-only partners.

This turns “wrong partner → wrong pitch” into “no pitch → safe”.

---

### Output improvements (auditing)

Add columns:
- `match_method` (e.g. `hybrid_v1`)
- `candidate_partners` (JSON list, top N)
- `candidate_partner_scores` (JSON list)
- keep `match_reasoning` from LLM as-is

---

### Next step recommendation

Implement Stage 1 pre-filter (deterministic) + Stage 2 rerank (LLM) + guardrail threshold.
This is the biggest win per unit complexity.

