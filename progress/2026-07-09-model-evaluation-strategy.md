# Model Evaluation Strategy — Emotional Premium

## Problem

Emotional premium (情绪溢价) has **no ground-truth labels** — there is no database of "true"
premium values to train or benchmark against.  Evaluation therefore requires multiple
independent lines of evidence, converging on a confidence assessment rather than a
single metric.

## Strategy Overview

```
                    ┌──────────────────────────────┐
                    │  4. Business Outcome          │  ← validates real-world utility
                    │  (price alignment, risk, tier)│
                    ├──────────────────────────────┤
                    │  3. Human-in-the-Loop         │  ← validates subjective quality
                    │  (expert ranking, pairwise)   │
                    ├──────────────────────────────┤
                    │  2. Proxy-Label Performance   │  ← validates statistical signal
                    │  (XGBoost on price-tier proxy)│
                    ├──────────────────────────────┤
                    │  1. Internal Consistency      │  ← validates data integrity
                    │  (coverage, monotonicity)     │
                    └──────────────────────────────┘
```

---

## Layer 1: Internal Consistency Checks

Cheap, automatable, run after every Phase 2 batch.  Must pass before any downstream
work.

| Check | Method | Pass Criterion |
|-------|--------|----------------|
| Feature coverage | QA report from `cli.py` | 4 VLM + 5–8 official fields populated per skin; 0 range violations |
| Monotonicity | Higher `official_tier` → higher `vlm_art_quality` / `vlm_effect_score` | Spearman ρ > 0.6 |
| Degradation audit | Compare CV-fallback L1 vs Ollama L1 on same skins | Same quality tier ≥ 80% of the time |
| Cache stability | Run same image twice | Identical feature values (content-hash validated) |

**Implementation**: add a `--validate` flag to the batch CLI that runs these checks and
reports violations to the QA report.

---

## Layer 2: Proxy-Label Performance (XGBoost)

Since the true premium is unobservable, train against **proxy targets** that correlate
with premium willingness-to-pay.

### Proxy Labels Available Today

| Proxy | Source | Signal | Type |
|-------|--------|--------|------|
| Price ratio | `avg_spend_to_obtain / base_price` | Higher ratio → higher premium | Regression |
| Price tier | `official_tier` (0–5) | Ordinal signal of rarity/value | Ordinal classification |
| Social signal | Weibo meaningful-comment count per skin | Community excitement / buzz | Regression |
| Premium binary | Limited + high-tier (`official_tier ≥ 3`) | Premium vs standard | Binary classification |

### Metrics

```
Regression (price ratio / social signal):
  - R² on held-out 20% test set
  - MAE, RMSE
  - Feature importance ranking vs AHP weights

Classification (premium vs standard):
  - ROC-AUC, precision@k, recall@k
  - Lift curve: top-20% predicted skins capture X% of actual premium skins

Ordinal (tier 0–5):
  - Mean absolute tier error
  - Confusion matrix (off-by-one tolerance)
```

### AHP Weight Alignment Check

The docs define initial AHP weights:
```
Aesthetic    0.30
Showing-off  0.25
Belonging    0.20
Collection   0.15
Surprise     0.10
```

After XGBoost training, aggregate `feature_importance_` (gain) by group and compare:

```python
import xgboost as xgb
importance = model.get_booster().get_score(importance_type='gain')
# Aggregate by FEATURE_GROUPS, normalize, compare to AHP
```

A large divergence is not necessarily wrong — it means XGBoost found signal the AHP
missed.  But every divergence must be explainable.

**Pass criterion**: aesthetic group is the dominant signal; surprise is the weakest.
Pearson correlation between AHP weights and learned importance > 0.7.

---

## Layer 3: Human-in-the-Loop Validation

The most important layer for a subjective task.  Requires 3 domain experts (experienced
game players/analysts familiar with Honor of Kings skins).

### 3a. Expert Ranking Test (15–20 skins)

```
Procedure:
  1. Select 15 skins spanning all 5 quality tiers (3 per tier)
  2. Each expert ranks them independently by "emotional premium" (1 = lowest, 15 = highest)
  3. Model ranks the same 15 skins by predicted premium score
  4. Compute Kendall τ between each expert ranking and model ranking

Target: mean τ > 0.7 across experts
```

### 3b. Pairwise Preference Test (20 pairs)

```
Procedure:
  1. Generate 20 random skin pairs from different quality tiers
  2. Experts pick "which skin commands higher emotional premium" for each pair
  3. Model picks the same (higher predicted score wins)

Target: > 75% agreement rate with expert majority
```

### 3c. Radar Chart Sanity Check (5 skins)

```
Procedure:
  1. Pick 5 representative skins (one per tier: 伴生, 史诗, 传说, 无双, 荣耀典藏)
  2. Show the 5-dimensional radar chart (aesthetic, belonging, showing-off,
     collection, surprise) for each skin
  3. Ask experts: "Does this profile match your intuition for this skin?"

Target: qualitative — flags dimension-level miscalibration that aggregate
scores would hide.
```

Expert agreement should be measured both as:
- **Inter-rater reliability** (Fleiss' κ) — are experts consistent with each other?
- **Model-expert agreement** (Kendall τ) — does the model align with human consensus?

---

## Layer 4: Business Outcome Validation

These are the metrics B2B customers (game studios, publishers) care about.

| Test | Method | Pass Criterion |
|------|--------|----------------|
| **Price alignment** | Compare model-suggested price range to actual market price | ≥ 80% of skins fall within suggested range |
| **Tier discrimination** | Model scores for 伴生 vs 荣耀典藏 skins | Non-overlapping distributions (Welch's t-test p < 0.001) |
| **Risk flag accuracy** | Compare model's risk flags against known community-backlash incidents | Recall > 80% on known incidents |
| **Competitor differentiation** | For skins with known competitor analogs, does the model correctly identify the differentiating factor? | Qualitative review (spot-check 10 pairs) |

---

## Implementation Path (Phase 3a–3c)

### Phase 3a: Proxy Label Construction (NEW)

```
models/
  labels.py         — price_ratio, tier_class, community_signal constructors
  dataset.py        — merge Phase 2 features + labels → training DataFrame
```

### Phase 3b: Baseline Models (MODIFY)

```
models/
  rule_engine.py    — AHP-weighted sum → premium index [0,100]
  xgboost_model.py  — train on each proxy label, compare feature importance
```

### Phase 3c: Evaluation Dashboard (NEW)

A Streamlit page (`app.py` extension) showing:
- Tab 1: Internal consistency results (from CLI `--validate`)
- Tab 2: Proxy-label model performance (R², AUC, lift curves)
- Tab 3: Expert ranking UI (skin pair comparison tool)
- Tab 4: Feature importance vs AHP weight comparison chart

---

## Summary: Is the Model Effective?

The model is **not** validated by a single number.  Effectiveness is a **convergence**
across four lines of evidence:

| Layer | If Failing | Root Cause |
|-------|-----------|------------|
| Internal consistency fails | Data pipeline is broken — fix before anything else |
| Proxy-label R² < 0.3 | Features lack signal — re-examine VLM quality, add external data |
| Expert τ < 0.5 | Model doesn't capture human intuition — recalibrate weights or features |
| Business metrics fail | Model is statistically sound but commercially useless — repivot |

A model that passes layers 1–3 but fails layer 4 still has value as a **decision-support
tool** (ranking skins, identifying outliers).  A model that passes all four layers can
be confidently deployed as a **pricing recommendation engine**.
