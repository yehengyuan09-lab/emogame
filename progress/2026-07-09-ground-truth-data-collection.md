# Ground-Truth Data Collection & Construction Strategy

## Problem Statement

Emotional premium (情绪溢价) has **no direct observable ground-truth labels** — there is no database of "true" premium values. Evaluation therefore requires constructing **proxy ground-truth labels** from multiple converging evidence sources, with quantified uncertainty for each label.

## Core Principle

> **No single source of truth, but multiple converging lines of evidence with confidence scores.**

Ground-truth labels are constructed as a **consensus-weighted average** of:
1. Expert-labeled gold standard (small subset)
2. Market-based behavioral signals
3. Community engagement metrics
4. Cross-platform consensus
5. Pairwise preference data

---

## 1. Expert-Labeled Gold Standard

### Dataset Scale
- **Target**: 150-200 skins with expert ratings
- **Initial**: 50 skins (10 per tier) for Phase 2.5
- **Stratified sampling** by tier:
  - 伴生: 20%
  - 史诗: 30%
  - 传说: 25%
  - 无双: 15%
  - 荣耀典藏: 10%

### Labeling Protocol

```python
# File: models/expert_labeling.py

class ExpertLabelingProtocol:
    """
    Protocol for collecting expert ratings as ground-truth proxy labels.
    """

    def __init__(self, num_experts: int = 3, num_skins: int = 100):
        self.experts = [...]  # List of expert identifiers
        self.skins = self.stratified_sample_by_tier(num_skins)

    def stratified_sample_by_tier(self, n: int) -> List[Skin]:
        """
        Sample skins proportional to tier distribution.
        """
        pass

    def labeling_session(self, skin: Skin) -> Dict:
        """
        Single skin labeling session with quality controls:
        1. Show full skin visual + official attributes
        2. Show radar chart (5 dimensions from VLM)
        3. Expert rates overall premium (0-100 slider)
        4. Expert rates 5 dimensions individually (0-100)
        5. Optional: Expert comments on decision rationale
        """
        return {
            "expert_id": "...",
            "skin_id": "...",
            "overall_premium": 0-100,  # Primary ground-truth label
            "dimension_ratings": {
                "aesthetic": 0-100,
                "showing_off": 0-100,
                "belonging": 0-100,
                "collection": 0-100,
                "surprise": 0-100
            },
            "comments": "...",
            "timestamp": "...",
            "session_duration_seconds": "..."
        }

    def compute_consensus_score(self, skin_id: str) -> float:
        """
        Aggregate expert ratings using:
        - Remove outliers (>2 SD from mean)
        - Compute trimmed mean
        - Weight by expert confidence (optional)
        """
        pass
```

### Quality Controls

| Control | Method | Target |
|---------|--------|--------|
| Inter-rater reliability | Fleiss' κ for each dimension | κ > 0.6 |
| Time constraint | Maximum 2 minutes per skin | Avoid fatigue bias |
| Calibration session | 10 known-consensus skins first | Establish baseline |
| Blind labeling | Experts don't see each other's ratings | Independent judgment |
| Expert qualification | 500+ hours Honor of Kings playtime | Domain expertise |

---

## 2. Market-Based Proxy Labels

### Transaction-Derived Premium

```python
# File: models/market_labels.py

class MarketPremiumEstimator:
    """
    Derive emotional premium from actual player spending behavior.
    """

    def price_ratio_label(self, skin: Skin) -> float:
        """
        Proxy: avg_spend_to_obtain / base_price

        Higher ratio → players willingly pay more → emotional premium
        """
        if skin.base_price == 0:
            return 0
        return skin.avg_spend_to_obtain / skin.base_price

    def willingness_to_pay_label(self, skin: Skin) -> float:
        """
        Derived from:
        - Secondary market prices (if trading exists)
        - Gacha pull count distribution
        - Time-unlock alternative cost
        """
        pass
```

### Data Sources

| Source | Field | Signal Type |
|--------|-------|-------------|
| Official store | `avg_spend_to_obtain` | Direct willingness-to-pay |
| Gacha mechanics | Pull count distribution | Effort premium |
| Trading platforms | Secondary market price | Residual value |
| Unlock alternatives | Time cost alternative | Opportunity cost |

---

## 3. Community Engagement Labels

### Weibo Excitement Score

```python
# File: models/community_labels.py

class CommunitySignalLabeler:
    """
    Extract emotional signals from social media and community data.
    """

    def weibo_excitement_score(self, skin_id: str) -> float:
        """
        Weibo comment sentiment & engagement score:
        - Positive sentiment ratio
        - Comment count normalized by follower count
        - Keywords: "值了", "必买", "绝美", "炫酷"
        """
        from vlm.schemas import WeiboComment

        comments = self.db.query(WeiboComment).filter(
            WeiboComment.skin_id == skin_id
        ).all()

        # Extract emotional keywords
        positive_keywords = ["值了", "必买", "绝美", "炫酷", "绝绝子", "yyds"]
        negative_keywords = ["不值", "丑", "后悔", "智商税"]

        excitement = 0
        for c in comments:
            # Sentiment analysis (rule-based + VLM semantic)
            # Engagement weight (likes, reposts)
            pass

        return excitement
```

### Cross-Platform Signals

| Platform | Signal | Extraction Method |
|----------|--------|-------------------|
| Weibo | Comment sentiment/engagement | NLP + keyword matching |
| Bilibili | Video view/like ratio | API scraping |
| NGA/贴吧 | Discussion engagement | Thread activity metrics |
| Douyin | Content creation volume | Hashtag count |
| Steam guides | Guide count/quality | Community contribution |

---

## 4. Comparative Preference Labels

### Pairwise Comparisons

```python
# File: models/comparative_labels.py

class PairwiseComparisonLabeler:
    """
    Collect A/B preference data from experts or players.
    """

    def generate_pairs(self, n_pairs: int = 200) -> List[Tuple]:
        """
        Generate skin pairs with controlled characteristics:
        - Same tier, different visual styles
        - Adjacent tiers (史诗 vs 传说)
        - High controversy (likely to generate disagreement)
        """
        pass

    def collect_preference(self, pair: Tuple) -> Dict:
        """
        Expert/player chooses which skin has higher emotional premium.
        Results in partial order constraints.
        """
        return {
            "skin_a_id": "...",
            "skin_b_id": "...",
            "preferred": "a" | "b" | "tie",
            "confidence": 1-5,
            "rationale": "..."
        }

    def BradleyTerry_scores(self, preferences: List[Dict]) -> Dict[str, float]:
        """
        Convert pairwise preferences to scalar scores using
        Bradley-Terry model (like Elo ratings).
        """
        pass
```

### Comparison Design

- **Same-tier pairs**: Isolate aesthetic preference
- **Cross-tier pairs**: Test tier-vs-premium monotonicity
- **Visual contrast pairs**: Different art styles (cyberpunk vs traditional)
- **Controversial pairs**: High disagreement expected → identify edge cases

---

## 5. Cross-Platform Consensus Labels

```python
# File: models/cross_platform_labels.py

class CrossPlatformConsensus:
    """
    Aggregate emotional premium signals across multiple platforms.
    """

    def platform_signals(self, skin_id: str) -> Dict:
        """
        Collect from:
        - Weibo: comment sentiment
        - Bilibili: video view/like ratio
        - NGA/贴吧: discussion engagement
        - Steam/Douyin: content creation
        """
        return {
            "weibo_score": 0-100,
            "bilibili_score": 0-100,
            "forum_score": 0-100,
            "content_creation_score": 0-100
        }

    def consensus_label(self, skin_id: str) -> float:
        """
        Weighted average of platform signals.
        Weights determined by correlation with expert labels.
        """
        pass
```

### Weight Learning

```python
# Weights are learned by maximizing correlation with expert consensus
# Initial weights (from literature):
PLATFORM_WEIGHTS = {
    "weibo": 0.35,
    "bilibili": 0.25,
    "forum": 0.20,
    "content_creation": 0.20
}

# Updated by linear regression on expert-labeled subset
# Weights adapt as more expert labels are collected
```

---

## 6. Label Integration & Validation

### Ground-Truth Label Construction

```python
# File: models/ground_truth.py

class GroundTruthLabel:
    """
    Construct final ground-truth label from multiple evidence sources.
    """

    def __init__(self, skin_id: str):
        self.skin_id = skin_id
        self.evidence = {
            "expert_consensus": self.get_expert_consensus(),
            "market_ratio": self.get_market_ratio(),
            "community_excitement": self.get_community_score(),
            "pairwise_bradley_terry": self.get_pairwise_score(),
            "cross_platform": self.get_cross_platform_score()
        }

    def compute_final_label(self, method: str = "weighted") -> float:
        """
        Method options:
        - "expert_only": Pure expert consensus (gold standard)
        - "market_weighted": Weight by market availability
        - "ensemble": Weight by inter-source correlation
        - "adaptive": Dynamic weights based on skin category
        """
        if method == "expert_only":
            return self.evidence["expert_consensus"]

        # Compute correlation matrix between sources
        # Determine weights by correlation with expert consensus
        # Weighted average
        pass

    def confidence_score(self) -> float:
        """
        Quantify label uncertainty:
        - Low expert agreement → low confidence
        - Conflicting market vs community signals → low confidence
        - High cross-platform consensus → high confidence
        """
        pass
```

### Label Weighting Strategy

| Method | When to Use | Weight Source |
|--------|-------------|---------------|
| `expert_only` | Validation set, model benchmarking | Expert consensus only |
| `market_weighted` | Training set for pricing models | Market data weighted higher |
| `ensemble` | General purpose model | Inter-source correlation |
| `adaptive` | Special skin categories (e.g., IP collabs) | Category-specific weights |

---

## 7. Dataset Construction Pipeline

```python
# File: models/dataset_construction.py

class GroundTruthDatasetBuilder:
    """
    End-to-end pipeline for constructing labeled dataset.
    """

    def build_dataset(
        self,
        n_skins: int = 200,
        include_expert: bool = True,
        include_market: bool = True,
        include_community: bool = True
    ) -> pd.DataFrame:
        """
        Construct dataset with multiple label columns:

        | skin_id | expert_label | market_label | community_label | \
          pairwise_label | consensus_label | confidence |
        """
        skins = self.stratified_sample(n_skins)

        rows = []
        for skin in skins:
            gt = GroundTruthLabel(skin.id)

            row = {
                "skin_id": skin.id,
                "expert_label": gt.evidence["expert_consensus"],
                "market_label": gt.evidence["market_ratio"],
                "community_label": gt.evidence["community_excitement"],
                "pairwise_label": gt.evidence["pairwise_bradley_terry"],
                "consensus_label": gt.compute_final_label(),
                "confidence": gt.confidence_score()
            }

            # Merge with Phase 2 features
            row.update(self.get_feature_vector(skin.id))

            rows.append(row)

        return pd.DataFrame(rows)

    def split_dataset(
        self,
        df: pd.DataFrame,
        expert_holdout: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits with special handling:
        - Expert-labeled skins: 80/20 train/test
        - Market/community only: 90/10 train/test

        Ensure test set has representative tier distribution.
        """
        pass
```

---

## 8. Label Validation Framework

```python
# File: models/label_validation.py

class LabelValidator:
    """
    Validate that constructed ground-truth labels are meaningful.
    """

    def validate_expert_labels(self, df: pd.DataFrame) -> Dict:
        """
        Check expert label quality:
        - Inter-rater reliability (Fleiss' κ) > 0.6
        - Tier monotonicity: 伴生 < 史诗 < 传说 < 无双 < 荣耀典藏
        - Dimension correlation: overall vs dimension ratings
        """
        pass

    def validate_proxy_correlation(self, df: pd.DataFrame) -> Dict:
        """
        Check correlation between proxy sources:
        - expert vs market: Spearman ρ > 0.5
        - expert vs community: Spearman ρ > 0.5
        - market vs community: Spearman ρ > 0.4
        """
        pass

    def label_stability_test(self, n_bootstrap: int = 100) -> Dict:
        """
        Bootstrap resampling to assess label stability:
        - Compute consensus label on 100 random subsamples
        - Mean and SD of label for each skin
        - Flag skins with high label uncertainty (SD > 10)
        """
        pass
```

### Validation Criteria

| Check | Metric | Pass Criterion |
|-------|--------|----------------|
| Expert reliability | Fleiss' κ | κ > 0.6 |
| Tier monotonicity | Mann-Whitney U test | p < 0.001 |
| Proxy correlation | Spearman ρ | ρ > 0.5 (vs expert) |
| Label stability | Bootstrap SD | SD < 10 for 90% of skins |

---

## Implementation Priority

### Phase 2.5 (Before Phase 3 models)

1. **Expert labeling of 50 skins** (10 per tier)
   - Establish gold standard for proxy validation
   - Compute inter-rater reliability
   - Validate tier monotonicity

2. **Market ratio labels** for all skins
   - Cheap, scalable proxy
   - Compute `avg_spend_to_obtain / base_price`
   - Normalize to 0-100 scale

3. **Label validation**
   - Correlation analysis between sources
   - Identify outlier skins (low confidence)

### Phase 3a (Model training)

4. **Expand to 150 expert-labeled skins**
   - Stratified by tier and visual style
   - Include controversial edge cases

5. **Community signal extraction**
   - Leverage existing Weibo data
   - Implement sentiment analysis
   - Add other platforms if available

6. **Consensus label construction**
   - Implement weighted ensemble
   - Compute confidence scores
   - Build training dataset

### Phase 3c (Evaluation)

7. **Pairwise comparison study** (200 pairs)
   - Bradley-Terry model scoring
   - Compare to scalar predictions

8. **Cross-platform signal aggregation**
   - Learn optimal weights
   - Validate against expert labels

9. **Final label uncertainty quantification**
   - Bootstrap stability analysis
   - Flag low-confidence skins

---

## 9. LLM-as-Expert: qwen3-vl-plus as a Standalone Expert

### Why

Recruiting 3 human experts for 150–200 skins is slow and expensive.  With the
AutoDL API available (OpenAI-compatible vision endpoint), we use
**`qwen3-vl-plus`** as a **standalone domain expert** that produces the same
structured 5-dimension premium label a human expert would, at scale and with
full reproducibility (`temperature=0`).  The expert model is configured
separately from VLM L3 (`settings.expert_model` vs `settings.l3_model`), so the
two tiers use **different models** — see the leakage note below.

### What we built (already implemented)

| File | Role |
|------|------|
| `models/llm_expert_labeler.py` | `LLMExpertLabeler` — single-skin + mini-batch labeling, caching, retries, calibration helpers |
| `models/run_expert_labeling.py` | CLI runner — stratified sampling, VLM-context option, JSONL/CSV/QA export |

Run it:

```bash
# 20-skin blind gold-standard set (no VLM peek), 10 per mini-batch
python -m models.run_expert_labeling --limit 20 --batch-size 10 --concurrency 3

# cross-check mode: feed the existing VLM radar into the expert
python -m models.run_expert_labeling --hero 李白 --use-vlm-context

# preview the stratified work queue
python -m models.run_expert_labeling --limit 60 --seed 7 --dry-run
```

Output: `data/expert_labels/expert_labels.jsonl`, `.csv`, and
`expert_label_report.json` (dimension means + tier-monotonicity Spearman ρ).

### Expert label schema (mirrors human protocol)

```json
{
  "skin_key": "0001-56304",
  "overall_premium": 0-100,
  "aesthetic": 0-100, "showing_off": 0-100, "belonging": 0-100,
  "collection": 0-100, "surprise": 0-100,
  "confidence": 1-5, "rationale": "...", "dimension_rationale": {...},
  "source": "llm_expert", "cached": false
}
```

### Positioning in the 4-layer framework

- **Layer 3 (Human-in-the-Loop)**: LLM-expert (`qwen3-vl-plus`) labels are a
  *scalable proxy* for the human-expert gold standard.  Use them to (a) label
  the bulk corpus, (b) serve as the **target** when measuring human-expert
  agreement (Kendall τ), (c) sanity-check the AHP rule engine.
- **Calibration requirement**: keep a small **human-anchored** subset
  (≥15 skins, the eval-strategy's expert-ranking test) and report Kendall τ
  between LLM-expert ranking and human consensus.  The LLM-expert is only
  trusted at gold-standard fidelity once τ > 0.7.

### ⚠️ Leakage caveat (critical)

VLM **L3 uses `gpt-5.4-mini`**; the expert labeler uses **`qwen3-vl-plus`** —
a *different* model.  This means direct label/feature leakage (the model
"knowing" the answer through a sibling of itself) is **substantially reduced**
compared to the original same-model design.  It is not eliminated, because both
models still encode similar visual priors, so the general caution stands:

Mitigations:
1. **Default to blind labeling** (`--use-vlm-context` off) so the expert judges
   from image + official attributes only.
2. For gold-standard training labels, still **exclude L3 features** (use only
   L1/L2 + official + CV features) when the label source is the LLM-expert — a
   different model lowers but does not remove correlated signal.
3. Reserve the **human-anchored subset** (never LLM-labeled) as the hold-out for
   the Layer-3 validation that actually proves deployment readiness.

### Mini-batch design rationale

- `batch_size` (default 10) bounds cost visibility and gives incremental
  progress + a natural checkpoint boundary (each batch = one gather).
- `concurrency` (default 3) caps in-flight calls to respect AutoDL rate limits.
- Per-request retry-with-backoff (max 3) + content-hash cache make reruns free
  and satisfy the Layer-1 cache-stability check.

---

## File Structure

```
models/
├── llm_expert_labeler.py       # qwen3-vl-plus as standalone expert (IMPLEMENTED)
├── run_expert_labeling.py      # Mini-batch CLI runner (IMPLEMENTED)
├── ground_truth/
│   ├── __init__.py
│   ├── expert_labeling.py       # Human expert labeling protocol (planned)
│   ├── market_labels.py         # Market-based proxies
│   ├── community_labels.py      # Community signal extraction
│   ├── comparative_labels.py    # Pairwise comparisons
│   ├── cross_platform_labels.py # Cross-platform consensus
│   ├── ground_truth.py          # Label integration
│   ├── dataset_construction.py  # Dataset builder
│   └── label_validation.py      # Label quality checks
```

---

## Integration with 4-Layer Validation

This ground-truth strategy directly supports the 4-layer validation framework:

| Layer | Ground-Truth Role |
|-------|-------------------|
| **Layer 1: Internal Consistency** | Feature coverage, monotonicity checks |
| **Layer 2: Proxy-Label Performance** | Train XGBoost on consensus labels |
| **Layer 3: Human-in-the-Loop** | Expert labels as validation gold standard |
| **Layer 4: Business Outcomes** | Market labels for price alignment tests |

---

## Key Insights

1. **No absolute truth** — only consensus with quantified uncertainty
2. **Expert labels anchor** — provide validation for proxy signals
3. **Multi-source convergence** — conflicting signals flag uncertainty
4. **Confidence scores** — essential for model trustworthiness
5. **Iterative refinement** — weights update as more data arrives

---

## Next Actions

- [ ] Design expert labeling UI (Streamlit or web form)
- [ ] Recruit 3 expert labelers
- [ ] Implement `expert_labeling.py` protocol
- [ ] Compute market ratio labels for all skins
- [ ] Extract community signals from Weibo data
- [ ] Build initial 50-skin expert-labeled dataset
- [ ] Run label validation suite
- [ ] Document correlation results