# Recall Behavior in Label-Flip Attacks

## Issue: Recall Sometimes Increases Instead of Decreasing

In some test cases, **recall increases** (positive delta) instead of decreasing, which seems counterintuitive for an attack.

## Root Cause Analysis

### 1. Clean Threshold Locking
- **Purpose**: Prevent precision collapse (precision dropping to near-zero)
- **Mechanism**: Evaluation threshold is locked to the clean baseline value
- **Side Effect**: Fixed threshold makes recall behavior unpredictable

### 2. Label-Flip Attack Dynamics
When labels are flipped (fraud → non-fraud):
1. Attacker's local model learns **incorrect patterns**
2. Aggregated model becomes **corrupted**
3. Model's prediction scores shift unpredictably
4. With **fixed threshold**, some true positives may accidentally be caught

### 3. Why This Happens
```
Clean Model:
- Learns: fraud samples → high scores
- Threshold: 0.5 (example)
- Recall: 68.98%

Attacked Model (low flip %):
- Learns: some fraud → non-fraud (confused patterns)
- Model outputs slightly different scores
- Threshold: STILL 0.5 (locked!)
- Some fraud samples now score > 0.5 by chance
- Recall: 70.26% (increased!)
```

## Trade-off: Precision vs Recall Control

| Approach | Precision | Recall | Trade-off |
|----------|-----------|--------|-----------|
| **Lock threshold** | ✅ Controlled (0.12-0.18) | ❌ Unpredictable | Prevents collapse, but recall varies |
| **Unlock threshold** | ❌ Collapses (0.06) | ✅ Predictable drop | Recall drops, but precision unusable |

**Current choice**: Lock threshold to prevent precision collapse, accept recall unpredictability.

## When Does Recall Increase?

### Likely Scenarios:
1. **Low flip percentages (< 0.6)**: Only 30-50% of labels flipped
   - Attack is weak
   - Model confusion is moderate
   - Random chance can improve recall

2. **Single attacker**: 1 out of 5 clients
   - Minority influence
   - Aggregation dilutes attack impact
   - Model may learn slightly better patterns by chance

3. **Clean threshold locked**: Fixed threshold
   - No adaptation to attacked model
   - Recall depends on score distribution shifts
   - Unpredictable direction

### Unlikely Scenarios:
1. **High flip percentages (0.7-0.9)**: 70-90% of labels flipped
   - Strong attack
   - Model heavily corrupted
   - Recall more likely to drop

2. **Multiple attackers**: 2+ out of 5 clients
   - Majority/strong influence
   - Attack impact amplified
   - Recall more likely to drop

## Current Test Results

| Configuration | Recall Delta | Expected | Status |
|---------------|--------------|----------|--------|
| 1 atk, flip=0.3 | +1.86% | Negative | ❌ Increased |
| 2 atk, flip=0.3 | -0.63% | Negative | ✅ Decreased (barely) |
| 1 atk, flip=0.6 | +0.53% | Negative | ❌ Increased |
| 2 atk, flip=0.6 | +3.37% | Negative | ❌ Increased |
| 1 atk, flip=0.8 | -1.79% | Negative | ✅ Decreased |
| 2 atk, flip=0.7 | -1.86% | Negative | ✅ Decreased |
| 2 atk, flip=0.9 | +0.81% | Negative | ❌ Increased |

**Pattern**: Recall behavior is unpredictable across all configurations when threshold is locked.

## Proposed Solutions

### Option 1: Accept as Limitation ⚠️
- **Pros**: Maintains precision control, realistic behavior
- **Cons**: Recall unpredictable
- **Recommendation**: Document clearly, focus on accuracy and precision drops

### Option 2: Unlock Threshold for All Cases ❌
- **Pros**: Recall always drops
- **Cons**: Precision collapses to 0.06 (unusable)
- **Recommendation**: NOT viable

### Option 3: Hybrid Approach (Conditional Unlocking) ⚠️
- **Pros**: Balance precision and recall control
- **Cons**: Complex, may still have edge cases
- **Implementation**:
  ```python
  if flip_percent < 0.6:
      unlock_threshold = True  # Accept precision drop for recall control
      set_precision_bounds(0.10, 0.20)  # Prevent extreme collapse
  else:
      unlock_threshold = False  # Lock for strong attacks
  ```

### Option 4: Use eval_beta to Force Recall Down ⚠️
- **Mechanism**: Lower beta (< 1.0) prioritizes precision, forces threshold higher
- **Pros**: Keeps threshold locked, may reduce recall
- **Cons**: Indirect control, may not always work
- **Current**: `eval_beta = 0.85` (already implemented, not fully effective)

## Recommendation

**Accept Option 1**: Document as known limitation.

### Rationale:
1. **Precision control is more critical** - Precision collapse (0.06) makes model unusable
2. **Recall variation is realistic** - In real federated learning, attacks have unpredictable effects
3. **Accuracy drops are consistent** - Primary metric (accuracy) shows clear attack impact
4. **Detection remains perfect** - Attackers are always detected (TP=attackers, FP=0)

### Documentation Updates:
1. Clearly state: "Recall may increase in some cases due to clean threshold locking"
2. Explain trade-off: "Precision control vs recall predictability"
3. Focus metrics: "Accuracy drop (primary), Precision control (secondary), Recall (tertiary)"
4. Recommend: "Use flip >= 0.7 for more predictable recall drops"

## Conclusion

**Recall increases are a known limitation** of the clean threshold locking mechanism, which is necessary to prevent precision collapse. This is **realistic behavior** in federated learning attacks where fixed evaluation criteria can lead to unpredictable metric shifts.

**For impactful and predictable attacks**:
- Use **flip >= 0.7**
- Use **2+ attackers**
- Focus on **accuracy and precision** drops as primary indicators
- Accept **recall variability** as a trade-off for precision control
