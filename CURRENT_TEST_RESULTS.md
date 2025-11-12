# Current Test Results - Iteration Summary

## Latest Test Results (After All Optimizations)

### Two Attackers (6m 7s - ❌ Still 2x too slow)
- **Accuracy**: -7.42% ❌ (target: -13% to -30%, STILL TOO LIGHT)
- **Precision**: -21.71% ⚠️ (target: -25% to -30%, CLOSE!)
- **Recall**: -17.05% ❌ (target: -20% to -35%, TOO LIGHT)
- **F1**: -19.58% ❌ (target: -25% to -40%, TOO LIGHT)
- **AUC**: -4.06% ✓ (target: -4% to -10%, IN RANGE!)
- **Detection**: 1.0000 ✓ (Perfect!)

### Single Attacker (6m 3s - ❌ Still 2x too slow)
- **Results truncated in output** - Need to verify metrics

## Current Parameters

### Single Attacker
```python
base_scaling_factor = 2.2
base_noise = 0.016
base_drop = 0.06
base_flip = 0.018
base_rounds = 14
agg_boost_rounds = 2
agg_learning_rate = 0.020
attacker_eval_weight = 1.4
poison_server_share_fraction = 0.008
honest_num_boost_round = 15
```

### Two Attackers
```python
base_scaling_factor = 9.0
base_noise = 0.20
base_drop = 0.40
base_flip = 0.32
base_rounds = 25 (actual: 26)
agg_boost_rounds = 8
agg_learning_rate = 0.20
attacker_eval_weight = 14.0
agg_risk_gain = 2.5
poison_server_share_fraction = 0.32
inject_false_positive_fraction = 0.38
eval_logit_shift = 1.45
honest_num_boost_round = 15
```

## Issues Identified

### 1. Training Time (CRITICAL)
- **Current**: 6+ minutes for both scenarios
- **Target**: 2.5-3 minutes
- **Gap**: ~3 minutes too slow (2x slower than target)
- **Root Cause**: `honest_num_boost_round = 15` is still too high
- **Solution**: Reduce to 10-12 rounds

### 2. Two-Attacker Metrics (CRITICAL)
All metrics except AUC are still too light:
- Accuracy needs -6% more drop (currently -7.42%, need -13% minimum)
- Precision needs -3% more drop (currently -21.71%, need -25% minimum)
- Recall needs -3% more drop (currently -17.05%, need -20% minimum)
- F1 needs -5% more drop (currently -19.58%, need -25% minimum)

**Root Cause**: Despite maximum parameters, the attack is not strong enough.

**Possible Solutions**:
1. Increase `eval_logit_shift` beyond 1.45 (try 1.6-1.8)
2. Increase `inject_false_positive_fraction` beyond 0.38 (try 0.42-0.45)
3. Increase `base_drop` beyond 0.40 (try 0.45-0.48)
4. Increase `base_flip` beyond 0.32 (try 0.35-0.38)

### 3. Single-Attacker Metrics (UNKNOWN)
Results were truncated - need to verify if metrics are in target range:
- Target: Acc -4% to -9%, Prec -10% to -25%, Recall -10% to -20%, F1 -12% to -25%, AUC -2% to -5%

## Recommended Next Steps

### Priority 1: Reduce Training Time
```python
# Change in interactive_attack_tester.py
tuned['honest_num_boost_round'] = 10  # Reduce from 15 to 10
```
**Expected Impact**: Reduce training time from 6min to ~3-4min

### Priority 2: Strengthen Two-Attacker Impact
```python
# Increase these parameters:
params['eval_logit_shift'] = 1.65  # Increase from 1.45
params['inject_false_positive_fraction'] = 0.42  # Increase from 0.38
params['base_drop'] = 0.45  # Increase from 0.40
params['base_flip'] = 0.35  # Increase from 0.32
```
**Expected Impact**: Push all metrics into target bands

### Priority 3: Verify Single-Attacker
Run isolated single-attacker test to verify metrics are in range.

## Progress Summary

### ✅ Completed
1. Implemented eval_logit_shift mechanism for precision drop
2. Implemented inject_false_positive_fraction for training-time corruption
3. Drastically softened single-attacker parameters
4. Maximized two-attacker parameters
5. Perfect detection accuracy (1.0000)
6. AUC drop is in target range for two attackers

### ⚠️ In Progress
1. Training time reduction (6min → 2.5-3min)
2. Two-attacker metric drops (need heavier impact)
3. Single-attacker verification (results truncated)

### ❌ Not Yet Achieved
1. 2.5-3 minute training time target
2. Two-attacker accuracy drop in -13% to -30% range
3. Two-attacker precision drop in -25% to -30% range
4. Two-attacker recall drop in -20% to -35% range
5. Two-attacker F1 drop in -25% to -40% range
