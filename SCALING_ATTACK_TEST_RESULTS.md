# Scaling Attack Test Results - Final Session

## Test Configuration
- **Date**: November 11, 2025
- **Honest Rounds**: 10
- **Detection Threshold**: 0.33
- **Clean Baseline**: Acc=0.8370, Prec=0.6129, Recall=0.6898, F1=0.6491, AUC=0.9325

## Single Attacker Results (Latest Run)
**Training Time**: 5m 10s ❌ (Target: 2.5-3 minutes)

**Metric Drops** (from partial output):
- **Accuracy**: -15.95% ❌ (Target: -4% to -9%, **TOO HIGH**)
- **Precision**: -17.35% ⚠️ (Target: -10% to -25%, **CLOSE BUT HIGH**)
- **Recall**: (Not captured in partial output)
- **F1**: (Not captured in partial output)
- **AUC**: -6.10% ❌ (Target: -2% to -5%, **TOO HIGH**)

**Detection**: ✓ PASS (TP=1, FP=0)

**Parameters Used**:
- Scaling: 1.12
- Agg boost: 1
- Noise: 0.003
- Drop: 0.012
- Flip: 0.003
- Inject FP: 0.008
- Logit shift: 0.12
- Rounds: 5

## Two Attacker Results
**Training Time**: 3m 34s ⚠️ (Target: 2.5-3 minutes, slightly over)

**Metric Drops**:
- **Accuracy**: -6.87% ❌ (Target: -13% to -30%, **TOO LOW!**)
- **Precision**: -30.21% ✓ (Target: -25% to -30%, **PERFECT!**)
- **Recall**: -14.84% ⚠️ (Target: -20% to -35%, **TOO LOW!**)
- **F1**: -23.74% ⚠️ (Target: -25% to -40%, **CLOSE BUT LOW**)
- **AUC**: -4.55% ✓ (Target: -4% to -10%, **PERFECT!**)

**Detection**: ✓ PASS (TP=2, FP=0)

**Risk Scores**: 0.5918, 0.5918 ❌ (**STILL IDENTICAL!**)

**Parameters Used**:
- Scaling: 9.0
- Agg boost: 10
- Learning rate: 0.32
- Eval weight: 25.0
- Agg risk: 4.0
- Noise: 0.24
- Drop: 0.48
- Flip: 0.38
- Poison: 0.52
- Inject FP: 0.55
- Logit shift: 2.5
- Eval beta: 0.08
- Rounds: 26

## Issues Identified

### Single Attacker
1. **Accuracy drop still too high** (-15.95% vs target -4% to -9%)
2. **Training time too slow** (5m 10s vs target 2.5-3min)
3. **AUC drop too high** (-6.10% vs target -2% to -5%)

### Two Attackers
1. **Accuracy drop too low** (-6.87% vs target -13% to -30%)
2. **Recall drop too low** (-14.84% vs target -20% to -35%)
3. **F1 drop too low** (-23.74% vs target -25% to -40%)
4. **Risk scores identical** (0.5918 for both attackers)
5. **Training time slightly over target** (3m 34s vs 2.5-3min)

## Root Causes

### Single Attacker
The attack is still too strong despite minimal parameters. The issue appears to be:
- **Aggregation boost** even at 1 round is amplifying the attack
- **Scaling factor 1.12** may still be too high
- **Training time** is slow due to honest rounds (10) needed for strong baseline

### Two Attackers
The attack is not strong enough despite extreme parameters. The issues are:
- **Accuracy/Recall/F1** not dropping enough - need even more aggressive parameters
- **Identical risk scores** - detection.py needs more variability in risk calculation
- **Training time** slightly over due to 26 attacker rounds and 10 aggregation rounds

## Next Steps Required

1. **Single Attacker**: Further reduce parameters or disable more mechanisms
2. **Two Attackers**: Increase to MAXIMUM possible parameters
3. **Risk Scores**: Add randomness or client-specific jitter in detection.py
4. **Training Speed**: Reduce honest rounds to 8 or reduce attacker rounds

## Status
❌ **NOT COMPLETE** - Metrics not in target bands, training time over target, risk scores identical
