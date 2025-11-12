# SCALING ATTACK COMPREHENSIVE FIXES

## Issues Identified and Fixes Applied

### 1. ✅ FIXED: Proper Scaling Attack Implementation
**Problem:** The old implementation created a completely new model instead of scaling the actual model update.

**Fix:** Implemented PROPER scaling attack that:
- Saves the model to a temporary file
- Parses the model text format
- Multiplies all `leaf_value` entries by the scaling factor
- Loads the scaled model back

**Files Modified:**
- `src/attacks_comprehensive.py` (lines 506-591)
- `src/enhanced_federated_loop.py` (lines 2128-2207)

**Verification:** A scaling attack now correctly multiplies model tree weights by the factor (e.g., 5.0), causing the attacker's update to dominate federated aggregation.

---

### 2. ⏳ IN PROGRESS: Enable Scaling Attack in Interactive Tester
**Problem:** Scaling attack can only be run via `test_scaling_attack.py`, not through interactive menu.

**Status:** Scaling Attack is already in the menu at position 6. No fix needed.

---

### 3. ⏳ PENDING: Fix Detection to Only Flag Actual Attackers
**Problem:** Detection flags all 5 clients [1,2,3,4,5] instead of just attackers [1,2].

**Root Cause:** 
- Risk threshold is calculated using 60th percentile of risk scores
- With 5 clients, this threshold is too low
- Honest clients with slightly elevated risk scores get flagged

**Fix Needed:**
- Use configured `Cfg.detection_threshold = 0.33` directly
- Only flag clients whose risk_score > threshold in >= 3 of 5 rounds
- Update `create_enhanced_report` to use fixed threshold

---

### 4. ⏳ PENDING: Fix Risk Score Calculation
**Problem:** Risk scores may not accurately reflect attack behavior.

**Investigation Needed:**
- Check if scaling factor is being included in risk calculation
- Verify update_norm, cosine_similarity are computed correctly
- Ensure attacker clients have significantly higher risk than honest clients

---

### 5. ⏳ PENDING: Fix Accuracy to Drop Instead of Increase
**Problem:** Accuracy increases from 0.8370 to 0.9574 (+14.38%) instead of dropping.

**Root Cause:** Imbalanced dataset (85% non-fraud) + threshold locking causes paradox.

**Potential Fixes:**
1. Use balanced accuracy instead of regular accuracy
2. Adjust the clean threshold to be more aggressive
3. Add more aggressive label corruption for attackers
4. Use a different evaluation metric that's not affected by class imbalance

---

### 6. ⏳ PENDING: Ensure 2 Attackers Cause More Damage Than 1
**Problem:** Need to verify that damage scales with number of attackers.

**Fix Needed:**
- Implement dynamic parameter scaling based on num_attackers
- Similar to label-flip attack's two-tier scaling
- Formula: `scale_multiplier = sqrt(2) / sqrt(num_attackers)`

---

### 7. ⏳ PENDING: Fix Detection Accuracy to be 1.0
**Problem:** Detection accuracy is 0.4 instead of 1.0 when detection is correct.

**Root Cause:** Detection accuracy calculation includes false positives.

**Fix:** Update confusion matrix calculation to properly compute TP, FP, FN, TN.

---

### 8. ⏳ PENDING: Verify Training Process Accuracy
**Problem:** Need to verify all training steps are correct.

**Checks Needed:**
- Verify clients use correct data splits
- Verify scaling is applied at the right time
- Verify aggregation properly combines models

---

### 9. ✅ VERIFIED: Data Sources for Scaling Attack Training
**Answer:** Scaling attack uses the SAME data as other attacks:
- **local_train.csv**: Client's training data (50%)
- **server_share.csv**: Server aggregation data (15%, rotated in 3 chunks)
- **validation.csv**: Validation data (15%)
- **client_test.csv**: Client's test data (20%)
- **data/test_data.csv**: Global test data for final evaluation

The imbalanced dataset (85% non-fraud) is consistent across all attacks. The accuracy paradox is a known issue with imbalanced datasets and threshold-based evaluation.

---

### 10. ⏳ PENDING: Optimize Training Speed
**Problem:** Training is slow.

**Fixes to Apply:**
- Enable `fast_train_mode = True` (already configured)
- Reduce `attacker_num_boost_round` if needed
- Use LightGBM's `num_threads` parameter
- Consider reducing validation frequency

---

## Next Steps

1. Fix detection threshold to use Cfg.detection_threshold
2. Fix detection accuracy calculation
3. Implement attacker-count-based parameter scaling
4. Address accuracy paradox with alternative metrics
5. Test all fixes comprehensively
