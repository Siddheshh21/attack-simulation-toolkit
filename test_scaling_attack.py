"""
Test script for scaling attack with optimized parameters
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.interactive_attack_tester import InteractiveAttackTester

def test_scaling_attack():
    """Test scaling attack with 2 attackers and 5 rounds"""
    print("\n" + "="*80)
    print("TESTING SCALING ATTACK WITH OPTIMIZED PARAMETERS")
    print("="*80 + "\n")
    
    tester = InteractiveAttackTester()
    
    # Configure scaling attack
    attack_type = "Scaling Attack"
    attacker_clients = [1, 2]  # 2 attackers
    num_rounds = 5
    
    print(f"Running {attack_type} with attackers {attacker_clients} for {num_rounds} rounds...")
    print("Expected outcomes with NEW OPTIMIZED PARAMETERS:")
    print("  - HEAVY metric degradation expected")
    print("  - Accuracy should drop significantly")
    print("  - F1 should drop by 50-70%")
    print("  - Precision should drop by 60-75%")
    print("  - Recall should drop significantly (label corruption applied)")
    print("  - AUC should drop by 10-20%")
    print("  - Detection should correctly flag attackers with TP=2, FP=0\n")
    print("New parameters:")
    print("  - scaling_factor: 5.0")
    print("  - agg_boost_rounds: 6")
    print("  - agg_learning_rate: 0.07")
    print("  - feature_noise_std: 0.05 (reduced DP noise)")
    print("  - drop_positive_fraction: 0.2")
    print("  - flip_labels_fraction: 0.1")
    print("  - detection_threshold: 0.33\n")
    
    # Execute attack with parameters
    attack_params = {'num_rounds': num_rounds}
    results = tester.execute_attack(
        attack_type=attack_type,
        attacker_clients=attacker_clients,
        attack_params=attack_params
    )
    
    print("\n" + "="*80)
    print("SCALING ATTACK TEST COMPLETED")
    print("="*80)
    
    return results

if __name__ == "__main__":
    test_scaling_attack()
