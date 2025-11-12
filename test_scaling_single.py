"""
Single-Attacker Scaling Attack Test
Runs only the 1-attacker scenario to speed up iteration and isolate tuning.
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from interactive_attack_tester import InteractiveAttackTester


def test_scaling_attack_single(num_rounds=5):
    """Test scaling attack with a single attacker."""
    print("\n" + "="*80)
    print(f"TESTING SCALING ATTACK: 1 ATTACKER, {num_rounds} ROUNDS")
    print("="*80)

    tester = InteractiveAttackTester()
    attacker_clients = [1]

    print(f"\nAttacker clients: {attacker_clients}")
    print(f"Total clients: 5")
    print(f"Rounds: {num_rounds}")

    start_time = time.time()
    try:
        results = tester.execute_attack(
            attack_type="Scaling Attack",
            attacker_clients=attacker_clients,
            attack_params={
                'num_clients': 5,
                'num_rounds': num_rounds,
                'attacker_clients': attacker_clients
            }
        )

        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print("\n" + "="*80)
        print("TEST COMPLETED: 1 ATTACKER")
        print("="*80)
        print(f"Training Time: {minutes}m {seconds}s")
        print(f"Target: 2.5-3 minutes ({'✓ PASS' if 150 <= elapsed_time <= 180 else '✗ FAIL'})")

        if results and isinstance(results, dict):
            detection = results.get('detection_results', {})
            evaluation = results.get('evaluation_results', {})

            if detection:
                predicted = detection.get('predicted_attackers', [])
                confusion = detection.get('confusion', {})
                det_acc = detection.get('detection_accuracy', 0.0)
                print(f"\nDetection Results:")
                print(f"  Predicted Attackers: {predicted}")
                print(f"  Expected: {attacker_clients}")
                print(f"  TP: {confusion.get('TP', 0)}, FP: {confusion.get('FP', 0)}")
                print(f"  Detection Accuracy: {det_acc:.4f}")
                print(f"  Status: {'✓ PASS' if det_acc == 1.0 else '✗ FAIL'}")

            if evaluation:
                metrics = evaluation.get('comparison', {})
                if metrics:
                    delta = metrics.get('delta', {})
                    print(f"\nMetric Degradation:")
                    print(f"  Accuracy: {delta.get('accuracy', 0):.4f} (balanced)")
                    print(f"  F1: {delta.get('f1', 0):.4f}")
                    print(f"  Precision: {delta.get('precision', 0):.4f}")
                    print(f"  Recall: {delta.get('recall', 0):.4f}")

        return results, elapsed_time

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0


def main():
    print("\n" + "="*80)
    print("SINGLE-ATTACKER SCALING ATTACK TEST")
    print("="*80)
    print("\nThis test will:")
    print("1. Test scaling attack with 1 attacker")
    print("2. Verify detection accuracy is 1.0")
    print("3. Verify training completes in 2.5-3 minutes")

    print("\n" + "="*80)
    print("TEST: SINGLE ATTACKER")
    print("="*80)
    test_scaling_attack_single(num_rounds=5)

    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
