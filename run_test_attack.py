#!/usr/bin/env python3
"""
Automated test script for label flip attack with flip_percent=0.3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.interactive_attack_tester import InteractiveAttackTester

def run_label_flip_test():
    """Run label flip attack with flip_percent=0.3 and single attacker"""
    
    print("="*80)
    print("AUTOMATED LABEL FLIP TEST (flip_percent=0.3, single attacker)")
    print("="*80)
    
    tester = InteractiveAttackTester()
    
    # Configure attack parameters
    attack_params = {
        'flip_percent': 0.3,
        'flip_ratio': 0.3,
        'num_clients': 5
    }
    
    # Select single attacker (client 4)
    attacker_clients = [4]
    
    print(f"Attack type: Label Flip Attack")
    print(f"Attacker clients: {attacker_clients}")
    print(f"Flip percentage: {attack_params['flip_percent']}")
    
    # Execute the attack
    try:
        tester.execute_attack("Label Flip Attack", attacker_clients, attack_params)
        print("\n✅ Attack simulation completed successfully!")
    except Exception as e:
        print(f"\n❌ Attack simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_label_flip_test()
