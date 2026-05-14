#!/usr/bin/env python3
"""
setup_and_run.py
=================
One-click setup script for the capstone project.
Run this first — it checks everything and launches the dashboard.

Usage:  python setup_and_run.py
"""

import os
import sys
import subprocess
import json

BANNER = """
╔══════════════════════════════════════════════════════════╗
║  CAPSTONE PROJECT — QUICK SETUP & LAUNCH               ║
║  Privacy-Preserving Heartbeat Classification           ║
║  NIET, Greater Noida — IT Department                   ║
╚══════════════════════════════════════════════════════════╝
"""

def check(label, fn):
    try:
        fn()
        print(f"  ✓  {label}")
        return True
    except Exception as e:
        print(f"  ✗  {label}: {e}")
        return False

def main():
    print(BANNER)
    print("Step 1: Checking environment...\n")

    checks = [
        ("Python 3.8+",   lambda: assert_python()),
        ("NumPy",         lambda: __import__('numpy')),
        ("SciPy",         lambda: __import__('scipy')),
        ("scikit-learn",  lambda: __import__('sklearn')),
        ("Matplotlib",    lambda: __import__('matplotlib')),
        ("Flask",         lambda: __import__('flask')),
    ]

    ok = all(check(lbl, fn) for lbl, fn in checks)

    # Optional
    print()
    opt_checks = [
        ("TensorFlow (optional)", lambda: __import__('tensorflow')),
        ("tensorflow-privacy",    lambda: __import__('tensorflow_privacy')),
        ("imbalanced-learn",      lambda: __import__('imblearn')),
        ("SHAP",                  lambda: __import__('shap')),
        ("WFDB",                  lambda: __import__('wfdb')),
    ]
    for lbl, fn in opt_checks:
        check(lbl, fn)

    print()
    print("Step 2: Running pipeline (demo mode)...")
    os.system(f"{sys.executable} main.py --mode demo")

    print()
    print("Step 3: Verifying outputs...")
    expected = [
        'results/full_results.json',
        'results/accuracy_comparison.png',
        'results/training_curves.png',
        'results/mia_analysis.png',
        'results/ablation_study.png',
        'results/shap_importance.png',
        'results/sensitivity_analysis.png',
        'results/signal_preprocessing.png',
        'results/confusion_matrices.png',
        'results/roc_curves.png',
        'results/system_architecture.png',
    ]
    all_found = True
    for f in expected:
        exists = os.path.exists(f)
        print(f"  {'✓' if exists else '✗'}  {f}")
        if not exists:
            all_found = False

    print()
    if all_found:
        print("✅ All outputs generated successfully!")
        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  🚀 LAUNCHING DASHBOARD                                ║")
        print("║  Open your browser at: http://localhost:5000           ║")
        print("║  Press Ctrl+C to stop                                  ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()
        os.system(f"{sys.executable} dashboard.py")
    else:
        print("⚠ Some outputs missing. Check errors above.")
        print("Try: python main.py --mode demo")


def assert_python():
    assert sys.version_info >= (3, 8), f"Need Python 3.8+, got {sys.version}"


if __name__ == '__main__':
    main()
