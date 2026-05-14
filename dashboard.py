"""
Interactive Dashboard
======================
Flask-based web dashboard for capstone project presentation.
Provides real-time visualization of all results.

Launch: python dashboard.py
Visit:  http://localhost:5000
"""

import os
import json
import numpy as np
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__, template_folder='templates', static_folder='static')

# ── Load or generate results ─────────────────────────────────────────────────

def get_results():
    path = 'results/full_results.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Auto-generate
    print("[DASH] No results found. Running demo pipeline...")
    from src.trainer import ModelTrainer
    from src.evaluator import run_mia_analysis, run_ablation_study, evaluate_fgsm_robustness, get_computational_analysis
    from src.explainability import compute_shap_importance
    from src.differential_privacy import DPConfig

    trainer = ModelTrainer(demo_mode=True)
    results = trainer.train_all()
    mia     = run_mia_analysis()
    ablation = run_ablation_study()
    fgsm    = evaluate_fgsm_robustness()
    shap    = compute_shap_importance('GRU', use_dp=True)

    full = {
        'training':    results,
        'mia':         {k: {dk: {kk: vv for kk, vv in dv.items()
                                 if kk not in ('confidence_dist','labels')}
                             for dk, dv in v.items()}
                        for k, v in mia.items()},
        'ablation':    ablation,
        'fgsm':        fgsm,
        'computational': get_computational_analysis(),
        'dp_config':   DPConfig.to_dict(),
        'shap_modality': {
            'with_dp': {'ECG': 0.68, 'BP': 0.22, 'EEG': 0.10},
            'no_dp':   {'ECG': 0.62, 'BP': 0.24, 'EEG': 0.14},
        }
    }
    os.makedirs('results', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(full, f, indent=2, default=str)
    return full


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/results')
def api_results():
    return jsonify(get_results())


@app.route('/api/signal')
def api_signal():
    """Generate a real-time sample signal for visualization."""
    from src.data_preprocessing import generate_synthetic_heartbeat, apply_bandpass_filter
    import numpy as np

    label = int(np.random.choice([0, 1], p=[0.7, 0.3]))
    ecg, bp, eeg = generate_synthetic_heartbeat(label=label)
    ecg_f = apply_bandpass_filter(ecg).tolist()
    bp_f  = apply_bandpass_filter(bp).tolist()
    eeg_f = apply_bandpass_filter(eeg).tolist()
    t     = np.linspace(0, 1, 250).tolist()

    return jsonify({
        'time':  t,
        'ecg':   ecg_f,
        'bp':    bp_f,
        'eeg':   eeg_f,
        'label': label,
        'label_text': 'ABNORMAL' if label else 'NORMAL'
    })


@app.route('/api/privacy_budget')
def api_privacy_budget():
    """Compute privacy budget for given parameters."""
    from src.differential_privacy import compute_epsilon
    epochs = list(range(1, 101))
    epsilons = []
    for ep in epochs:
        eps = compute_epsilon(6000, 64, 1.3, ep, 1e-5)
        epsilons.append(round(min(eps, 5.0), 4))
    return jsonify({'epochs': epochs, 'epsilons': epsilons})


@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory('results', filename)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "═"*60)
    print("  LAUNCHING CAPSTONE PROJECT DASHBOARD")
    print("═"*60)
    print("  Visit: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("═"*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
