"""
Visualization Module
=====================
Generates all charts and plots for the project.
Saves to results/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

# ── Color palette ──────────────────────────────
COLORS = {
    'CNN':  '#3b82f6',
    'RNN':  '#ef4444',
    'GRU':  '#22c55e',
    'LSTM': '#a855f7',
}
BG      = '#0f172a'
CARD    = '#1e293b'
TEXT    = '#f1f5f9'
ACCENT  = '#06b6d4'
GREEN   = '#22c55e'
RED     = '#ef4444'
YELLOW  = '#eab308'


def setup_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor':   CARD,
        'axes.edgecolor':   '#334155',
        'axes.labelcolor':  TEXT,
        'xtick.color':      '#94a3b8',
        'ytick.color':      '#94a3b8',
        'text.color':       TEXT,
        'grid.color':       '#1e293b',
        'grid.alpha':       0.8,
        'font.family':      'monospace',
    })


# ─────────────────────────────────────────────
#  1. Accuracy Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_accuracy_comparison(training_results, save=True):
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)

    models   = ['CNN', 'RNN', 'GRU', 'LSTM']
    no_dp    = [training_results.get(f'{m}_nodp', {}).get('accuracy', 0) for m in models]
    with_dp  = [training_results.get(f'{m}_dp',   {}).get('accuracy', 0) for m in models]

    x   = np.arange(len(models))
    w   = 0.35

    bars1 = ax.bar(x - w/2, no_dp,   w, label='Without DP', color='#475569', alpha=0.9, edgecolor='#64748b')
    bars2 = ax.bar(x + w/2, with_dp, w, label='With DP-SGD', color=[COLORS[m] for m in models], alpha=0.95, edgecolor='white', linewidth=0.5)

    # Value labels
    for bar, v in zip(bars1, no_dp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{v:.2%}', ha='center', va='bottom', fontsize=9, color='#94a3b8')
    for bar, v in zip(bars2, with_dp):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{v:.2%}', ha='center', va='bottom', fontsize=9, color=TEXT, fontweight='bold')

    # Star for best model
    best_idx = np.argmax(with_dp)
    ax.annotate('★ BEST', xy=(x[best_idx] + w/2, with_dp[best_idx] + 0.015),
                ha='center', fontsize=10, color=GREEN, fontweight='bold')

    ax.set_ylim(0.7, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_ylabel('Testing Accuracy', fontsize=11)
    ax.set_title('Model Accuracy Comparison: With vs Without Differential Privacy',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10, loc='lower left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.axhline(y=0.99, color=GREEN, linestyle='--', alpha=0.3, linewidth=1)
    ax.text(3.6, 0.991, '99%', color=GREEN, fontsize=8, alpha=0.6)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        path = 'results/accuracy_comparison.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {path}")
    return fig


# ─────────────────────────────────────────────
#  2. Training Loss/Accuracy Curves
# ─────────────────────────────────────────────

def plot_training_curves(training_results, save=True):
    setup_style()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor=BG)
    fig.suptitle('Training vs Validation Curves (Accuracy & Loss)',
                 fontsize=14, fontweight='bold', y=1.01)

    models = ['CNN', 'RNN', 'GRU', 'LSTM']

    for col, model in enumerate(models):
        for row, use_dp in enumerate([False, True]):
            ax   = axes[row][col]
            key  = f"{model}_{'dp' if use_dp else 'nodp'}"
            hist = training_results.get(key, {}).get('history', {})

            if hist:
                eps  = range(1, len(hist.get('accuracy', [])) + 1)
                color = COLORS[model]
                ax.plot(eps, hist.get('accuracy', []),     color=color,    linewidth=2, label='Train Acc')
                ax.plot(eps, hist.get('val_accuracy', []), color=color,    linewidth=2, linestyle='--', alpha=0.7, label='Val Acc')
                ax.plot(eps, hist.get('loss', []),         color=YELLOW,   linewidth=1.5, alpha=0.6, label='Loss')
                ax.plot(eps, hist.get('val_loss', []),     color=YELLOW,   linewidth=1.5, linestyle='--', alpha=0.4, label='Val Loss')

            tag = 'WITH DP' if use_dp else 'NO DP'
            ax.set_title(f'{model} | {tag}', fontsize=10, fontweight='bold', color=COLORS[model])
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylim(-0.05, 1.1)
            ax.legend(fontsize=6, loc='lower right')
            ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = 'results/training_curves.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {path}")
    return fig


# ─────────────────────────────────────────────
#  3. MIA Security Analysis
# ─────────────────────────────────────────────

def plot_mia_analysis(mia_results, save=True):
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)

    models = list(mia_results.keys())
    no_dp  = [mia_results[m]['no_dp']['mia_accuracy']   for m in models]
    with_dp= [mia_results[m]['with_dp']['mia_accuracy'] for m in models]

    x = np.arange(len(models))
    w = 0.35

    ax1.bar(x - w/2, no_dp,   w, label='Without DP', color=RED,   alpha=0.85)
    ax1.bar(x + w/2, with_dp, w, label='With DP-SGD', color=GREEN, alpha=0.85)
    ax1.axhline(y=0.5, color=ACCENT, linestyle='--', linewidth=2, label='Random Guess (50%)')

    for i, (nd, wd) in enumerate(zip(no_dp, with_dp)):
        ax1.text(x[i] - w/2, nd + 0.005, f'{nd:.1%}', ha='center', fontsize=9, color=RED, fontweight='bold')
        ax1.text(x[i] + w/2, wd + 0.005, f'{wd:.1%}', ha='center', fontsize=9, color=GREEN, fontweight='bold')

    ax1.set_ylim(0.4, 0.80)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax1.set_ylabel('MIA Attack Accuracy', fontsize=11)
    ax1.set_title('Membership Inference Attack Accuracy\n(lower = better privacy protection)', fontsize=11, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Risk reduction bar
    reductions = [mia_results[m]['with_dp']['risk_reduction'] for m in models]
    bars = ax2.bar(models, reductions, color=[COLORS[m] for m in models], alpha=0.9, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, reductions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{v:.1%}', ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel('MIA Risk Reduction', fontsize=11)
    ax2.set_title('Privacy Protection Gain with DP-SGD\n(MIA risk reduction)', fontsize=11, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        path = 'results/mia_analysis.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {path}")
    return fig


# ─────────────────────────────────────────────
#  4. Ablation Study
# ─────────────────────────────────────────────

def plot_ablation_study(ablation_results, save=True):
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)

    modalities = [r['modality'].replace('\n', ' ') for r in ablation_results]
    metrics    = ['accuracy', 'precision', 'recall', 'f1']
    colors     = ['#3b82f6', '#a855f7', '#f97316', '#22c55e']

    x = np.arange(len(modalities))
    w = 0.20

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] for r in ablation_results]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=metric.capitalize(),
                      color=color, alpha=0.88, edgecolor='white', linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{v:.1%}', ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax.set_ylim(0.88, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontsize=9, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Ablation Study: Impact of Multimodal Fusion (GRU with DP)',
                 fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        path = 'results/ablation_study.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {path}")
    return fig


# ─────────────────────────────────────────────
#  5. SHAP Feature Importance
# ─────────────────────────────────────────────

def plot_shap_importance(shap_data, save=True):
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

    top_n   = min(15, len(shap_data['top_features']))
    top_f   = shap_data['top_features'][:top_n]
    feats   = [f[0] for f in top_f]
    vals    = [f[1] for f in top_f]
    colors  = [COLORS['CNN'] if 'ECG' in f else
               COLORS['LSTM'] if 'BP' in f else COLORS['RNN']
               for f in feats]
    # Ensure lengths match
    assert len(feats) == len(vals) == len(colors) == top_n

    ax1.barh(list(range(top_n)), vals, color=colors, alpha=0.88, edgecolor='white', linewidth=0.4)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(feats, fontsize=8)
    ax1.set_xlabel('SHAP Importance', fontsize=10)
    ax1.set_title(f'Top {top_n} Features by SHAP Value\n(GRU with DP)', fontsize=11, fontweight='bold')

    patches = [
        mpatches.Patch(color=COLORS['CNN'],  label='ECG Features'),
        mpatches.Patch(color=COLORS['LSTM'], label='BP Features'),
        mpatches.Patch(color=COLORS['RNN'],  label='EEG Features'),
    ]
    ax1.legend(handles=patches, fontsize=8, loc='lower right')
    ax1.grid(axis='x', alpha=0.3)

    # Modality pie chart
    from src.explainability import get_modality_importance
    imp = get_modality_importance('GRU', use_dp=True)
    sizes  = list(imp.values())
    labels = list(imp.keys())
    wedge_colors = [COLORS['CNN'], COLORS['LSTM'], COLORS['RNN']]
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=wedge_colors, startangle=90,
        textprops={'fontsize': 11, 'color': TEXT},
        wedgeprops={'edgecolor': BG, 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(12)

    ax2.set_title('Modality Contribution to Classification\n(SHAP aggregation)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save:
        path = 'results/shap_importance.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {path}")
    return fig


# ─────────────────────────────────────────────
#  6. Sensitivity Analysis
# ─────────────────────────────────────────────

def plot_sensitivity_analysis(save=True):
    from src.differential_privacy import sensitivity_analysis
    setup_style()

    noise_mults, clip_thresholds, acc_data = sensitivity_analysis()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)

    colors = [COLORS['CNN'], COLORS['GRU'], COLORS['RNN']]
    for (clip, acc_vals), color in zip(acc_data.items(), colors):
        ax1.plot(noise_mults, acc_vals, 'o-', color=color, linewidth=2.5,
                 markersize=7, label=f'C={clip}', markerfacecolor='white')

    ax1.axvline(x=1.3, color=GREEN, linestyle='--', linewidth=2, alpha=0.8, label='Optimal σ=1.3')
    ax1.axhline(y=99.0, color=YELLOW, linestyle=':', alpha=0.5)
    ax1.fill_betweenx([98, 100], 0.8, 1.3, color=GREEN, alpha=0.05)
    ax1.set_xlabel('Noise Multiplier (σ)', fontsize=11)
    ax1.set_ylabel('GRU Accuracy (%)', fontsize=11)
    ax1.set_title('Accuracy vs Noise Multiplier\n(GRU model)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(93, 100.5)

    # Privacy budget vs epochs
    epsilons = []
    epochs   = range(1, 101)
    from src.differential_privacy import compute_epsilon, DPConfig
    config = DPConfig()
    for ep in epochs:
        eps = compute_epsilon(6000, config.BATCH_SIZE, config.NOISE_MULTIPLIER, ep, config.DELTA)
        epsilons.append(min(eps, 5.0))

    ax2.plot(list(epochs), epsilons, color=ACCENT, linewidth=2.5, label='ε (privacy budget)')
    ax2.axhline(y=1.0, color=GREEN, linestyle='--', linewidth=2, label='Target ε=1.0')
    ax2.axvline(x=100, color=YELLOW, linestyle=':', linewidth=1.5, alpha=0.7, label='Training end (ep=100)')
    ax2.fill_between(list(epochs), epsilons, 1.0, where=[e <= 1.0 for e in epsilons],
                     color=GREEN, alpha=0.1, label='Within budget')
    ax2.set_xlabel('Training Epochs', fontsize=11)
    ax2.set_ylabel('Privacy Budget (ε)', fontsize=11)
    ax2.set_title('Privacy Budget Consumption vs Epochs\n(Moments Accountant)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 2.5)

    plt.tight_layout()
    if save:
        path = 'results/sensitivity_analysis.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {path}")
    return fig


# ─────────────────────────────────────────────
#  7. Sample Signal Visualization
# ─────────────────────────────────────────────

def plot_signal_preprocessing(save=True):
    from src.data_preprocessing import generate_synthetic_heartbeat, apply_bandpass_filter
    setup_style()

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), facecolor=BG)
    fig.suptitle('Physiological Signal Preprocessing: Raw vs Filtered',
                 fontsize=13, fontweight='bold')

    fs = 250
    t  = np.linspace(0, 1, fs)
    signals_info = [('ECG', COLORS['GRU']), ('BP', COLORS['LSTM']), ('EEG', COLORS['RNN'])]

    ecg_r, bp_r, eeg_r = generate_synthetic_heartbeat(label=0)
    ecg_f = apply_bandpass_filter(ecg_r)
    bp_f  = apply_bandpass_filter(bp_r)
    eeg_f = apply_bandpass_filter(eeg_r)

    raws     = [ecg_r, bp_r, eeg_r]
    filtered = [ecg_f, bp_f, eeg_f]

    for row, (sig_name, color), raw, filt in zip(range(3), signals_info, raws, filtered):
        axes[row][0].plot(t, raw,  color='#64748b', linewidth=1.5, alpha=0.9, label='Raw Signal')
        axes[row][0].plot(t, filt, color=color,     linewidth=2,   alpha=0.9, label='Filtered')
        axes[row][0].set_title(f'{sig_name}: Raw vs Filtered', fontsize=10, fontweight='bold')
        axes[row][0].legend(fontsize=8)
        axes[row][0].grid(alpha=0.2)
        axes[row][0].set_xlabel('Time (s)', fontsize=8)

        # Abnormal version
        ecg_ab, bp_ab, eeg_ab = generate_synthetic_heartbeat(label=1)
        ab_signals = [ecg_ab, bp_ab, eeg_ab]
        ab_filt    = apply_bandpass_filter(ab_signals[row])

        axes[row][1].plot(t, ab_signals[row], color='#64748b', linewidth=1.5, alpha=0.9, label='Raw (Abnormal)')
        axes[row][1].plot(t, ab_filt,         color=RED,       linewidth=2,   alpha=0.9, label='Filtered')
        axes[row][1].set_title(f'{sig_name}: Abnormal Beat', fontsize=10, fontweight='bold', color=RED)
        axes[row][1].legend(fontsize=8)
        axes[row][1].grid(alpha=0.2)
        axes[row][1].set_xlabel('Time (s)', fontsize=8)

    plt.tight_layout()
    if save:
        path = 'results/signal_preprocessing.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {path}")
    return fig


# ─────────────────────────────────────────────
#  Generate All Plots
# ─────────────────────────────────────────────

def generate_all_plots(training_results, mia_results, ablation_results, shap_data):
    """Generate and save all project visualizations."""
    print("\n[VIZ] Generating all plots...")
    plot_accuracy_comparison(training_results)
    plot_training_curves(training_results)
    plot_mia_analysis(mia_results)
    plot_ablation_study(ablation_results)
    plot_shap_importance(shap_data)
    plot_sensitivity_analysis()
    plot_signal_preprocessing()
    print("[VIZ] All plots saved to results/")
    plt.close('all')


if __name__ == '__main__':
    from src.trainer import ModelTrainer
    from src.evaluator import run_mia_analysis, run_ablation_study
    from src.explainability import compute_shap_importance

    trainer = ModelTrainer(demo_mode=True)
    results = trainer.train_all()
    mia     = run_mia_analysis()
    ablation = run_ablation_study()
    shap    = compute_shap_importance('GRU', use_dp=True)
    generate_all_plots(results, mia, ablation, shap)
