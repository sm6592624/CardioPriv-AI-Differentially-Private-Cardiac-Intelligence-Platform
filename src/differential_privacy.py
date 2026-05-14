"""
Differential Privacy Framework
================================
Implements DP-SGD (Differentially Private Stochastic Gradient Descent)
for training deep learning models with formal privacy guarantees.

Mathematical Framework:
  P(M(D) ∈ S) ≤ e^ε · P(M(D') ∈ S) + δ

DP-SGD Steps:
  1. Gradient Clipping: ḡᵢ = gᵢ / max(1, ‖gᵢ‖₂/C)
  2. Noise Injection: g̃ = Σḡᵢ + N(0, σ²C²I)
  3. Privacy Accounting via Moments Accountant

References:
  - Abadi et al. (2016) Deep Learning with Differential Privacy
  - ElKomy et al. (2025) Journal of Big Data 12:216
"""

import numpy as np
import math


# ─────────────────────────────────────────────
#  DP-SGD Parameters (Paper-Exact Values)
# ─────────────────────────────────────────────

class DPConfig:
    """Differential Privacy configuration matching the paper's exact parameters."""

    # Core DP-SGD parameters (Table 1 from ElKomy et al. 2025)
    L2_NORM_CLIP      = 1.5    # C: gradient clipping threshold
    NOISE_MULTIPLIER  = 1.3    # σ: scale of Gaussian noise
    MICRO_BATCH_SIZE  = 64     # number of micro-batches per subset
    DELTA             = 1e-5   # δ: probability of privacy failure

    # Derived epsilon (tracked via Moments Accountant)
    TARGET_EPSILON    = 1.0    # ε: target privacy budget

    # Training config
    LEARNING_RATE     = 1e-4
    EPOCHS            = 100
    BATCH_SIZE        = 64

    @classmethod
    def to_dict(cls):
        return {
            'l2_norm_clip':     cls.L2_NORM_CLIP,
            'noise_multiplier': cls.NOISE_MULTIPLIER,
            'micro_batch_size': cls.MICRO_BATCH_SIZE,
            'delta':            cls.DELTA,
            'epsilon':          cls.TARGET_EPSILON,
            'learning_rate':    cls.LEARNING_RATE,
        }


# ─────────────────────────────────────────────
#  Privacy Accounting (Moments Accountant)
# ─────────────────────────────────────────────

def compute_epsilon(n_samples, batch_size, noise_multiplier, epochs, delta=1e-5):
    """
    Compute privacy budget ε using Moments Accountant approximation.
    Based on Abadi et al. (2016) implementation.

    Args:
        n_samples:        Total number of training samples
        batch_size:       Mini-batch size
        noise_multiplier: σ (Gaussian noise scale)
        epochs:           Number of training epochs
        delta:            δ (privacy failure probability)

    Returns:
        epsilon: Privacy budget consumed
    """
    q = batch_size / n_samples          # Sampling ratio
    steps = epochs * (n_samples // batch_size)  # Total SGD steps

    # Renyi Differential Privacy bound (simplified RDP accountant)
    # This is an approximation of the tight Moments Accountant
    orders = list(range(2, 64)) + [128, 256, 512]
    rdp_epsilons = []

    for alpha in orders:
        # RDP for Gaussian mechanism
        rdp = _compute_rdp_gaussian(q, noise_multiplier, steps, alpha)
        rdp_epsilons.append(rdp)

    # Convert RDP to (ε, δ)-DP
    epsilon = _rdp_to_dp(rdp_epsilons, orders, delta)
    return epsilon


def _compute_rdp_gaussian(q, sigma, steps, alpha):
    """Compute RDP for subsampled Gaussian mechanism."""
    if q == 0:
        return 0.0
    if sigma == 0:
        return float("inf")
    rdp_per_step = alpha / (2 * sigma ** 2)
    if q < 1:
        exponent = rdp_per_step * alpha
        if exponent > 500:
            rdp_sampled = rdp_per_step
        else:
            try:
                rdp_sampled = min(rdp_per_step,
                    math.log(1 + q**2 * (math.exp(exponent) - 1)/alpha + 1e-8)/alpha)
            except (OverflowError, ValueError):
                rdp_sampled = rdp_per_step
    else:
        rdp_sampled = rdp_per_step
    return steps * rdp_sampled


def _rdp_to_dp(rdp_list, orders, delta):
    """Convert RDP to (ε, δ)-DP."""
    eps_list = []
    for rdp, alpha in zip(rdp_list, orders):
        if rdp == float('inf'):
            continue
        if delta > 0:
            eps = rdp + math.log(1 / delta) / (alpha - 1)
        else:
            eps = float('inf')
        eps_list.append(eps)

    return min(eps_list) if eps_list else float('inf')


# ─────────────────────────────────────────────
#  Gradient Clipping & Noise (NumPy)
# ─────────────────────────────────────────────

def clip_gradient(gradient, l2_norm_clip):
    """Clip gradient L2 norm to bound sensitivity."""
    norm = np.linalg.norm(gradient)
    return gradient / max(1.0, norm / l2_norm_clip)


def add_gaussian_noise(gradient, l2_norm_clip, noise_multiplier):
    """Add calibrated Gaussian noise for differential privacy."""
    noise_std = noise_multiplier * l2_norm_clip
    noise = np.random.normal(0, noise_std, gradient.shape)
    return gradient + noise


def dp_sgd_step(gradients, l2_norm_clip, noise_multiplier):
    """
    Full DP-SGD update step:
    1. Clip per-sample gradients
    2. Aggregate
    3. Add Gaussian noise
    """
    clipped = [clip_gradient(g, l2_norm_clip) for g in gradients]
    aggregated = np.mean(clipped, axis=0)
    noisy = add_gaussian_noise(aggregated, l2_norm_clip, noise_multiplier)
    return noisy


# ─────────────────────────────────────────────
#  TensorFlow Privacy Integration
# ─────────────────────────────────────────────

def get_dp_optimizer(optimizer_class='adam', config: DPConfig = None):
    """
    Returns a DP-SGD optimizer using tensorflow-privacy.
    Falls back to standard optimizer if tensorflow-privacy not available.
    """
    if config is None:
        config = DPConfig()

    try:
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
            make_keras_optimizer_class
        )
        DPAdam = make_keras_optimizer_class(
            __import__('tensorflow').keras.optimizers.legacy.Adam
        )
        optimizer = DPAdam(
            l2_norm_clip=config.L2_NORM_CLIP,
            noise_multiplier=config.NOISE_MULTIPLIER,
            num_microbatches=config.MICRO_BATCH_SIZE,
            learning_rate=config.LEARNING_RATE
        )
        print(f"[DP] Using tensorflow-privacy DP-SGD optimizer")
        print(f"     L2 Clip={config.L2_NORM_CLIP}, σ={config.NOISE_MULTIPLIER}, δ={config.DELTA}")
        return optimizer, True

    except Exception as e:
        print(f"[DP] tensorflow-privacy not available ({e}). Using standard Adam.")
        try:
            import tensorflow as tf
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        except:
            optimizer = None
        return optimizer, False


# ─────────────────────────────────────────────
#  Privacy Budget Reporter
# ─────────────────────────────────────────────

class PrivacyAccountant:
    """Tracks and reports privacy budget consumption."""

    def __init__(self, n_samples, config: DPConfig = None):
        self.config = config or DPConfig()
        self.n_samples = n_samples
        self.epoch = 0
        self.history = []

    def step(self):
        """Record one epoch of training and compute current ε."""
        self.epoch += 1
        eps = compute_epsilon(
            n_samples=self.n_samples,
            batch_size=self.config.BATCH_SIZE,
            noise_multiplier=self.config.NOISE_MULTIPLIER,
            epochs=self.epoch,
            delta=self.config.DELTA
        )
        self.history.append({'epoch': self.epoch, 'epsilon': eps})
        return eps

    def report(self):
        """Print privacy budget summary."""
        if self.history:
            eps = self.history[-1]['epsilon']
            print(f"\n{'='*50}")
            print(f"  Privacy Budget Report")
            print(f"{'='*50}")
            print(f"  Epochs trained:   {self.epoch}")
            print(f"  ε (epsilon):      {eps:.4f}")
            print(f"  δ (delta):        {self.config.DELTA:.1e}")
            print(f"  Noise multiplier: {self.config.NOISE_MULTIPLIER}")
            print(f"  L2 norm clip:     {self.config.L2_NORM_CLIP}")
            print(f"  Privacy budget:   {'✓ WITHIN BUDGET' if eps <= self.config.TARGET_EPSILON else '⚠ EXCEEDED'}")
            print(f"{'='*50}\n")
        return self.history


# ─────────────────────────────────────────────
#  Sensitivity Analysis
# ─────────────────────────────────────────────

def sensitivity_analysis():
    """
    Compute accuracy vs noise_multiplier and clipping_threshold
    based on paper's Figure 1 results.
    Returns data for visualization.
    """
    noise_multipliers = [0.8, 1.0, 1.3, 1.5, 1.8, 2.0]
    clipping_thresholds = [1.0, 1.5, 2.0]

    # Results from paper (GRU model, Table VI equivalent)
    # Key finding: GRU stable up to σ=1.3, drops after
    accuracy_vs_noise = {
        1.0: [99.8, 99.7, 99.5, 99.1, 98.2, 97.0],   # C=1.0
        1.5: [99.9, 99.8, 99.5, 99.0, 97.8, 96.5],   # C=1.5 (optimal)
        2.0: [99.7, 99.5, 99.0, 98.2, 96.5, 94.0],   # C=2.0
    }

    return noise_multipliers, clipping_thresholds, accuracy_vs_noise


if __name__ == '__main__':
    # Example: compute epsilon for paper's configuration
    config = DPConfig()
    eps = compute_epsilon(
        n_samples=6000,
        batch_size=config.BATCH_SIZE,
        noise_multiplier=config.NOISE_MULTIPLIER,
        epochs=config.EPOCHS,
        delta=config.DELTA
    )
    print(f"Privacy budget: ε={eps:.4f}, δ={config.DELTA:.1e}")
    print(f"Config: {config.to_dict()}")

    # Test privacy accountant
    accountant = PrivacyAccountant(n_samples=6000)
    for ep in range(1, 11):
        eps = accountant.step()
        print(f"  Epoch {ep:3d}: ε = {eps:.4f}")
    accountant.report()
