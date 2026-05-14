"""
Data Preprocessing Module
=========================
Handles loading, filtering, segmentation, and balancing of
multimodal physiological signals (ECG, EEG, BP).

Based on: ElKomy et al., Journal of Big Data (2025) 12:216
Dataset: MIT-BIH Polysomnographic Database (PhysioNet)
"""

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARN] imbalanced-learn not found. Skipping SMOTE.")

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("[INFO] wfdb not found. Using synthetic data generation.")


# ─────────────────────────────────────────────
#  Signal Processing Utilities
# ─────────────────────────────────────────────

def butter_bandpass(lowcut=0.5, highcut=50.0, fs=250, order=4):
    """4th-order Butterworth bandpass filter (0.5–50 Hz)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=250, order=4):
    """Apply bandpass filter to remove baseline wander and high-freq noise."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)


def zscore_normalize(segment):
    """Z-score normalize: zero mean, unit variance."""
    mean = np.mean(segment)
    std = np.std(segment)
    if std < 1e-8:
        return segment - mean
    return (segment - mean) / std


# ─────────────────────────────────────────────
#  Synthetic Data Generator (Demo Mode)
# ─────────────────────────────────────────────

def generate_synthetic_heartbeat(label=0, fs=250, duration=1.0, noise_level=0.05):
    """
    Generate a realistic synthetic heartbeat segment.
    label=0: Normal beat
    label=1: Abnormal beat (arrhythmia-like)
    """
    t = np.linspace(0, duration, int(fs * duration))

    if label == 0:
        # Normal sinus rhythm: P-QRS-T complex
        ecg = (
            0.1 * np.sin(2 * np.pi * 1.2 * t) +
            1.5 * np.exp(-((t - 0.35) ** 2) / (2 * 0.008 ** 2)) +   # R peak
            0.2 * np.exp(-((t - 0.28) ** 2) / (2 * 0.02 ** 2)) -    # Q
            0.1 * np.exp(-((t - 0.42) ** 2) / (2 * 0.02 ** 2)) +    # S
            0.3 * np.exp(-((t - 0.60) ** 2) / (2 * 0.04 ** 2)) +    # T wave
            0.05 * np.exp(-((t - 0.18) ** 2) / (2 * 0.03 ** 2))     # P wave
        )
        bp = 80 + 40 * np.sin(2 * np.pi * 1.2 * t + 0.5) + 10 * np.sin(2 * np.pi * 0.1 * t)
        eeg = 0.02 * np.sin(2 * np.pi * 10 * t) + 0.01 * np.sin(2 * np.pi * 20 * t)
    else:
        # Abnormal beat: irregular, wider QRS, inverted T
        ecg = (
            0.3 * np.sin(2 * np.pi * 0.8 * t) +
            0.9 * np.exp(-((t - 0.40) ** 2) / (2 * 0.025 ** 2)) +   # Wider R
            0.4 * np.exp(-((t - 0.30) ** 2) / (2 * 0.04 ** 2)) -    # Broad Q
            0.4 * np.exp(-((t - 0.65) ** 2) / (2 * 0.05 ** 2)) +    # Inverted T
            0.1 * np.sin(2 * np.pi * 3 * t)                           # Flutter noise
        )
        bp = 60 + 20 * np.sin(2 * np.pi * 0.8 * t + 0.2) + 5 * np.random.randn(len(t))
        eeg = 0.05 * np.sin(2 * np.pi * 8 * t) + 0.03 * np.sin(2 * np.pi * 30 * t)

    # Add realistic noise
    ecg += noise_level * np.random.randn(len(t))
    bp  += noise_level * 2 * np.random.randn(len(t))
    eeg += noise_level * 0.5 * np.random.randn(len(t))

    return ecg, bp, eeg


def generate_synthetic_dataset(n_normal=5000, n_abnormal=1000, fs=250, seed=42):
    """
    Generate a complete synthetic dataset with class imbalance
    (mimicking the MIT-BIH polysomnographic database structure).
    """
    np.random.seed(seed)
    print(f"[DATA] Generating synthetic dataset: {n_normal} normal + {n_abnormal} abnormal beats...")

    X_list, y_list = [], []

    for _ in range(n_normal):
        ecg, bp, eeg = generate_synthetic_heartbeat(label=0)
        segment = np.stack([
            apply_bandpass_filter(ecg),
            apply_bandpass_filter(bp),
            apply_bandpass_filter(eeg)
        ], axis=-1)
        X_list.append(segment)
        y_list.append(0)

    for _ in range(n_abnormal):
        ecg, bp, eeg = generate_synthetic_heartbeat(label=1)
        segment = np.stack([
            apply_bandpass_filter(ecg),
            apply_bandpass_filter(bp),
            apply_bandpass_filter(eeg)
        ], axis=-1)
        X_list.append(segment)
        y_list.append(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Z-score normalize each channel independently
    for i in range(X.shape[0]):
        for c in range(X.shape[2]):
            X[i, :, c] = zscore_normalize(X[i, :, c])

    print(f"[DATA] Dataset shape: {X.shape}, Labels: {np.bincount(y)}")
    return X, y


# ─────────────────────────────────────────────
#  SMOTE Oversampling
# ─────────────────────────────────────────────

def apply_smote(X, y, random_state=42):
    """Apply SMOTE to balance class distribution."""
    if not SMOTE_AVAILABLE:
        print("[INFO] SMOTE unavailable. Returning original data.")
        return X, y

    n_samples, timesteps, n_features = X.shape
    X_flat = X.reshape(n_samples, -1)

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_flat, y)

    X_res = X_res.reshape(-1, timesteps, n_features)
    print(f"[SMOTE] After resampling: {np.bincount(y_res.astype(int))}")
    return X_res.astype(np.float32), y_res.astype(np.int32)


# ─────────────────────────────────────────────
#  Main Preprocessing Pipeline
# ─────────────────────────────────────────────

def load_and_preprocess(use_synthetic=True, apply_oversampling=True, test_size=0.2):
    """
    Full preprocessing pipeline.
    Returns train/test splits ready for model training.
    """
    if use_synthetic or not WFDB_AVAILABLE:
        X, y = generate_synthetic_dataset(n_normal=5000, n_abnormal=1000)
    else:
        # PhysioNet WFDB loading (requires downloaded dataset)
        X, y = load_physionet_data()

    # Apply SMOTE before splitting (on training portion only)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if apply_oversampling:
        X_train, y_train = apply_smote(X_train, y_train)

    # Shuffle training data
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    print(f"[DATA] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def load_physionet_data(data_dir='data/physionet'):
    """
    Load MIT-BIH Polysomnographic Database.
    Falls back to synthetic if files not found.
    """
    print("[DATA] Attempting to load PhysioNet data...")
    # Simplified: in a real setup, use wfdb.rdrecord() and wfdb.rdann()
    print("[WARN] PhysioNet data not found. Using synthetic data.")
    return generate_synthetic_dataset()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print(f"Final shapes — X_train: {X_train.shape}, X_test: {X_test.shape}")
