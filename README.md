# 🫀 Privacy-Preserving Heartbeat Abnormality Classification

**Using Differentially Private Deep Learning Models with Formal Privacy Guarantees**

## 📝 Project Overview

This project demonstrates privacy-preserving machine learning for medical heartbeat classification. It uses Differentially Private Stochastic Gradient Descent (DP-SGD) for formal privacy, supporting multiple deep learning models (CNN, RNN, LSTM, GRU), explainability, adversarial robustness, and an interactive dashboard.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- (Optional) NVIDIA GPU with CUDA for full training mode

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/heartbeat_privacy_project.git
cd heartbeat_privacy_project
pip install -r requirements.txt
```

### One-Click Setup (Recommended)

```bash
python setup_and_run.py
# Opens dashboard automatically at http://localhost:5000
```

### Run Individual Modes

Demo Mode (No GPU required):

```bash
python main.py --mode demo
```

Full Training (GPU recommended):

```bash
python main.py --mode full
```

Generate Plots Only:

```bash
python main.py --mode plots
```

Evaluation & Security Analysis:

```bash
python main.py --mode evaluate
```

Interactive Dashboard:

```bash
python dashboard.py
# Visit: http://localhost:5000
```

---

## 📁 Project Structure

```
heartbeat_privacy_project/
│
├── main.py                         # Entry point - orchestrates all modes
├── dashboard.py                    # Flask web dashboard
├── setup_and_run.py                # One-click environment setup
├── requirements.txt                # Python package dependencies
├── README.md                       # This file
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── differential_privacy.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── explainability.py
│   ├── visualizer.py
│   └── extended_plots.py
│
├── data/                           # Input signals
│
├── models/                         # Saved model checkpoints
│
├── results/                        # Output analysis & visualizations
│
├── templates/
│   └── dashboard.html
│
└── static/
    ├── css/
    └── js/
```

---

**For full details and documentation, see the source code and doc files in the repo!**
