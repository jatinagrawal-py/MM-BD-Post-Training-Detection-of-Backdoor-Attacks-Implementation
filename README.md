# MM-BD: Post-Training Backdoor Detection using Maximum Margin Statistics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20S%26P%202024-red.svg)](https://doi.org/10.1109/SP54263.2024.00015)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)](#license)

---

> **Implementation of the research paper:**
> *"MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic"*
> — Hang Wang, Zhen Xiang, David J. Miller, George Kesidis
> **2024 IEEE Symposium on Security and Privacy (SP)**

---

## Table of Contents

- [Overview](#overview)
- [Key Idea](#key-idea)
- [Paper Reference](#paper-reference)
- [Architecture](#architecture)
- [Experimental Setup](#experimental-setup)
- [Detection Pipeline](#detection-pipeline)
- [Results](#results)
- [Mitigation — MM-BM](#mitigation--mm-bm)
- [Why Model 2 Was Not Detected](#why-model-2-was-not-detected)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)
- [License](#license)

---

## Overview

This repository contains my implementation of the **MM-BD (Maximum Margin Backdoor Detection)** framework, applied to the **CIFAR-10** dataset using a modified **ResNet-18** architecture with an additional fully connected layer before the softmax output.

**What is a Backdoor Attack?**
A backdoor (Trojan) attack poisons a model's training data so that the model:
- Behaves **completely normally** on clean inputs
- **Misclassifies** any input embedded with a hidden trigger to the attacker's chosen target class

This makes backdoor attacks extremely dangerous — they cannot be detected using standard validation accuracy metrics.

**What does MM-BD do?**
MM-BD detects backdoor attacks **post-training** by analyzing the model's logit (pre-softmax) outputs. It requires:
- ❌ No clean training samples
- ❌ No knowledge of the trigger type
- ❌ No access to the original training set
- ✅ Only the trained model weights

---

## Key Idea

When a model is backdoor-attacked, the **target class develops an abnormally large margin** in logit space due to two effects caused by repeated trigger exposure during training:

| Effect | Description |
|--------|-------------|
| **Logit Boosting** | The target class logit becomes abnormally elevated |
| **Logit Suppression** | All other class logits are simultaneously depressed |

This creates a **maximum margin statistic** for the target class that is a clear statistical outlier compared to all non-target classes — and MM-BD exploits this signature for detection, regardless of the trigger type used.

---

## Paper Reference

```bibtex
@inproceedings{wang2024mmbd,
  title     = {MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary
               Backdoor Pattern Types Using a Maximum Margin Statistic},
  author    = {Hang Wang and Zhen Xiang and David J. Miller and George Kesidis},
  booktitle = {2024 IEEE Symposium on Security and Privacy (SP)},
  year      = {2024},
  doi       = {10.1109/SP54263.2024.00015}
}
```

---

## Architecture

The base architecture is **ResNet-18**, enhanced with an **extra fully connected layer with ReLU activation** inserted before the final classification layer.

```
┌──────────────────────────────────────┐
│         Input Image (32×32×3)        │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│     ResNet-18 Backbone               │
│     (Convolutional Feature Extractor)│
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│     Fully Connected Layer            │
│     (512 units, ReLU)                │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│  ★ ENHANCED: Extra FC Layer          │
│     (256 units, ReLU)                │
│     ← Custom Addition                │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│     Output FC Layer                  │
│     (10 units — one per class)       │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│            Softmax                   │
└──────────────────────────────────────┘
```

### Why the Extra Layer?

Adding an extra FC layer before softmax increases the representational capacity of the classification head. This allows the model to learn richer feature combinations from the ResNet backbone before producing final class scores (logits), potentially making the maximum margin anomaly even more pronounced for backdoored target classes and improving the sensitivity of MM-BD detection.

---

## Experimental Setup

### Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (60,000 images, 10 classes) |
| Architecture | ResNet-18 + Extra FC Layer (256 units) |
| Attack Type | BadNet — 3×3 patch trigger (bottom-right corner) |
| Number of Models | 3 backdoored models |
| Training Epochs | 20 per model |
| Optimizer | Adam |
| Poisoned Train Samples | 500 per model |
| Poisoned Test Samples | 1000 per model |
| Detection Threshold (θ) | 0.05 (95% confidence) |
| Null Distribution | Gamma distribution |

### Backdoor Attack Configurations

| Model | Source Class | Target Class | Train Poison | Test Poison |
|-------|-------------|--------------|--------------|-------------|
| Model 0 | Airplane | Car | 500 | 1000 |
| Model 1 | Car | Bird | 500 | 1000 |
| Model 2 | Bird | Cat | 500 | 1000 |

### Backdoor Embedding Formula (BadNet)

```
x̃ = (1 − m) ⊙ x + m ⊙ u
```

| Symbol | Description |
|--------|-------------|
| `x` | Original clean image |
| `u` | Trigger pattern (3×3 patch) |
| `m` | Binary mask (1 at trigger location, 0 elsewhere) |
| `x̃` | Resulting poisoned image |
| `⊙` | Element-wise multiplication |

---

## Detection Pipeline

MM-BD detection runs in two steps:

### Step 1 — Estimation: Compute Maximum Margin Statistic

For each class `c`, solve the following optimization via **gradient ascent**:

```
r_c = maximize  [ g_c(x) − max g_k(x) ]
        x ∈ X              k ≠ c
```

- `g_c(x)` = logit (raw score) of class `c` for input `x`
- Gradient ascent starts from **multiple random initializations**
- No real data samples are needed — inputs start as random noise
- The largest solution across all initializations is taken as `r_c`

### Step 2 — Inference: Statistical Anomaly Detection

```
r_max = max r_c          (largest margin across all classes)
         c ∈ Y

pv = 1 − H_0(r_max)^(K−1)
```

| Symbol | Description |
|--------|-------------|
| `r_max` | Largest maximum margin statistic |
| `H_0` | Null distribution estimated from all statistics except `r_max` |
| `K` | Total number of classes (10 for CIFAR-10) |
| `pv` | Computed p-value |

### Decision Rule

```
if pv < 0.05  →  ✅ Backdoor Attack Detected
                    (class with r_max is the predicted target class)

if pv ≥ 0.05  →  ✅ Model appears Clean
```

---

## Results

### Training Performance

All three backdoored models successfully achieved the hallmark of a stealthy backdoor attack — high clean accuracy alongside a very high attack success rate.

| Model | Attack | Clean ACC | ASR (Before Mitigation) |
|-------|--------|-----------|--------------------------|
| Model 0 | plane → car | 88.4% | 99.9% |
| Model 1 | car → bird | 89.1% | 97.7% |
| Model 2 | bird → cat | 87.9% | 98.7% |

> Clean accuracy stays ~88–89% while ASR reaches ~98–99%, confirming all three attacks are **stealthy and effective**.

---

### Detection Results

| Model | True Target | Predicted Target | P-value | Detected? |
|-------|------------|-----------------|---------|-----------|
| Model 0 | car | car ✓ | 5.31e-10 | ✅ YES |
| Model 1 | bird | bird ✓ | 1.57e-12 | ✅ YES |
| Model 2 | cat | cat ✓ | 3.22e-01 | ❌ NO |

- **Model 0 & 1**: Extremely low p-values (far below 0.05) confirm the target class is a **strong statistical outlier** in the margin distribution.
- **Model 2**: P-value of 0.322 is well above the 0.05 threshold — not detected. See [Why Model 2 Was Not Detected](#why-model-2-was-not-detected).

---

### After Mitigation (MM-BM)

| Model | Attack | Clean ACC (After) | ASR (After) | Result |
|-------|--------|-------------------|-------------|--------|
| Model 0 | plane → car | 86.4% | 7.7% | ⚠️ Partial |
| Model 1 | car → bird | 86.4% | 0.8% | ⚠️ Partial |
| Model 2 | bird → cat | 87.3% | 20.8% | ⚠️ Partial |

---

### Comparison with Original Paper

| Metric | Original Paper | This Implementation |
|--------|---------------|---------------------|
| Dataset | CIFAR-10 | CIFAR-10 |
| Architecture | ResNet-18 | ResNet-18 + Extra FC Layer |
| Training Epochs | 60 | 20 |
| ASR Before Mitigation | ~99% | 98.77% (avg) |
| ASR After Mitigation | ~3–10% | 9.77% (avg) |
| Detection Rate | ~90% | 66.7% (2 out of 3) |

---

## Mitigation — MM-BM

Once a backdoor attack is detected, **MM-BM (Maximum Margin Backdoor Mitigation)** is applied to neutralize it.

### Core Concept

A backdoored model has neurons that fire with **abnormally large activations** in response to the trigger. MM-BM suppresses this by applying an **optimized upper bound (clamp)** to each neuron:

```
output = min(activation, z)
```

### How MM-BM Works

```
1. For each neuron in selected layers, initialize a large upper bound z
2. Collect ~20 clean images per class (very few needed)
3. Optimize z using gradient descent:
   - Keep logits unchanged for clean images (preserves accuracy)
   - Minimize the L2 norm of z (suppresses large activations)
4. Apply the learned clamps to the model at inference time
```

### Optimization Objective

```
minimize   Σ_l  ||z_l||²
   Z

subject to:  (1/|D|) × Σ 1[y = argmax g̃_c(x; Z)] ≥ π = 0.95
```

### Key Properties of MM-BM

| Property | Details |
|----------|---------|
| Clean images needed | ~20 per class (very few) |
| Model parameters modified | ❌ None — original weights untouched |
| Architecture modified | ❌ No — only clamps are added |
| Works for multiple trigger types | ✅ Yes (except very small global patterns) |

---

## Why Model 2 Was Not Detected

Model 2 (bird → cat) returned a p-value of **0.322**, well above the detection threshold of 0.05. Analysis of possible causes:

| Cause | Explanation |
|-------|-------------|
| ⏱️ Too few training epochs | Only 20 epochs used; paper recommends 60–100 for full convergence |
| 📉 Weak backdoor signal | The margin anomaly was not pronounced enough to be a statistical outlier |
| 🔁 Insufficient overfitting | The target logit separation had not fully developed by epoch 20 |
| 🐦🐱 Semantic class similarity | Natural closeness of "bird" and "cat" in CIFAR-10 may reduce the relative margin gap |

This is consistent with the original paper's discussion that attacks with lower attack convergence (or lower poisoning rates) are harder for MM-BD to detect.

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn jupyter
```

### Clone the Repository

```bash
git clone https://github.com/your-username/MM-BD-Implementation.git
cd MM-BD-Implementation
```

### Run the Notebook

```bash
jupyter notebook research-paper-implementation.ipynb
```

### Notebook Structure

The notebook is organized in the following order:

```
Section 1  →  Imports and Environment Setup
Section 2  →  CIFAR-10 Dataset Loading and Preprocessing
Section 3  →  BadNet Trigger Generation (3×3 patch)
Section 4  →  Training Data Poisoning
Section 5  →  Model Definition (ResNet-18 + Extra FC Layer)
Section 6  →  Model Training (3 backdoored models)
Section 7  →  Training Curve Visualization (ACC + ASR per epoch)
Section 8  →  Backdoor Trigger Visualization
Section 9  →  MM-BD Detection Pipeline
              ├── Gradient Ascent Optimization
              ├── Maximum Margin Statistic Computation
              ├── Null Distribution Estimation (Gamma)
              ├── P-value Calculation
              └── Detection Decision
Section 10 →  Detection Result Visualization (P-value bar charts)
Section 11 →  MM-BM Mitigation
              ├── Neuron Activation Upper Bound Optimization
              └── Post-mitigation ASR and ACC Evaluation
Section 12 →  Final Results Table and Comparison with Paper
```

---

## Limitations

| Limitation | Details |
|------------|---------|
| Small scale | Only 3 models trained — limited statistical significance |
| Low epoch count | 20 epochs vs the recommended 60–100 in the paper |
| Single trigger type | Only BadNet patch evaluated; paper tests 5 trigger types |
| One detection failure | Model 2 not detected due to weak backdoor convergence |
| Partial mitigation | MM-BM could not fully suppress ASR in all cases |
| Compute budget | Experiments constrained by available GPU/CPU resources |

---

## Future Work

- [ ] Increase training epochs to 60+ for stronger and more reliable backdoor convergence
- [ ] Evaluate additional trigger types:
  - [ ] Chessboard (global additive pattern)
  - [ ] 1-pixel perturbation
  - [ ] Blended noise patch
  - [ ] WaNet (warping-based trigger)
  - [ ] Sample-specific / input-aware triggers
- [ ] Scale experiments to CIFAR-100 and TinyImageNet
- [ ] Implement hybrid detection combining MM-BD with:
  - [ ] Activation Clustering
  - [ ] Frequency Domain Analysis
  - [ ] Entropy-Based Detection
- [ ] Systematically study how extra FC layer depth affects the margin signal
- [ ] Implement online backdoor detection during training
- [ ] Evaluate robustness against adaptive attacks (min-max optimization)
- [ ] Extend to NLP and speech classification domains

---

## References

1. **Wang, H., Xiang, Z., Miller, D. J., & Kesidis, G. (2024).**
   *MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic.*
   2024 IEEE Symposium on Security and Privacy (SP).
   https://doi.org/10.1109/SP54263.2024.00015

2. **Gu, T., Liu, K., Dolan-Gavitt, B., & Garg, S. (2019).**
   *BadNets: Evaluating Backdooring Attacks on Deep Neural Networks.*
   IEEE Access, 7, 47230–47244.

3. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).**
   *Deep Residual Learning for Image Recognition.*
   CVPR 2016.

4. **Wang, B., Yao, Y., Shan, S., et al. (2019).**
   *Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks.*
   IEEE S&P 2019.

5. **Liu, Y., Lee, W., Tao, G., et al. (2019).**
   *ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation.*
   ACM CCS 2019.

6. **Xiang, Z., Miller, D. J., & Kesidis, G. (2020).**
   *Detection of Backdoors in Trained Classifiers Without Access to the Training Set.*
   IEEE TNNLS.

---

## Author

**Jatin Agrawal**
Roll No: `2023UAI1792`

---

## License

This repository is intended for **academic and educational purposes only**.
All backdoor attack implementations are strictly for research use and
for understanding defensive mechanisms against adversarial threats in
deep learning systems.

---

*Submitted as part of a research paper implementation assignment — May 2026*
