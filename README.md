
# 🧠 AGTSU (Attention-Guided Time Series Unlearning)

**Agatsu (吾勝)** — _"Victory over oneself"_ — lies at the heart of this work.  
Inspired by this concept, our framework advocates not just for better performance, but for **model introspection, refinement, and the selective unlearning of what no longer serves**.  

Hence, by learning to forget spurious patterns, models embody the essence of Agatsu — achieving robustness through **self-directed improvement**.

---

This repository accompanies the paper:  
**"Unlearning the Irrelevant: Machine Unlearning in Time Series Forecasting and Classification" (AAAI-26 Submission)**

We introduce the first end-to-end framework for **selective unlearning in time series data**, addressing spurious temporal segments in both forecasting and classification tasks. The pipeline includes anomaly detection using attention-performance discrepancy (APD) scores, interpretable temporal replacement, and efficient unlearning through exact and approximate techniques.

---

## 📌 Key Contributions

- ⚡ **Attention-guided Unlearning** for identifying and replacing harmful temporal segments.
- 📈 **Exact Unlearning** via Temporal SISA and ARCANE-like modular retraining.
- 🔁 **Approximate Unlearning** via temporal knowledge distillation, pruning, and calibration.
- 🔍 **APD Score** to measure misalignment between attention, prediction error, and local deviation.
- 📊 Compatible with **Autoformer**, **VSFormer**, and other SOTA time series models.

---

## 🗂️ Project Structure

| File / Module                                | Description |
|---------------------------------------------|-------------|
| `ADP_score.py`                               | Computes APD score using attention, loss, and deviation. Detects spurious time steps. |
| `AttentionVectorExtraction.py`              | Extracts and visualizes attention maps (layer-wise) for trained models. |
| `Model_ADP_TempReplace_forclean.py`         | Core script for spurious segment identification and temporal replacement via attention-context similarity. |
| `TempReplace.py`                             | Lightweight replacement strategies (context-aware interpolation, autocorr/trend/seasonal alignment). |
| `Single-seriesAGTSU.py`                     | Implements Temporal-SISA exact unlearning for a single time series. |
| `Multi-seriesAGTSU_with_ImprovedTempReplace.py` | Full unlearning pipeline across multiple time series, combining APD, replacement, and retraining. |
| `README.md`                                  | This document. |
| `Time_Series_Unlearning___AAAI26.pdf`        | The AAAI paper describing the methodology and experiments. |

---

## 🔧 Setup Instructions

### 📦 Dependencies
Install the required packages:

```bash
pip install -r requirements.txt
```

**Required Libraries:**

- `torch`, `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`, `scipy`
- `Autoformer` (from [Autoformer GitHub](https://github.com/thuml/Autoformer))

---

## 🚀 How to Run

### 1. Extract Attention & Visualize

```bash
python AttentionVectorExtraction.py
```

- Extracts encoder attention maps from trained Autoformer models.
- Saves heatmaps and `.npy` attention arrays.

---

### 2. Compute APD Scores (Spurious Detection)

```bash
python ADP_score.py
```

- Computes the Attention-Performance Discrepancy score.
- Identifies spurious time steps and generates diagnostic plots.

---

### 3. Perform Temporal Replacement

```bash
python Model_ADP_TempReplace_forclean.py
```

- Replaces spurious segments using attention-aware and temporal-consistency criteria.
- Saves replacement logs and quality metrics.

---

### 4. Run Multi-Series Unlearning Pipeline

```bash
python Multi-seriesAGTSU_with_ImprovedTempReplace.py
```

- Applies APD + replacement + evaluation across thousands of series.
- Saves improved metrics (MSE, MAE, RMSE, MAPE).

---

### 5. Apply Exact Unlearning (Temporal SISA)

```bash
python Single-seriesAGTSU.py
```

- Runs autocorrelation-aware sharding.
- Retrains only affected shards with removed spurious segments.

---

## 📊 Example Outputs

- Attention heatmaps (per layer/head)
- Time series plots with spurious segments highlighted
- APD score curves and thresholds
- Replacement quality metrics (autocorr, trend, seasonality)
- Improved forecast and classification metrics

---

## 🧪 Datasets

This repo supports multiple benchmark datasets:
- M5 Forecasting (default path: `./Autoformer/dataset/m5_series_split`)
- ECL, METR-LA, Traffic (modify path in config)
- Classification datasets via UEA format (for VSFormer etc.)

To reproduce M5 experiments, ensure the time series CSVs are placed in the directory structure:
```
Autoformer/
└── dataset/
    └── m5_series_split/
        ├── FOODS_1_001_CA_1.csv
        ├── FOODS_1_001_CA_1_train.csv
        └── ...
```

---

## 🧠 Method Summary

Our unlearning framework follows 3 key stages:

1. **Spurious Detection**: Compute APD scores using:
   - Attention alignment failure
   - Prediction error
   - Statistical deviation

2. **Replacement**:
   - Find a similar clean segment using attention-context similarity.
   - Ensure temporal coherence via autocorrelation, trend continuity, and season alignment.

3. **Unlearning**:
   - Exact: Temporal SISA or ARCANE with sharded retraining.
   - Approximate: SCRUB (masked KD), weight pruning, or output calibration.

---

## 📄 Citation

If you find this work useful, please cite:

```
@article{anonymous2025unlearning,
  title={Unlearning the Irrelevant: Machine Unlearning in Time Series Forecasting and Classification},
  author={Anonymous Submission},
  journal={AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---

## 📬 Contact

For questions, issues, or collaborations, please open a GitHub issue or contact the paper authors after the anonymity period.
