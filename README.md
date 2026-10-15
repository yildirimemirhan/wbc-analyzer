# WBC Analyzer: Robust OOD Generalization in Peripheral Blood Smears

[![DOI](https://img.shields.io/badge/DOI-10.13140/RG.2.2.34201.79208-blue.svg)](https://doi.org/10.13140/RG.2.2.34201.79208)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-FF6F00.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000.svg)](https://flask.palletsprojects.com/)

An end-to-end, production-ready computer vision and software architecture designed to solve the domain shift problem in automated white blood cell (WBC) classification.

While vanilla deep learning models severely degrade when exposed to out-of-distribution (OOD) data arising from different microscopy hardware and staining protocols, this system achieves a **98.53% in-distribution accuracy** while simultaneously demonstrating extreme resilience to domain shift, reaching **89.05% OOD accuracy (a +32.09 percentage point boost over the unadapted baseline)** via a completely retraining-free inference adaptation pipeline.

<p align="center">
  <a href="README.tr.md">🇹🇷 Türkçe</a> &nbsp;|&nbsp; 🇬🇧 English
</p>

<p align="center">
  <a href="https://emirhanyildirim.me/wbc-analyzer/" target="_blank" rel="noopener">
Live Demo
  </a>
</p>

---

## ⚡ Key Contributions

- **Medical Enhanced Filter (MEF):** A deterministic, 5-step image preprocessing pipeline that normalizes cross-device brightness, exposure, and color variability at the pixel level _before_ feature extraction.
- **WBCAttention & MedSwish:** A sequential, parameter-efficient CBAM-style attention block (132K params) combined with a custom activation function utilizing learnable parameters ($\alpha, \beta$) to suppress the "Dying ReLU" effect on fine morphological chromatin details.
- **Dynamic Training-Time XAI Guardrail:** Features `XAIFocusMonitor`, a custom Keras callback that actively calculates Grad-CAM foreground focus ratios during training, automatically stopping execution if the model attempts to exploit spurious background correlations (shortcut learning).
- **Closed-Loop Remediation Interface:** Implements an agentic inference head powered by an autonomous multi-modal LLM (GPT-4o with a localized Gemini 2.5 Flash fallback) that interprets Grad-CAM heatmaps post-hoc, validating model focus against hematological criteria and dynamically triggering stain normalization if background focus is detected.

---

## 📊 Benchmarks & Performance

### 1. Robustness Under Domain Shift (Raabin-WBC Dataset)

Evaluated on Giemsa-stained peripheral blood smear images across hardware splits (Professional Laboratory Camera vs. Consumer Smartphone).

| Evaluation Set | Target Distribution       | n     | Base Accuracy | Proposed Pipeline Accuracy | Weighted F1 |
| :------------- | :------------------------ | :---- | :-----------: | :------------------------: | :---------: |
| **TestA**      | In-Distribution (IND)     | 4,339 |    97.46%     |         **98.53%**         | **0.9854**  |
| **TestB**      | Out-of-Distribution (OOD) | 2,119 |    56.96%     |         **89.05%**         | **0.9111**  |
| **Combined**   | Joint Evaluation          | 6,458 |    84.17%     |         **95.42%**         | **0.9554**  |

_Note: TestB captures severe hardware-induced domain shift (contains Lymphocyte and Neutrophil classes collected from unseen acquisition devices)._

### 2. Class-Level Granularity (TestA IND)

Extreme class imbalances (e.g., Basophil rarity) are managed natively via class-weighted **WBCFocalLoss**:

| Leukocyte Subtype | Precision | Recall | F1-Score | Support |
| :---------------- | :-------: | :----: | :------: | :-----: |
| **Basophil**      |  1.0000   | 1.0000 |  1.0000  |   89    |
| **Eosinophil**    |  0.9265   | 0.9783 |  0.9517  |   322   |
| **Lymphocyte**    |  0.9865   | 0.9884 |  0.9874  |  1,034  |
| **Monocyte**      |  0.9372   | 0.9573 |  0.9471  |   234   |
| **Neutrophil**    |  0.9962   | 0.9868 |  0.9915  |  2,660  |

### 3. Cross-Backbone Efficiency & Latency

All models were trained under identical configurations to benchmark the footprint vs. performance trade-offs.

| Core Architecture Backbone                           | Total Parameters | Validation Accuracy |  Macro F1  | Inference Latency |
| :--------------------------------------------------- | :--------------: | :-----------------: | :--------: | :---------------: |
| VGG16                                                |     15.11 M      |       98.56%        |   0.9724   |      18.1 ms      |
| ResNet50V2                                           |     24.75 M      |       98.17%        |   0.9704   |     103.9 ms      |
| MobileNetV2                                          |      3.05 M      |       97.90%        |   0.9577   |      96.0 ms      |
| EfficientNetB0                                       |      4.84 M      |       97.05%        |   0.9418   |     185.4 ms      |
| DenseNet121 (Vanilla)                                |      7.70 M      |     **98.89%**      |   0.9803   |     232.2 ms      |
| **Proposed (DenseNet121 + WBCAttention + MedSwish)** |    **7.83 M**    |       98.53%        | **0.9853** |    **14.2 ms**    |

---

## 🛠️ Deep Dive: Preprocessing & Adaptation

### The 5-Step Medical Enhanced Filter (MEF) Pipeline

1. **Percentile Clipping:** Standardizes luminance by stretching the 2nd–98th percentile per channel to suppress microscope exposure variations.
2. **Dual-Scale LAB CLAHE:** Applies Local Contrast Enhancement exclusively to the L-channel using fused tile configurations ($4\times4$ for nuclear chromatin, $8\times8$ for cytoplasmic boundaries) via Canny edge-weighted masks to block hue shifts.
3. **Bilateral Filtering:** Cleans microscopy shot noise ($d=9, \sigma_c=65, \sigma_s=65$) while preserving membrane boundaries.
4. **Morphological Nucleus Highlighting:** Computes blended inner ($k_{3\times3}$) and outer ($k_{7\times7}$) elliptical gradients to explicitly amplify nuclear lobation structures.
5. **Selective LoG Sharpening:** Applies localized Laplacian-of-Gaussian sharpening exclusively to edge boundaries, leaving flat backgrounds intact.

### Preprocessing Ablation Insights

The data demonstrates that aggressive structural tampering without calibration degrades cytoplasm-dependent subtypes:

| Preprocessing Variant Configuration                  | TestA (IND) | TestB (OOD) |  Combined  |
| :--------------------------------------------------- | :---------: | :---------: | :--------: |
| **v1 — MEF Original (Proposed Configuration)**       | **98.41%**  |   85.65%    |   94.22%   |
| v2 — Adaptive CLAHE TileGrid ($8\times8$ Only)       |   97.99%    | **87.92%**  | **94.69%** |
| v3 — v2 + Top-Hat / Bottom-Hat Morphology            |   95.18%    |   77.58%    |   89.41%   |
| v4 — v3 + Macenko Stain Normalization (Uncalibrated) |   57.78%    |   42.28%    |   52.69%   |

---

## 📂 Project Structure

wbc-final/
├── app.py # Production Flask API + Multi-Modal Agent Orchestration
├── train_main_model.py # Two-Phase Curriculum Training + Online XAI Monitoring
├── train_baseline_comparison.py # Comparative Benchmarking Engine for Cross-Backbones
├── eval_final.py # Evaluation Wrapper (TTA + Binary Routing + Reinhard)
├── eval_baseline.py # Baseline Backbone Isolation Validation Engine
├── preprint/ # Academic Publication Artifacts
│ ├── wbc_preprint.pdf # Compiled arXiv preprint (Full Paper)
│ ├── main.tex # LaTeX source code for the manuscript
│ └── references.bib # BibTeX citation library
├── src/
│ ├── custom_layers.py # Tensor Definitions for WBCAttentionBlock & MedSwish
│ ├── custom_losses.py # Class-Weighted WBCFocalLoss Matrix Definitions
│ └── preprocessing.py # Operational Implementations of MEF (v1–v4)
├── data/
│ ├── models/ # Local Storage Bin for Production Weights
│ └── raabin-wbc-data/ # Structural Directory for Train/TestA/TestB Partitions
└── outputs/ # Runtime Target Directory for Classification Matrices & Reports

## 🚀 Quick Start & Deployment

### Installation

Clone the repository and install Python dependencies:

```bash
git clone https://github.com/yildirimemirhan/wbc-analyzer.git
cd wbc-analyzer
pip install -r requirements.txt
```

### Fetch Production Model Weights

Download the production model file `wbc_final_model_densenet.keras` and place it under:

```
data/models/wbc_final_model_densenet.keras
```

(The repository includes a `data/models/` folder where production weights are expected.)

### Environment Configuration

Create a `.env` file in the project root to store API tokens used by the multi-modal agent layers. Example:

```
GITHUB_TOKEN=your_github_models_token
GEMINI_API_KEY=your_gemini_api_key
```

Keep this file out of version control (add to `.gitignore`) for security.

### Run the Production Server

Start the Flask production engine:

```bash
python app.py
```

The server will start on http://localhost:5000 by default. You can POST microscopy images to the `/predict` endpoint to receive class predictions, Grad-CAM overlays and LLM-based analytical reports.

> Note for Windows developers: for native GPU acceleration run scripts in WSL2 with CUDA Toolkit configured.

### Reproduce Evaluation & Training

Run the final evaluation (inference-time adaptation stack: Reinhard color normalization + binary routing + light TTA):

```bash
python eval_final.py \
  --model-path data/models/wbc_final_model_densenet.keras \
  --data-root data/raabin-wbc-data \
  --output-dir outputs/final_model_results \
  --testb-binary-mode main \
  --tta light \
  --color-normalization reinhard \
  --preprocessing v1
```

Train the two-phase curriculum from scratch:

```bash
python train_main_model.py \
  --data-root data/raabin-wbc-data \
  --phase1-epochs 15 \
  --phase2-epochs 15 \
  --main-loss cce \
  --label-smoothing 0.1 \
  --crop-prob 0.2 \
  --bg-randomization-prob 0.15 \
  --stain-jitter-prob 0.3 \
  --aux-loss-weight 1.0 \
  --xai-focus-threshold 0.55 \
  --xai-every-n-epochs 2 \
  --model-path data/models/wbc_final_model_densenet.keras
```

### Kaggle Notebook (no setup)

`notebooks/wbc_analyzer_kaggle_quickstart.ipynb` is a self-contained quickstart: it embeds the
custom layers, the preprocessing filter and five sample images, loads the model, runs predictions
and Grad-CAM, classifies a folder of your own images, and optionally evaluates on Raabin-WBC. It
also loads the model on Keras versions older than the 3.14 it was saved with. After changing
`src/` or `docs/samples/`, regenerate it with `python notebooks/build_kaggle_notebook.py`
(`tests/test_notebook.py` fails if it is stale).

### API Reference

Request (multipart/form-data):

POST /predict

Form field:

- `file` — binary stream of the microscopy image (JPG, PNG, BMP, TIFF, WebP accepted)

Successful JSON response (200 OK) example:

```json
{
    "class": "Neutrophil",
    "confidence": 0.977,
    "all_probabilities": {
        "Basophil": 0.001,
        "Eosinophil": 0.002,
        "Lymphocyte": 0.012,
        "Monocyte": 0.008,
        "Neutrophil": 0.977
    },
    "gradcam_image": "data:image/png;base64,iVBORw0KGgo...",
    "llm_report": "Grad-CAM confirmation report: Model focus heavily localized on primary nuclear lobation patterns and fine violet cytoplasmic granulation. Zero background shortcuts detected."
}
```

### Citation

If you use this work, cite:

```bibtex
@article{yildirim2026wbc,
  title={Achieving Robust Out-of-Distribution Generalization in Peripheral Blood Smears via Custom Attention Mechanisms, Medical Enhanced Filtering, and Inference-Time Domain Adaptation},
  author={Yildirim, Emirhan},
  publisher={ResearchGate},
  doi={10.13140/RG.2.2.34201.79208},
  url={https://doi.org/10.13140/RG.2.2.34201.79208},
  year={2026}
}
```
