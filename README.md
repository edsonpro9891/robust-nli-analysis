# Data-Centric NLP: Detecting Bias and Stopping LLMs from Cheating to Achieve 89% Accuracy with 40% Less Data (Mitigating Hypothesis-Only Artifacts in SNLI via Hard-Subset Filtering)

### 🚀 Project Overview
Modern NLP models often "cheat" on benchmarks by exploiting statistical loopholes (dataset artifacts) instead of learning semantic reasoning. This project identifies these vulnerabilities in the SNLI dataset and implements a **Data-Centric AI pipeline** to fix them.

**The Result:** A robust model that maintains state-of-the-art performance (89.0%) despite being trained on a dataset that is **40% smaller and significantly harder**.

---

### 🚩 The Problem: Models that "Guess" instead of "Read"
I trained a baseline model given **only the hypothesis** (hiding the premise). In a rigorous test, this model should score ~33% (random chance).
* **My Analysis:** The model achieved **70.3% accuracy** without seeing the context.
* **Root Cause:** The model learned that words like "nobody" or "sleeping" strongly correlate with specific labels, allowing it to bypass actual reasoning.

### 🛠️ The Solution: Hard-Subset Filtering
Instead of making the model larger, I improved the data quality.
1.  **Audit:** Used the "cheating" model to identify 400,000+ "easy" examples that contained artifacts.
2.  **Filter:** Automatically curated a new training set by removing low-quality, artifact-heavy data.
3.  **Retrain:** Trained a new model on this "Hard Subset."

### 📊 Key Results

| Model | Training Data Size | Accuracy | Note |
| :--- | :--- | :--- | :--- |
| **Standard Baseline** | 550k (100%) | 89.8% | Bloated with artifacts |
| **Robust Model (Ours)** | **350k (60%)** | **89.0%** | **Efficient & Semantic** |

### 🧠 Skills Demonstrated
* **NLP & Transformers:** Fine-tuning ELECTRA models using Hugging Face & PyTorch.
* **Model Debugging:** Identifying overfitting behaviors and data leakage.
* **Data Engineering:** Programmatic dataset curation and filtering.
* **Experimentation:** rigorous ablation studies and error analysis.

### 📄 Full Research Paper
For a deep dive into the methodology and error analysis, please read the full report:
[📄 **Download Full PDF Report**](Robust_NLI_Analysis.pdf)

### Code
`filter_dataset.py` – custom filtering logic using the hypothesis-only model as a bias probe  

---
*Note: The core training code is omitted to comply with course academic integrity policies. This repository contains the custom filtering logic (`filter_dataset.py`) and the final analysis.*

**Private full version** (code + notebooks + reproduction steps) available upon request.
