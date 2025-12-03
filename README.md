

# ANALYZING THE ROBUSTNESS OF BLOOD CELL CLASSIFIERS


This is a **complete, detailed, and step-by-step README** for our **White Blood Cell (WBC) Classification** project using Deep Learning, Trustworthiness evaluation, and a Streamlit user interface.

It explains:

* What the project does
* Which models we used (Custom CNN, VGG16, ResNet50)
* How we trained and evaluated them
* How we evaluated **trustworthiness**, with a primary focus on **Robustness & Sensitivity**
* How we used **Fisher Information Matrix (FIM)** and **Influence Functions (IF)**
* How we used **Grad-CAM** for explainability
* Why our selected model is the **recommended model for deployment**
* How to **run the notebooks** and **Streamlit application**
* How to reproduce robustness and trustworthiness results

---

## Table of Contents

1. [Project Overview](#overview)
2. [Project Goals](#project-goals)
3. [Dataset Details](#dataset-details)
4. [Environment Setup](#environment-setup)
5. [Folder Structure](#folder-structure)
6. [Models Trained](#models-trained)
7. [Performance Metrics (Accuracy & Validation)](#performance-metrics-accuracy--validation)
8. [Confusion Matrix & Classification Metrics](#confusion-matrix--classification-metrics)
9. [Robustness Evaluation (Trustworthiness)](#robustness-evaluation-trustworthiness)
10. [Sensitivity Evaluation (Fisher Information Matrix)](#sensitivity-evaluation-fisher-information-matrix)
11. [Influence Functions Evaluation](#influence-functions-evaluation)
12. [Explainability Evaluation (Grad-CAM)](#explainability-evaluation-grad-cam)
13. [Why the Selected Model is Recommended](#why-the-selected-model-is-recommended)
14. [Reliability & Limitations](#reliability--limitations)
15. [Download Final Models](#download-final-models)
16. [Streamlit Deployment](#streamlit-deployment)
17. [Run the Application](#run-the-application)
18. [Workflow Summary (Step-by-Step)](#workflow-summary-step-by-step)
19. [Quick Commands Summary](#quick-commands-summary)
20. [Troubleshooting](#troubleshooting)
21. [Future Enhancements](#future-enhancements)
22. [References & Code Attribution](#references--code-attribution)
23. [Final Summary](#final-summary)
24. [Team Members & Contributions](#team-members--contributions)

---

<a name="overview"></a>

# 1. Project Overview

### 1.1 What this project does

This project:

1. Takes a **microscopic image of a white blood cell** as input.
2. Uses deep learning models to classify it into one of **five WBC types**:

   * Neutrophil
   * Lymphocyte
   * Monocyte
   * Eosinophil
   * Basophil
3. Evaluates **how trustworthy** the classifier is by studying:

   * Robustness to perturbations
   * Sensitivity using **Fisher Information**
   * Training data impact using **Influence Functions**
   * Visual explanations with **Grad-CAM**
4. Exposes the final model via a **Streamlit web app** to allow interactive testing.

### 1.2 Technologies used

* **Python** – core scripting language
* **TensorFlow + Keras** – deep learning framework
* **NumPy, Pandas** – data processing
* **Matplotlib, Seaborn** – visualizations
* **Scikit-learn** – metrics and confusion matrices
* **Pillow (PIL)** – image handling
* **KaggleHub** – dataset download
* **Streamlit** – web user interface

### 1.3 What is included in this project

This repository includes:

1. **Data loading and preprocessing** code
2. Training of **three CNN-based models**:

   * Custom CNN
   * VGG16 (transfer learning)
   * ResNet50 (transfer learning)
3. **Accuracy and metric evaluation** to compare models
4. **Trustworthiness analysis**, focusing on:

   * Sensitivity via **Fisher Information Matrix (FIM)**
   * Training sample impact via **Influence Functions (IF)**
   * Robustness under perturbations (noise, blur, brightness)
   * Explainability via **Grad-CAM heatmaps**
5. A **Streamlit app** for real-time WBC classification and interpretability.

---

<a name="project-goals"></a>

# 2. Project Goals

### 2.1 Core goals

1. Build an accurate classifier for **white blood cell types**.
2. Use multiple architectures (Custom CNN, VGG16, ResNet50) and compare them.
3. Implement a user interface to demonstrate the model’s behavior.

### 2.2 Trustworthiness goals

The main **trustworthiness aspect** considered in this project is:

> **Robustness & Sensitivity (Reliability of Predictions)**

Concretely, we:

* Analyze **sensitivity** using **Fisher Information Matrix (FIM)** to identify:

  * Stable vs unstable predictions
* Analyze **training sample influence** using **Influence Functions (IF)** to identify:

  * Helpful and harmful training samples
* Evaluate **robustness** under input perturbations:

  * Gaussian noise
  * Gaussian blur
  * Brightness changes
* Use **Grad-CAM** as an additional explainability tool to verify that:

  * The model focuses on **biologically meaningful regions** (e.g., nucleus, cytoplasm), not background.

### 2.3 Final high-level conclusion

After all experiments:

* All models achieve **high accuracy** on clean test data.
* Robustness and sensitivity analysis show that:

  * Certain samples are much more sensitive (high Fisher scores).
  * Some training images strongly influence predictions (Influence Functions).
* **The selected model (stored as `best_wbc_model.keras`)** offers a strong trade-off between:

  * Accuracy
  * Robustness
  * Sensitivity profile
  * Computational efficiency
  * Explainability (with Grad-CAM)

---

<a name="dataset-details"></a>

# 3. Dataset Details

### 3.1 Dataset Source

We use the **White Blood Cells Dataset** from Kaggle:

* Author: **Masoud Nickparvar**
* Contains labeled WBC images for 5 classes.

### 3.2 Classes

The dataset contains five major WBC categories:

* **Neutrophil** – multi-lobed nucleus, most abundant type
* **Lymphocyte** – large, round nucleus, adaptive immunity
* **Monocyte** – kidney-shaped nucleus, largest WBC
* **Eosinophil** – bilobed nucleus, red/orange granules
* **Basophil** – rare, dark granules, involved in allergic responses

### 3.3 Dataset Size (approximate)

* Total images: ~14,514
* Training images: ~10,175
* Validation images: ~2,035
* Test images: ~4,339

(These numbers can be adjusted to match your exact splits.)

### 3.4 Preprocessing Steps

1. Convert all images to **RGB**.
2. Resize to:

   * `128 x 128` (for Custom CNN)
   * `224 x 224` (for VGG16, ResNet50)
3. Normalize pixel values to **[0, 1]**.
4. Encode labels as **one-hot vectors**.
5. Use **stratified splitting** for train/validation/test.

### 3.5 Data Augmentation

We apply data augmentation during training to improve generalization:

* Rotation (`±20°`)
* Width and height shifts (up to 20%)
* Shear transformation
* Zoom (up to 20%)
* Horizontal and vertical flips
* `fill_mode = 'nearest'` for padded pixels

These augmentations simulate realistic variations in microscopic capture conditions.

---

<a name="environment-setup"></a>

# 4. Environment Setup

### 4.1 Create a virtual environment

**Windows (PowerShell):**

```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.2 Install dependencies

Create a `requirements.txt` (example):

```txt
tensorflow>=2.8.0
numpy>=1.19.5
pandas>=1.3.0
matplotlib>=3.4.3
seaborn>=0.11.0
pillow>=8.3.0
scikit-learn>=1.0.2
tqdm>=4.62.0
kagglehub>=0.1.0
streamlit>=1.20.0
opencv-python
```

Then install:

```bash
pip install -r requirements.txt
```

### 4.3 Launch notebooks

All training and analysis can be run using Jupyter or Google Colab:

```bash
jupyter notebook
```

Open:

* `Code_WBC_Classification.ipynb` (or your training notebook)
* `Trustworthiness_Final.ipynb` (FIM, IF, robustness, Grad-CAM)

---

<a name="folder-structure"></a>

# 5. Folder Structure

An example folder structure for this project:

```bash
wbc-trustworthy-ai/
│
|
├── Trustworthiness_Final.ipynb          # End-to-end training of CNN + VGG16 + ResNet50  FIM, Influence Functions, robustness, Grad-CAM
│
├── app_streamlit.py                     # Streamlit UI for WBC classifier
│
├── models/
│   ├── best_wbc_model.keras             #  Custom CNN
│   ├── vgg16_wbc_model.keras            # VGG16-based model
│   └── resnet50_wbc_model.keras         # ResNet50-based model
|
|
└── README.md
```

You can adapt names to match your exact structure.

---

<a name="models-trained"></a>

# 6. Models Trained

We trained **three deep learning models**:

1. **Custom CNN (from scratch)**
2. **VGG16 (transfer learning)**
3. **ResNet50 (transfer learning)**

### 6.1 Custom CNN

* Input: `128 x 128 x 3`

* Architecture:

  * Block 1: Conv(64) → Conv(64) → BatchNorm → MaxPool → Dropout
  * Block 2: Conv(128) → Conv(128) → BatchNorm → MaxPool → Dropout
  * Block 3: Conv(256) → Conv(256) → BatchNorm → MaxPool → Dropout
  * Block 4: Conv(512) → Conv(512) → BatchNorm → MaxPool → Dropout
  * Dense: 1024 → 512 → Softmax(5)

* Saved as: `models/best_wbc_model.keras` (if you choose this as the main model)

### 6.2 VGG16 (Transfer Learning)

* Pretrained on ImageNet
* Convolutional base **frozen**
* Custom classification head:

  * Flatten → Dense(200, ReLU) → Dropout → Dense(5, Softmax)
* Input size: `224 x 224 x 3`
* Saved as: `models/vgg16_wbc_model.keras`

### 6.3 ResNet50 (Transfer Learning)

* Pretrained on ImageNet
* Convolutional layers frozen
* Custom classifier:

  * Flatten → Dense(150, ReLU) → Dense(5, Softmax)
* Input size: `224 x 224 x 3`
* Saved as: `models/resnet50_wbc_model.keras`

---

<a name="performance-metrics-accuracy--validation"></a>

# 7. Performance Metrics (Accuracy & Validation)

Each model was evaluated on:

* Training accuracy
* Validation accuracy
* Test accuracy

A typical summary (you can fill with your exact numbers):

| Model      | Train Accuracy | Validation Accuracy | Test Accuracy |
| ---------- | -------------- | ------------------- | ------------- |
| Custom CNN | ~94–96%        | ~94–96%             | ~95%          |
| VGG16      | High           | High                | High          |
| ResNet50   | High           | High                | High          |

Key points:

* All three models reach high accuracy on clean test data.
* Transfer learning models (VGG16, ResNet50) slightly outperform the custom CNN in some runs, but with higher computational cost.
* The selected deployment model is chosen based on **accuracy + robustness + explainability**, not just accuracy.

---

<a name="confusion-matrix--classification-metrics"></a>

# 8. Confusion Matrix & Classification Metrics

For the main model (e.g., Custom CNN), we compute:

* **Precision, Recall, F1-score** per class
* **Confusion matrix** over the five WBC classes

Example outputs:

```python
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
```
 confusion matrix figure like:


<img width="867" height="667" alt="Screenshot 2025-12-01 143143" src="https://github.com/user-attachments/assets/40c2a682-6fdc-4670-85f5-53a1f3a17292" />


Typical observations:

* High precision and recall for **Neutrophils** and **Lymphocytes**
* Slightly lower performance on **Monocytes** and **Eosinophils**, due to morphological similarities
* Basophils are rare → may show slightly lower recall due to class imbalance

---

<a name="robustness-evaluation-trustworthiness"></a>

# 9. Robustness Evaluation (Trustworthiness)

### 9.1 What we mean by robustness

A robust classifier should maintain good performance when inputs are perturbed in realistic ways:

* Microscope noise
* Slight blur/misfocus
* Changes in brightness/contrast

### 9.2 Perturbations used

We evaluate models under:

1. **Gaussian Noise** – simulates sensor noise
2. **Gaussian Blur** – simulates out-of-focus images
3. **Brightness Change** – simulates different illumination conditions

For each perturbation type and severity level, we:

* Apply the transformation to the test set
* Measure the test accuracy again
* Compare with clean test accuracy

### 9.3 Robustness results

This graph can show:

* X-axis: perturbation type / severity
* Y-axis: accuracy
* Curves: CNN, VGG16, ResNet50

Typical trends:

* All models lose some accuracy under heavy perturbations.
* Transfer learning models may be slightly more stable to blur/brightness.
* However, the selected model (e.g., Custom CNN) provides a good trade-off between robustness and computational cost.

---

<a name="sensitivity-evaluation-fisher-information-matrix"></a>

# 10. Sensitivity Evaluation (Fisher Information Matrix)

### 10.1 Idea

Fisher Information Matrix (FIM) measures **how sensitive the model is** to small changes in its inputs or parameters for a particular sample.

Here, we use a simplified scalar Fisher score per test sample:

Let:

* ( p ) = predicted probability vector
* ( y ) = one-hot ground truth vector

Compute gradient of loss:

[
g = p - y
]

We define the **Fisher score** as:

[
F(x) = | g |_2^2
]

### 10.2 Interpretation

* **Low Fisher score** → model is confident and stable for that sample
* **High Fisher score** → sample is close to a decision boundary, prediction is more fragile

### 10.3 FIM results

We compute Fisher scores for all test samples and plot their distribution:


<img width="735" height="387" alt="Screenshot 2025-12-01 115508" src="https://github.com/user-attachments/assets/6350a3e2-a822-4d4b-8a07-7dabff14a606" />


Typical observations:

* Most samples lie in a low-to-moderate Fisher range (stable).
* A small tail of high-Fisher samples indicates:

  * Ambiguous cells
  * Low contrast images
  * Borderline cases between two WBC types

These high-Fisher images are important for understanding where the model is least confident.

---

<a name="influence-functions-evaluation"></a>

# 11. Influence Functions Evaluation

### 11.1 Idea

Influence Functions estimate **how much each training point contributes** to a particular test prediction.

For our approximate implementation, we use:

[
\text{Influence}(i, j) \approx - (g_{train}[i] \cdot g_{test}[j])
]

where:

* ( g_{train}[i] ) – gradient of loss for a training sample
* ( g_{test}[j] ) – gradient of loss for a test sample

### 11.2 Interpretation

* **Positive Influence** → training sample *supports* the correct prediction
* **Negative Influence** → training sample *harms* the prediction (could be mislabeled or noisy)

### 11.3 Influence results

We compute influence scores for selected test samples and plot them:


<img width="662" height="410" alt="Screenshot 2025-12-01 122309" src="https://github.com/user-attachments/assets/990fa201-aa17-4022-8b8e-a7f61d96ff24" />

```

From this, we can:

* Identify the **most helpful training images** for a test prediction
* Identify the **most harmful training images** that push the model towards errors
* Inspect these cases manually to find:

  * Label noise
  * Very atypical WBC examples
  * Edge cases in the training data

This improves **dataset transparency** and supports trustworthiness.

---

<a name="explainability-evaluation-grad-cam"></a>

# 12. Explainability Evaluation (Grad-CAM)

### 12.1 Why Grad-CAM?

Grad-CAM helps answer:

> *“Where is the model looking when it decides this is a Neutrophil or Lymphocyte?”*

We want to see if the model focuses on:

* Nucleus shape
* Cytoplasm texture
* Granules
  rather than background noise.

### 12.2 Grad-CAM output examples

We generate Grad-CAM on:

* Correctly classified images
* Incorrectly classified images


Typical observations:

* For correct predictions, the heatmap is focused on:

  * Nuclear region
  * Cytoplasmic structure
* For incorrect predictions, the heatmap may be:

  * Spread out
  * Focused on irrelevant regions

Grad-CAM thus gives **visual evidence** of model reasoning and is a key part of explainable AI in this project.

---

<a name="why-the-selected-model-is-recommended"></a>

# 13. Why the Selected Model is Recommended

Combining all dimensions:

1. **Accuracy on clean test data**
2. **Robustness under noise/blur/brightness**
3. **Sensitivity profile (FIM)**
4. **Training data influence behavior (IF)**
5. **Explainability (Grad-CAM)**
6. **Computation and deployment cost**

We select the model saved as:

```text
models/best_wbc_model.keras
```

as the **recommended deployment model** for:

* The Streamlit application
* Further experiments and extensions

This model provides the best balance of:

* Strong performance
* Reasonable robustness
* Good explainability
* Practical runtime efficiency

---

<a name="reliability--limitations=""></a>

# 14. Reliability & Limitations

### 14.1 Reliability strengths

* High accuracy on the WBC test set
* Good robustness to realistic perturbations
* High-confidence predictions for most test samples (low-to-moderate Fisher scores)
* Clear Grad-CAM focus on biologically relevant regions
* Training sample analysis via Influence Functions helps detect problematic data points

### 14.2 Limitations

* The dataset is still relatively controlled; real hospital images might have additional variability.
* Basophils and other rare classes may still suffer from class imbalance.
* Influence Function implementation uses approximations, not full Hessian inversion.
* Robustness to strong adversarial attacks (e.g., PGD with many steps) is not fully addressed.

These limitations form natural directions for future work.

---

<a name="download-final-models"></a>

# 15. Download Final Models

You can store trained models as:

```bash
models/
├── best_wbc_model.keras
├── vgg16_wbc_model.keras
└── resnet50_wbc_model.keras
```

Make sure the Streamlit app points to the correct model path (e.g., `best_wbc_model.keras`).

---

<a name="streamlit-deployment"></a>

# 16. Streamlit Deployment

The Streamlit app (`app_streamlit.py`) provides:

* **Image upload** component
* **Model prediction with confidence**
* **Class probability bar chart**
* Optional **Grad-CAM heatmap** (if enabled)

---

<a name="run-the-application"></a>

# 17. Run the Application

From your project root:

```bash
streamlit run app_streamlit.py
```

Then open:

```text
http://localhost:8501
```

Workflow:

1. Upload a WBC image (`.jpg`, `.png`).
2. The app preprocesses and feeds it into the selected model.
3. The app shows:

   * Predicted class
   * Confidence score
   * Probability distribution over all 5 classes
   * (Optional) Grad-CAM heatmap to visualize attention (if implemented in the app).

---

<a name="workflow-summary-step-by-step"></a>

# 18. Workflow Summary (Step-by-Step)

1. **Download dataset** (via KaggleHub or manually).
2. **Preprocess images** (resize, normalize, encode labels).
3. **Train models**:

   * Custom CNN (128×128)
   * VGG16 (224×224)
   * ResNet50 (224×224)
4. **Evaluate models** on clean test set (accuracy, confusion matrix, metrics).
5. **Run robustness experiments** (noise, blur, brightness) and record accuracy drop.
6. **Compute Fisher scores** for test samples and analyze sensitivity distribution.
7. **Compute Influence Functions** to estimate impact of training samples.
8. **Generate Grad-CAM heatmaps** for correct and incorrect predictions.
9. **Compare models** based on performance + trustworthiness.
10. **Select final model** (`best_wbc_model.keras`).
11. **Deploy Streamlit app** for interactive WBC classification.

---

<a name="quick-commands-summary"></a>

# 19. Quick Commands Summary

```bash
# Activate environment
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Run Streamlit app
streamlit run app_streamlit.py
```

---

<a name="troubleshooting"></a>

# 20. Troubleshooting

| Issue                               | Possible Cause                        | Fix                                           |
| ----------------------------------- | ------------------------------------- | --------------------------------------------- |
| `ImportError: tensorflow`           | TensorFlow not installed              | `pip install tensorflow>=2.8.0`               |
| `DLL load failed` on Windows        | TensorFlow vs Python mismatch         | Use Python 3.9/3.10 and TF 2.8+               |
| Model file not found                | Wrong path to `.keras` model file     | Check `models/` folder and update path in app |
| Strange predictions / wrong labels  | Class order mismatch                  | Ensure `class_names` matches training order   |
| Streamlit error `no cache_resource` | Older Streamlit version               | Upgrade via `pip install --upgrade streamlit` |
| GPU RAM issues                      | Model too big or batch size too large | Reduce batch size or train smaller model      |

---

<a name="future-enhancements"></a>

# 21. Future Enhancements

* Add **adversarial attack** robustness (FGSM, PGD) explicitly for WBC images.
* Use **Grad-CAM++** for sharper explanations.
* Perform **cross-dataset evaluation** with other WBC datasets or real hospital images.
* Incorporate **uncertainty estimation** (e.g., MC Dropout).
* Explore **Vision Transformers (ViT)** for WBC classification.

---

Got it. Here is the updated **Section 22. References & Code Attribution** with your source code links added as a clearly separated subsection.

You can replace just this section in your README:

---

<a name="references--code-attribution"></a>

# 22. References & Code Attribution

### Dataset

* Masoud Nickparvar, *White Blood Cells Dataset* (Kaggle)

### Concepts and Methods

* Fisher Information and sensitivity-based analysis (standard statistical methodology)
* Influence Functions: Koh & Liang, “Understanding Black-box Predictions via Influence Functions”
* Grad-CAM: Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization”
* CNN, VGG16, ResNet50 architectures: Keras / TensorFlow model zoo and original papers

### External Code References / Source Code Links

Parts of the data loading, CNN design patterns, transfer learning setup, and training utilities were inspired or adapted (with modifications) from the following public notebooks and repositories:

* Link 1 – Blood Cell CNN (Kaggle):
  [https://www.kaggle.com/code/eliatoma/blood-cell-cnn](https://www.kaggle.com/code/eliatoma/blood-cell-cnn)

* Link 2 – WBC Classification Notebook (Kaggle):
  [https://www.kaggle.com/code/amrawni/notebookf80d2791a8](https://www.kaggle.com/code/amrawni/notebookf80d2791a8)

* Link 3 – White Blood Cell Classification with Two Pretrained Models (Kaggle):
  [https://www.kaggle.com/code/abdelrahmanammar/wblood-cell-classification-with-2-pretrained-model](https://www.kaggle.com/code/abdelrahmanammar/wblood-cell-classification-with-2-pretrained-model)

* Link 4 – White Blood Cell Classification (GitHub):
  [https://github.com/M-Husnain-Ali/White-Blood-Cell-Classification/blob/main/code.ipynb](https://github.com/M-Husnain-Ali/White-Blood-Cell-Classification/blob/main/code.ipynb)

These sources were used as references for:

* General CNN training structure
* Transfer learning setups for VGG16 and ResNet-style models
* Handling of WBC datasets and preprocessing flows

### Code Developed for This Project

* The integrated pipeline for **Custom CNN + VGG16 + ResNet50 comparison**,
* The **trustworthiness components** (FIM, Influence Functions, robustness experiments),
* The **Grad-CAM adaptation for this WBC classifier**, and
* The **Streamlit user interface and overall integration**


<a name="final-summary"></a>

# 23. Final Summary

This project implements a **White Blood Cell classification system** with:

* High accuracy using multiple deep learning models
* A strong focus on **trustworthiness**, through:

  * Robustness evaluation under perturbations
  * Sensitivity analysis via **Fisher Information**
  * Influence analysis via **Influence Functions**
  * Visual explanations via **Grad-CAM**
* A practical **Streamlit-based user interface** for interactive usage.

The selected model (`best_wbc_model.keras`) delivers a well-balanced combination of:

* Performance
* Robustness
* Interpretability
* Practical deployability

making it a strong candidate for future integration into clinical decision support workflows.

---

<a name="team-members--contributions"></a>

# 24. Team Members & Contributions

This project was completed by a team of **three members**:

* **Srujan Alumula**

  * Implemented **trustworthiness analysis**:

    * Fisher Information Matrix (FIM)
    * Influence Functions (IF)
    * Robustness experiments (noise, blur, brightness)
  * Integrated **Grad-CAM** explainability for WBC images
  * Helped with result interpretation and documentation (trustworthiness sections)

* **Tarun**

  * Implemented and fine-tuned **VGG16** and **ResNet50** transfer learning models
  * Ran baseline **accuracy evaluations**, confusion matrices, and metric comparisons
  * Contributed to hyperparameter tuning and model selection logic

* **Sireesha**

  * Implemented and trained the **Custom CNN** model
  * Developed the **Streamlit user interface** (`app_streamlit.py`)
  * Integrated saved models into the app and prepared usage instructions



---

