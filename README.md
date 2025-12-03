# Statistical Learning Final Project  
### **Evaluating Classification, Linear, and Shrinkage Models for Predicting a Simulated Disease Status and Polygenic Trait Within A Population**

**Author:** Makenna Worley  
**Course:** Statistical Learning (Fall 2025)  
**Dataset:** Generated using `make_msprime_dataset.py` with seed 3195663216 
**Tools:** Python, scikit-learn, pandas, matplotlib, seaborn

---

## 📌 Project Overview

This project uses a fully simulated genetic dataset generated via **msprime** to evaluate both **classification** and **regression** methods within a controlled, biologically realistic setting.

### ✔ Classification

The **classification task** uses the binary variable `disease_status`. This task demonstrates familiarity with statistical learning classification methods (Logistic Regression, LDA, QDA, KNN, SVM), but the disease phenotype in the simulation is intentionally noisy, making the regression task the scientifically meaningful component.

### ✔ Regression

**Linear models, subset selection methods, and shrinkage techniques** (ridge, lasso, elastic net) are used to recover the **true genetic architecture** of a simulated polygenic `quant_trait`. Because the dataset includes the *true causal effect sizes*, this analysis enables a direct comparison between estimated and real underlying model coefficients.

---

## 🧬 Dataset Description

The simulation generates two CSV files:

### **Cohort-level data**
* `quant_trait` — continuous quantitative phenotype
* `polygenic_score` — standardized polygenic risk for the trait
* `disease_status` — binary outcome (0/1)
* `age`, `sex`, `env_index` — demographic and environmental covariates
* `PC1`, `PC2` — principal components capturing population structure

---

## 🔑 **Key Findings**

### 🦠 **Classification (Predicting Disease Status)**

The most robust classification model was **Logistic Regression**, which achieved the highest balance of predictive power and stability.

* **Top Performance (AUC):** The **Logistic Regression model** achieved the highest **AUC ($\approx 0.758$)** and **Average Precision ($\approx 0.751$)** on the test set.
* **Accuracy:** The best model's overall accuracy was **$\approx 68.7\%$**.
* **Overfitting:** Unlike the Linear Regression, several non-linear models (KNN, Decision Tree, Random Forest) showed signs of **significant overfitting**.
* **Feature Importance (Classification):** The **Polygenic Score** remained the single most important predictor across all stable classification models.

### 🎯 **Regression (Predicting Quantitative Trait)**

The most robust and predictive regression model was **Linear Regression using Full Features + PCA** ($R^2 \approx 0.567$).

* **Optimal Performance** The highest predictive power achieved by any model was an **$R^2 \approx 0.567$** on the test set, with a low $\text{RMSE} \approx 0.644$.
* **Best Model:** The **Linear Regression (LR) Full + PCA model** was selected as the top model due to its high accuracy, robust stability, and its use of orthogonal (uncorrelated) principal components as predictors.
* **Feature Importance:** The most significant predictors for the `quant_trait` were the **Polygenic Score** (coefficient $\approx 0.657$) and the **Environmental Index** (coefficient $\approx 0.353$).
* **Shrinkage Models:** Ridge, Lasso, and ElasticNet models converged on a final model with nearly the **exact same performance** as the standard Linear Regression, indicating that the original OLS model was already well-conditioned.
* **Model Validation:** All top linear and shrinkage models showed **low error and strong stability**, successfully meeting the assumptions of randomly scattered, normally distributed errors.
* **Poor Performers:** Regression Tree models were **disqualified due to severe overfitting** (Test $R^2$ in the $0.35$ to $0.39$ range), confirming the underlying relationship between predictors and the Quantitative Trait is fundamentally **linear**.

---

## 🧪 **Evaluation Metrics**

### For classification performance:

* **Test Accuracy**
* **Test AUC** (Area Under the ROC Curve - primary metric)
* **Test Average Precision**

### For regression performance:

* **Test $\text{RMSE}$**
* **Test $R^2$**
* **Overfitting Gap** (Difference between Train $R^2$ and Test $R^2$)

---

## 🎯 Research Questions

### **Classification**
> **How accurately can disease status be predicted from the features?**

#### **Classification Models**
- Logistic Regression  
- Linear Discriminant Analysis (LDA)  
- Quadratic Discriminant Analysis (QDA)  
- KNN (k = 11)  
- SVM with RBF kernel 

#### **Classification Evaluation Metrics**
- Accuracy
- ROC curves
- AUC
- Confusion matrix  

The classification model is less meaningful biologically due to the high stochasticity in the binary disease simulation.

### **Regression**
> **How well do linear, subset-selection, and shrinkage models recover the true genetic architecture of a simulated polygenic quantitative trait?**

#### **Regression Models**
- Simple Linear Regression (`quant_trait ~ PRS`)
- Multiple Linear Regression (`PRS + sex + age + env_index`)
- Linear Regression with PCs (`+ PC1 + PC2`)
- Forward & Backward Stepwise Selection (AIC/BIC)
- **Ridge Regression**
- **Lasso Regression**
- **Elastic Net**
- **Bootstrap Coefficient Intervals (n=500)**

#### **Regression Evaluation Metrics**
- RMSE (train/test)  
- R² (train/test)  
- Cross-validation RMSE  
- Coefficient stability (bootstrap)  
- Comparison to true β values  
- Shrinkage paths

### **Sub-questions:**
1. How much variance is explained by PRS vs environmental factors?  
2. Which model yields the best predictive performance (RMSE, R²)?  
3. Do shrinkage methods improve coefficient stability?  
4. How closely do estimated coefficients match the true simulation parameters?  
5. Do PCs from neutral structure influence prediction?

---

## 📂 Repository Structure

```
project-root/
│
├── data/
│   ├── msprime_sim_cohort.csv
│   └── msprime_effect_sizes.csv
│
├── notebooks/
│   ├── final.ipynb                 # Main Jupyter analysis notebook
│   ├── final.html                  # HTML export of final.ipynb
│   ├── analysis.ipynb              # Playground for my analysis
│   └── exploratory.ipynb           # EDA and initial exploration
│
├── streamlit/
│   ├── app.py                      # Streamlit visualization interface
│   └── requirements.txt            # Requirements for the Streamlit
│
└── README.md
```

---

## 🚀 **How to Run the Project**

### ⚙️ **Installation (Conda)**

#### 1️⃣ **Clone the repository**
```bash
git clone https://github.com/MakennaWorley/SL-Final-Python.git
cd SL-Final-Python
```

#### 2️⃣ **Create and activate the environment**
```bash
conda env create -f environment.yml
conda activate data370
```

#### 3️⃣ **Run the final notebook**
Open:

```
notebooks/final.ipynb
```

#### 4️⃣ **(Optional) Launch the Streamlit visualization**
```bash
cd streamlit
streamlit run app.py
```

#### 4️⃣ **(Optional) Launch the Streamlit visualization**
```bash
cd streamlit
streamlit run app.py
```

This provides an interactive comparison of model performance.
