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
**Classification task** using the binary variable `disease_status`. This task demonstrates familiarity with statistical learning classification methods (logistic regression, LDA, QDA, KNN, SVM), but the disease phenotype is intentionally noisy in the simulation, so the regression task is the scientifically meaningful component.

### ✔ Regression
**Linear models, subset selection methods, and shrinkage techniques** (ridge, lasso, elastic net) recover the **true genetic architecture** of a simulated polygenic quantitative trait. Because the dataset includes the *true causal effect sizes*, this analysis enables direct comparison between estimated and real underlying model coefficients.

---

## 🧬 Dataset Description

The simulation generates two CSV files:

### **Cohort-level data**
- `quant_trait` — continuous quantitative phenotype  
- `polygenic_score` — aggregate genetic risk score  
- `env_index` — environmental exposure  
- `sex` — binary categorical  
- `age` — numerical  
- `disease_status` — binary response for classification  
- `PC1`, `PC2` — simulated population structure (neutral)  

### **Variant-level data**
- `beta` — true SNP effect sizes  
- `is_causal` — indicator for causal variants  
- Only ~5% of variants are causal  

This structure allows evaluation of:
- Prediction accuracy  
- Coefficient recovery vs true β  
- Effect of population structure  
- Bias/variance performance under shrinkage  

---

## 🎯 Research Questions

### **Classification**
> **How accurately can disease status be predicted from polygenic and environmental predictors?**

This task demonstrates:
- Logistic regression  
- LDA / QDA  
- KNN  
- SVM  
- ROC curves, AUC, confusion matrix  

The classification model is less meaningful biologically due to the high stochasticity in the binary disease simulation.

### **Regression**
> **How well do linear, subset-selection, and shrinkage models recover the true genetic architecture of a simulated polygenic quantitative trait?**

Sub-questions:
1. How much variance is explained by PRS vs environmental factors?  
2. Which model yields the best predictive performance (RMSE, R²)?  
3. Do shrinkage methods improve coefficient stability?  
4. How closely do estimated coefficients match the true simulation parameters?  
5. Do PCs from neutral structure influence prediction?

---

## 📊 Methods

### **Classification Models**
- Logistic Regression  
- Linear Discriminant Analysis (LDA)  
- Quadratic Discriminant Analysis (QDA)  
- KNN (k = 11)  
- SVM with RBF kernel  

All classification models are evaluated using:
- Accuracy  
- ROC AUC  
- Confusion Matrix  
- ROC Curves  

### **Regression Models**
- Simple Linear Regression (`quant_trait ~ PRS`)
- Multiple Linear Regression (`PRS + sex + age + env_index`)
- Linear Regression with PCs (`+ PC1 + PC2`)
- Forward & Backward Stepwise Selection (AIC/BIC)
- **Ridge Regression**
- **Lasso Regression**
- **Elastic Net**
- **Bootstrap Coefficient Intervals (n=500)**

---

## 🧪 Evaluation Metrics

### **Classification**
- Accuracy  
- AUC  
- ROC curve  
- Confusion matrix  

### **Regression**
- RMSE (train/test)  
- R² (train/test)  
- Cross-validation RMSE  
- Coefficient stability (bootstrap)  
- Comparison to true β values  
- Shrinkage paths

---

## 📈 Key Results

### Classification
Best models:
- Logistic Regression: accuracy = **0.686**, AUC = **0.758**
- LDA: accuracy = **0.686**, AUC = **0.758**

Moderate performance due to:
- Noisy disease simulation  
- Weak environmental effect  
- Bernoulli sampling randomness  

KNN performs worst; SVM is decent but doesn’t beat linear methods.

### Regression
- Full linear model achieves **RMSE ≈ 0.644** and **R² ≈ 0.566**
- Subset selection consistently chooses:  
  `['polygenic_score', 'env_index', 'sex']`
- Shrinkage models yield nearly identical performance  
- Coefficients align with true architecture hierarchy:
  - PRS (strongest)
  - Environment (moderate)
  - Sex (small)
  - Age (very small)
- Bootstrap confirms stability of PRS and env effects  
- PCs do **not** improve prediction (expected due to neutral simulation)

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
|   ├── analysis.ipynb              # Main Jupyter analysis notebook
│   └── exploratory.ipynb           # EDA and initial exploration
│
│
└── README.md
```

---

## 📊 **Methods & Models Used**

### **Baseline Linear Models**
- Simple Linear Regression (`quant_trait ~ PRS`)
- Multiple Linear Regression (`PRS + sex + age + env_index`)
- Linear Model with Population Structure (`+ PC1 + PC2`)

### **Subset Selection**
- Forward stepwise
- Backward stepwise
- Best subset (if available)

### **Shrinkage / Regularization**
- Ridge Regression
- Lasso
- Elastic Net  
(using cross-validation to choose penalties)

### **Statistical Tools**
- Cross-validation  
- Bootstrap coefficient intervals  
- PCA (for optional population structure)

---

## 🧪 **Evaluation Metrics**

### For prediction performance:
- **Test RMSE**
- **Test R²**
- **Cross-validation RMSE**

### For coefficient analysis:
- Shrinkage coefficient paths  
- Coefficient stability across bootstrap samples  
- Correlation with **true** betas  
- MSE between estimated β and true β  

---

## 📈 **Figures & Visualizations**

The project generates:

- Histogram of the quantitative trait  
- Trait vs PRS scatterplots  
- PCA scree plot + PC1/PC2 ancestry plot  
- Shrinkage coefficient paths for ridge/lasso  
- Bootstrap coefficient distributions  
- Bar charts of model RMSE / R² comparison  
- True vs estimated effect size plots  

All figures are saved under the `figures/` directory.

---

## 🚀 **How to Run the Project**

### ⚙️ **Installation (Conda)**

#### 1️⃣ **Clone the repository**
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

#### 2️⃣ **Create and activate the environment**
```bash
conda env create -f environment.yml
conda activate data370
```

#### 3️⃣ **Run the analysis notebook**
Open:

```
notebooks/analysis.ipynb
```

This provides an interactive comparison of model performance.
