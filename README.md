# Statistical Learning Final Project  
### **Evaluating Linear Models & Shrinkage Methods for Predicting a Simulated Polygenic Trait**

**Author:** Makenna Worley  
**Course:** Statistical Learning (Fall 2025)  
**Dataset:** Generated data using [**make_msprime_dataset.py**](https://github.com/MakennaWorley/Capstone-Playground-Python/blob/main/README.md?plain=1) from my Capstone-Playground-Python Repo with seed: 3195663216

---

## 📌 **Project Overview**

This project uses a fully simulated genetic dataset generated with [**msprime**](https://tskit.dev/msprime/docs/stable/intro.html) to investigate how accurately different linear modeling strategies recover the underlying genetic architecture of a polygenic quantitative trait. The dataset includes a polygenic score derived from true causal variants, environmental covariates, demographic information, and optional population structure (PCs), allowing precise control over the ground truth. By comparing simple linear models, subset selection procedures, and shrinkage methods (ridge, lasso, elastic net), this analysis quantifies how much of the trait’s variance is attributable to genetic vs environmental factors and evaluates the stability and interpretability of coefficient estimates. Because the simulation provides the true causal effect sizes (β) used to construct the polygenic score, the project can directly assess how closely each model recovers the real underlying signal.

This work is completed as part of a Statistical Learning course and focuses on linear regression, subset selection, and shrinkage methods within a controlled genetic simulation framework.

The dataset contains:

- A quantitative phenotype (`quant_trait`)
- A polygenic score (`polygenic_score`)
- Environmental covariates (`env_index`)
- Demographic factors (`sex`, `age`)
- Optional population structure (PC1, PC2)
- A separate variant file with **true effect sizes** (`beta`) and causal status (`is_causal`)

This setup allows us to directly test which statistical models:

- Predict phenotype best  
- Capture genetic vs environmental contributions  
- Produce the most stable and interpretable coefficients  
- Most closely match the **true** simulated effect sizes  

This project serves as a **stepping stone** for my capstone on probabilistic ancestral genotype inference.

---

## 🎯 **Research Question**

> **How well do linear models, subset selection methods, and shrinkage methods recover the true genetic architecture of a polygenic quantitative trait?**

Sub-questions include:

1. How much variance in the trait is explained by PRS alone vs environment?  
2. Do shrinkage methods (ridge, lasso, elastic net) produce more stable and accurate coefficient estimates?  
3. How do coefficient estimates compare to the true underlying effect sizes?  
4. Does adding population structure (PCs) improve predictive performance?

---

## 📂 **Repository Structure**

```
project-root/
│
├── data/
│   ├── msprime_sim_cohort.csv
│   ├── msprime_effect_sizes.csv
│   └── (optional) documentation.txt
│
├── notebooks/
│   ├── analysis.ipynb              # Main Jupyter analysis notebook
│   └── exploratory.ipynb           # EDA and initial exploration
│
├── streamlit_app/
│   ├── app.py                      # Streamlit visualization interface
│   └── utils.py
│
├── src/
│   ├── data_loading.py
│   ├── models_linear.py
│   ├── models_shrinkage.py
│   ├── subset_selection.py
│   ├── evaluation.py
│   ├── pca_analysis.py
│   └── plotting.py
│
└── README.md                      # (this file)
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
conda activate data370_final
```

#### 3️⃣ **Run the analysis notebook**
Open:

```
notebooks/analysis.ipynb
```

#### 4️⃣ **(Optional) Launch the Streamlit visualization**
```bash
cd streamlit_app
streamlit run app.py
```

This provides an interactive comparison of model performance.
