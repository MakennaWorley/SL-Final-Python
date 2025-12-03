import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# --- Configuration ---
COHORT_FILE = "../data/3195663216_msprime_sim_cohort.csv"
SEED = 42


@st.cache_data
def load_data(file_path):
    try:
        # NOTE: The original notebook used a path that suggests the file is in a parent directory:
        # cohort_path = "../data/3195663216_msprime_sim_cohort.csv"
        # Adjust the path below if your file is in a different location.
        cohort = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(
            f"Error: Data file not found at {file_path}. Please ensure '{COHORT_FILE}' is in the correct directory.")
        st.stop()
    return cohort


def preprocess_data(cohort):
    # --- Feature Engineering and Selection ---
    pcs_features = [col for col in cohort.columns if col.startswith("PC")]

    # Dropping all targets/IDs to define the feature space X:
    X = cohort.drop(columns=["individual_id", "quant_trait", "disease_status", "disease_prob"], errors='ignore')

    # Re-define features based on the notebook's final feature sets
    numeric_predictors = ["age", "env_index", "polygenic_score"] + [f for f in pcs_features if f in X.columns]

    # Define targets
    y_reg = cohort["quant_trait"]
    y_clf = cohort["disease_status"]

    # --- Train-Test Split (SEED=42 used in notebook) ---
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.3, random_state=SEED
    )

    # --- Standardization on Numerical Predictors ---
    scaler = StandardScaler()
    X_train_scaled_num = scaler.fit_transform(X_train[numeric_predictors])
    X_test_scaled_num = scaler.transform(X_test[numeric_predictors])

    X_train_scaled = pd.DataFrame(X_train_scaled_num, columns=numeric_predictors, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_num, columns=numeric_predictors, index=X_test.index)

    # Add back the categorical 'sex' feature
    if "sex" in X_train.columns:
        X_train_scaled["sex"] = X_train["sex"]
        X_test_scaled["sex"] = X_test["sex"]

    return X_train_scaled, X_test_scaled, y_clf_train, y_clf_test, y_reg_train, y_reg_test


# --- Model Training and Evaluation Utilities ---

def train_and_evaluate_clf(model_class, X_train, y_train, X_test, y_test, name, **params):
    # FIX: Only pass random_state if the model explicitly supports it (Fix for LDA error)
    kwargs = params.copy()
    if 'random_state' in model_class().get_params():
        kwargs['random_state'] = SEED

    model = model_class(**kwargs)

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Calculate probabilities/decision function
    if hasattr(model, 'predict_proba'):
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_prob_train = model.predict_proba(X_train)[:, 1]
    else:
        y_prob_test = model.decision_function(X_test)
        y_prob_train = model.decision_function(X_train)

    # Calculate metrics
    test_acc = accuracy_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_prob_test)
    test_ap = average_precision_score(y_test, y_prob_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    train_auc = roc_auc_score(y_train, y_prob_train)

    # Generate graph
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob_test)
    no_skill = y_test.mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: ROC Curve ---
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {test_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC Curve: {name}')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Precision-Recall Curve ---
    axes[1].plot(recall, precision, color='green', lw=2, label=f'Avg Precision = {test_ap:.3f}')
    axes[1].plot([0, 1], [no_skill, no_skill], color='navy', linestyle='--', label='No Skill')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'PR Curve: {name}')
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Return model, test metrics, train metrics, and figure
    return model, test_acc, test_auc, train_acc, train_auc, fig


def train_and_evaluate_reg(model_class, X_train, y_train, X_test, y_test, name, **params):
    # Ensure RidgeCV (Shrinkage) uses the optimal alpha found via Cross-Validation
    if model_class == RidgeCV:
        model = model_class(alphas=np.logspace(-4, 4, 200), **params)
    else:
        model = model_class(**params)

    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # Generate graph
    residuals = y_test - y_pred_test
    r2 = r2_test

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Predicted vs True
    axs[0, 0].scatter(y_test, y_pred_test, alpha=0.5)
    slope, intercept = np.polyfit(y_test, y_pred_test, 1)
    axs[0, 0].plot(y_test, slope * y_test + intercept, color="blue", label="Regression Line")
    axs[0, 0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red", linestyle="--", label="Identity Line"
    )
    axs[0, 0].set_xlabel("True Values")
    axs[0, 0].set_ylabel("Predicted Values")
    axs[0, 0].set_title(f"Predicted vs True ({name})\n$R^2 = {r2:.3f}$")
    axs[0, 0].legend()

    # 2) Residual Histogram
    sns.histplot(residuals, kde=True, bins=20, ax=axs[0, 1])
    axs[0, 1].set_title("Residual Distribution")
    axs[0, 1].set_xlabel("Residual")
    axs[0, 1].set_ylabel("Count")

    # 3) Residuals vs Fitted
    axs[1, 0].scatter(y_pred_test, residuals, alpha=0.5)
    axs[1, 0].axhline(0, color="red", linestyle="--")
    axs[1, 0].set_xlabel("Fitted Values")
    axs[1, 0].set_ylabel("Residuals")
    axs[1, 0].set_title("Residuals vs Fitted")

    # 4) QQ Plot
    stats.probplot(residuals, dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title("QQ Plot")

    plt.tight_layout()

    # Return model, test metrics, train metrics, and figure
    return model, rmse_test, r2_test, rmse_train, r2_train, fig


# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide")
    st.title("Jupyter Notebook to Streamlit App: Predictive Modeling")
    st.markdown(
        "This application implements the **top three** Classification and **top three** Regression models identified in the original Jupyter Notebook analysis.")
    st.markdown("---")

    # Load and Preprocess Data
    cohort = load_data(COHORT_FILE)
    if cohort.empty:
        return

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = preprocess_data(cohort)

    # --- Define Feature Subsets ---
    # Full set of predictors (age, env_index, polygenic_score, sex, PC1, PC2)
    features_full_pca = [col for col in X_train.columns if
                         col not in ["individual_id", "quant_trait", "disease_status", "disease_prob"]]
    # Full set excluding PCs (age, env_index, polygenic_score, sex)
    features_full_no_pca = [f for f in features_full_pca if not f.startswith('PC')]

    # Classification: Top 3 Valid Models (Logistic Regression, LDA, SVM (Linear))
    clf_models = [
        ("Logistic Regression", LogisticRegression, features_full_pca, {'solver': 'liblinear'}),
        ("LDA", LDA, features_full_pca, {}),
        ("SVM (Linear)", SVC, features_full_pca, {'kernel': 'linear', 'probability': True, 'cache_size': 1000}),
    ]

    # Regression: Top 3 Models (LR Full + PCA, Ridge (Full - No PCA), LR Full (PRS + Covariates - No PCA))
    reg_models = [
        ("LR Full + PCA (Best Overall)", LinearRegression, features_full_pca, {}),
        ("Ridge (Full - No PCA)", RidgeCV, features_full_no_pca, {}),
        ("LR Full (PRS + Covariates - No PCA)", LinearRegression, features_full_no_pca, {}),
    ]

    # --- Run Models and Display Results ---
    tab1, tab2 = st.tabs(["Classification (Disease Status)", "Regression (Quant Trait)"])

    # Store results for sidebar summary
    clf_results_display = []
    reg_results_display = []

    # =========================================================================
    # TAB 1: CLASSIFICATION
    # =========================================================================
    with tab1:
        st.header("Classification: Predicting `disease_status`")
        st.markdown(
            "The top 3 *valid* models (based on AUC and balanced overfitting in the notebook) are displayed below, along with ROC and Precision-Recall Curves.")

        for name, model_class, features, params in clf_models:
            clf_X_train = X_train[features]
            clf_X_test = X_test[features]

            # UPDATED CALL: capturing train metrics and the figure
            model, test_acc, test_auc, train_acc, train_auc, fig = train_and_evaluate_clf(
                model_class, clf_X_train, y_clf_train, clf_X_test, y_clf_test, name, **params
            )

            # Display model details
            st.subheader(f"{name}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test Accuracy", f"{test_acc:.4f}")
            col2.metric("Test AUC", f"{test_auc:.4f}")
            col3.metric("Train AUC", f"{train_auc:.4f}")

            # Overfitting Warning based on AUC difference
            if (train_auc - test_auc) > 0.05:
                col4.warning("⚠️ Overfitting Warning: Train AUC much higher than Test AUC.")
            else:
                col4.info("✅ Model seems balanced")

            # Display Graphs
            st.pyplot(fig, clear_figure=True)

            # Display Feature Coefficients
            if hasattr(model, 'coef_'):
                coef_df = pd.DataFrame(model.coef_[0], index=features, columns=['Coefficient'])
                st.write("**Feature Coefficients** (Impact on log-odds):")
                st.dataframe(coef_df.sort_values(by='Coefficient', ascending=False), use_container_width=True)

            clf_results_display.append(
                {'Model': name, 'Test Accuracy': f"{test_acc:.4f}", 'Test AUC': f"{test_auc:.4f}"})
            st.markdown("---")

    # =========================================================================
    # TAB 2: REGRESSION
    # =========================================================================
    with tab2:
        st.header("Regression: Predicting `quant_trait`")
        st.markdown(
            "The top 3 Regression models (based on $R^2$/RMSE in the notebook) are displayed below, along with diagnostic plots.")

        for name, model_class, features, params in reg_models:
            reg_X_train = X_train[features]
            reg_X_test = X_test[features]

            # UPDATED CALL: capturing train metrics and the figure
            model, test_rmse, test_r2, train_rmse, train_r2, fig = train_and_evaluate_reg(
                model_class, reg_X_train, y_reg_train, reg_X_test, y_reg_test, name, **params
            )

            # Display model details
            st.subheader(f"{name}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test R²", f"{test_r2:.4f}")
            col2.metric("Test RMSE", f"{test_rmse:.4f}")
            col3.metric("Train R²", f"{train_r2:.4f}")

            # Overfitting Warning based on R² difference
            if (train_r2 - test_r2) > 0.05:
                col4.warning("⚠️ Overfitting Warning: Train R² much higher than Test R².")
            else:
                col4.info("✅ Model seems reasonably balanced")

            # Display Graphs
            st.pyplot(fig, clear_figure=True)

            if hasattr(model, 'coef_'):
                # Handle RidgeCV coefficient access
                coef_val = model.coef_ if not isinstance(model, RidgeCV) else model.coef_

                coef_df = pd.DataFrame(coef_val, index=features, columns=['Coefficient'])
                st.write("**Feature Coefficients** (Impact on `quant_trait`):")
                st.dataframe(coef_df.sort_values(by='Coefficient', ascending=False), use_container_width=True)

            reg_results_display.append({'Model': name, 'Test R²': f"{test_r2:.4f}", 'Test RMSE': f"{test_rmse:.4f}"})
            st.markdown("---")

    # Display Sidebars with Summary Tables
    with st.sidebar:
        st.title("Summary of Top Models")
        st.subheader("Classification")
        st.dataframe(pd.DataFrame(clf_results_display), hide_index=True, use_container_width=True)
        st.subheader("Regression")
        st.dataframe(pd.DataFrame(reg_results_display), hide_index=True, use_container_width=True)


if __name__ == '__main__':
    main()