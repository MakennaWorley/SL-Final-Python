import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import stats
from scipy.stats import zscore
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
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Configuration ---
COHORT_FILE = "https://raw.githubusercontent.com/MakennaWorley/SL-Final-Python/main/data/3195663216_msprime_sim_cohort.csv"
SEED = 42


@st.cache_data
def load_data(file_path):
    try:
        cohort = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(
            f"Error: Data file not found at {file_path}. Please ensure '{COHORT_FILE}' is in the correct directory.")
        st.stop()
    return cohort


def calculate_vif(df, features):
    """Calculates VIF for a DataFrame using selected features."""
    X = df[features].select_dtypes(include=np.number)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data.sort_values(by="VIF", ascending=False)


def display_eda(cohort):
    st.header("Cohort Data Overview & Summary Statistics")

    # --- Initial Checks ---
    st.subheader("Initial Data Sample (First 5 Rows)")
    st.dataframe(cohort.head(), use_container_width=True)

    st.subheader("Summary Statistics")
    # Display the transposed describe table for easier reading
    st.dataframe(cohort.describe(include="all").T, use_container_width=True)
    st.markdown("---")

    # --- Numerical Variable Distributions ---
    st.header("Numerical Feature Distributions")

    # Distributions by Feature
    st.subheader("Age")
    fig_age, axes_age = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(cohort["age"], kde=True, bins=30, ax=axes_age[0])
    axes_age[0].set_title("Distribution of Age")
    sns.boxplot(x=cohort["age"], ax=axes_age[1])
    axes_age[1].set_title("Age Distribution")
    axes_age[1].set_xlabel("Age")
    st.pyplot(fig_age, clear_figure=True)
    st.dataframe(cohort["age"].describe())
    st.markdown("---")

    st.subheader("Env Index (`env_index`)")
    fig_env, ax_env = plt.subplots(figsize=(7, 4))
    sns.histplot(cohort["env_index"], kde=True, bins=30, ax=ax_env)
    ax_env.set_title("Distribution of Env Index")
    st.pyplot(fig_env, clear_figure=True)
    plt.show()
    st.dataframe(cohort["env_index"].describe())
    st.markdown("---")

    st.subheader("Polygenic Score (`polygenic_score`)")
    fig_prs, ax_prs = plt.subplots(figsize=(7, 4))
    sns.histplot(cohort["polygenic_score"], kde=True, bins=30, ax=ax_prs)
    ax_prs.set_title("Distribution of Polygenic Score")
    st.pyplot(fig_prs, clear_figure=True)
    st.dataframe(cohort["polygenic_score"].describe())
    st.markdown("---")

    st.subheader("Quantitative Trait (`quant_trait`)")
    fig_quant, ax_quant = plt.subplots(figsize=(7, 4))
    sns.histplot(cohort["quant_trait"], kde=True, bins=30, ax=ax_quant)
    ax_quant.set_title("Distribution of Quantitative Trait")
    st.pyplot(fig_quant, clear_figure=True)
    st.dataframe(cohort["quant_trait"].describe())
    st.markdown("---")

    st.subheader("Disease Prob (`disease_prob`)")
    fig_dis_prob, ax_dis_prob = plt.subplots(figsize=(7, 4))
    sns.histplot(cohort["disease_prob"], kde=True, bins=30, ax=ax_dis_prob)
    ax_dis_prob.set_title("Distribution of Disease Prob")
    st.pyplot(fig_dis_prob, clear_figure=True)
    st.dataframe(cohort["disease_prob"].describe())
    st.markdown("---")

    st.subheader("PC1")
    fig_pc1, ax_pc1 = plt.subplots(figsize=(7, 4))
    sns.histplot(cohort["PC1"], kde=True, bins=30, ax=ax_pc1)
    ax_pc1.set_title("Distribution of PC1")
    st.pyplot(fig_pc1, clear_figure=True)
    st.dataframe(cohort["PC1"].describe())
    st.markdown("---")

    st.subheader("PC2")
    fig_pc2, ax_pc2 = plt.subplots(figsize=(7, 4))
    sns.histplot(cohort["PC2"], kde=True, bins=30, ax=ax_pc2)
    ax_pc2.set_title("Distribution of PC2")
    st.pyplot(fig_pc2, clear_figure=True)
    st.dataframe(cohort["PC2"].describe())
    st.markdown("---")

    # --- Correlation and Multicollinearity ---
    st.header("Correlation and Multicollinearity Analysis")

    # 1. Correlation Matrix
    st.subheader("Correlation Matrix (Numerical Features)")
    numerical_vars_for_corr = [
        "age", "env_index", "polygenic_score", "quant_trait", "disease_prob", "PC1", "PC2"
    ]
    corr_matrix = cohort[numerical_vars_for_corr].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax_corr, linewidths=.5,
                linecolor='gray')
    ax_corr.set_title("Correlation Matrix")
    st.pyplot(fig_corr, clear_figure=True)

    # 2. VIF Analysis (Multicollinearity Check)
    st.subheader("Variance Inflation Factor (VIF) Results")

    # Note: We must exclude 'disease_status' and 'quant_trait' from the VIF calculation
    # as VIF is only calculated for predictor variables.
    predictor_features = [
        "age", "env_index", "polygenic_score", "disease_prob", "PC1", "PC2"
    ]
    vif_df = calculate_vif(cohort, predictor_features)
    st.dataframe(vif_df, hide_index=True, use_container_width=True)

    # --- Categorical Variable Distributions ---
    st.header("Categorical Feature Distributions")

    # Distributions by Feature
    st.subheader("Sex")
    fig_sex, ax_sex = plt.subplots(figsize=(7, 4))
    sns.countplot(x=cohort["sex"], ax=ax_sex, palette="viridis")
    ax_sex.set_title("Count of Sex Categories")
    ax_sex.set_xlabel("Sex")
    ax_sex.set_ylabel("Count")
    st.pyplot(fig_sex, clear_figure=True)
    st.dataframe(cohort["sex"].describe())
    st.markdown("---")

    st.subheader("Disease Status")
    fig_dis_stat, ax_dis_stat = plt.subplots(figsize=(7, 4))
    sns.countplot(x=cohort["disease_status"], ax=ax_dis_stat, palette="viridis")
    ax_dis_stat.set_title("Count of Disease Status Categories")
    ax_dis_stat.set_xlabel("Disease Status")
    ax_dis_stat.set_ylabel("Count")
    st.pyplot(fig_dis_stat, clear_figure=True)
    st.dataframe(cohort["disease_status"].describe())
    st.markdown("---")

    # --- Correlation and Multicollinearity ---
    def display_categorical_matrix(cohort):
        """
        Generates and displays a matrix of visualizations for categorical variables.
        - Diagonal: Count plots showing individual variable distribution.
        - Off-Diagonal: Heatmaps showing the row-normalized contingency table (relationship).
        """
        st.header("Categorical Variable Relationship Matrix")

        # Define the categorical variables to analyze
        cohort_categorical_vars = ["sex", "disease_status"]
        n_cohort_categorical_vars = len(cohort_categorical_vars)

        # Setup the figure and axes for the subplot matrix
        fig, axes = plt.subplots(
            n_cohort_categorical_vars,
            n_cohort_categorical_vars,
            figsize=(n_cohort_categorical_vars * 4, n_cohort_categorical_vars * 3.5)
        )

        plt.suptitle("Pairwise Categorical Relationships", y=1.01, fontsize=16)

        for i in range(n_cohort_categorical_vars):
            for j in range(n_cohort_categorical_vars):
                var1 = cohort_categorical_vars[i]
                var2 = cohort_categorical_vars[j]

                # --- Ensure 'axes' is indexed correctly, especially for 1x1 case ---
                if n_cohort_categorical_vars == 1:
                    ax = axes
                else:
                    ax = axes[i, j]

                # --- Diagonal: Individual Distribution (Count Plot) ---
                if i == j:
                    sns.countplot(
                        y=cohort[var1],
                        ax=ax,  # Use the correct ax variable
                        palette="Pastel1",
                        order=cohort[var1].value_counts().index,
                        # hue is redundant for a single variable countplot
                        legend=False
                    )
                    ax.set_title(f"Distribution of **{var1}**", fontsize=12)
                    ax.set_ylabel("")  # Clear y-label as it's the variable name
                    ax.set_xlabel("Count")

                # --- Off-Diagonal: Pairwise Relationship (Heatmap) ---
                else:
                    # Create the contingency table and normalize it by row (index)
                    # This shows the conditional probability P(var2 | var1)
                    contingency_table = pd.crosstab(cohort[var1], cohort[var2], normalize='index')

                    sns.heatmap(
                        contingency_table,
                        annot=True,
                        fmt=".2f",  # Format values to 2 decimal places
                        cmap="YlGnBu",
                        cbar=False,
                        ax=ax,  # Use the correct ax variable
                        linewidths=.5,
                        linecolor='gray'
                    )
                    ax.set_title(f"**{var1}** vs **{var2}** (Row Normalized)", fontsize=12)
                    ax.set_ylabel(var1)
                    ax.set_xlabel(var2)

        # Adjust layout to prevent overlapping titles/labels
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Display the Matplotlib figure in Streamlit
        st.pyplot(fig, clear_figure=True)

        st.markdown("---")

    display_categorical_matrix(cohort)

    def display_correlation_matrix(cohort):
        """
        Generates and displays a heatmap of the correlation matrix for the cohort DataFrame.
        """
        st.header("Correlation Matrix Heatmap️")

        # 1. Create the Matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # 2. Generate the heatmap using seaborn
        sns.heatmap(
            cohort.corr(numeric_only=True),  # Calculate correlation. Use numeric_only=True for safety.
            annot=True,  # Display the correlation value on the map
            fmt=".2f",  # Format the annotation to 2 decimal places
            cmap="coolwarm",  # Choose a color map (coolwarm is excellent for correlation)
            cbar=True,  # Display the color bar
            ax=ax  # Pass the axis object
        )

        ax.set_title("Correlation Matrix")

        # 3. Display the Matplotlib figure in Streamlit
        st.pyplot(fig, clear_figure=True)

        st.markdown("---")

    display_correlation_matrix(cohort)

    def display_pca_comparison(cohort):
        """
        Displays three PCA scatter plots (PC1 vs PC2) side-by-side,
        colored by quantitative trait, sex, and environmental index.
        """
        st.header("PCA Bi-Plot Comparison (PC1 vs PC2)")

        # 1. Setup the combined figure with 1 row and 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.35)

        ### --- Plot 1: Colored by Quantitative Trait ---
        ax1 = axes[0]
        sns.scatterplot(
            x="PC1",
            y="PC2",
            data=cohort,
            hue="quant_trait",
            palette="viridis",
            alpha=0.7,
            legend=False,
            ax=ax1
        )
        # Add colorbar manually for continuous variable
        norm = Normalize(cohort["quant_trait"].min(), cohort["quant_trait"].max())
        sm = ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])  # Necessary for ScalarMappable
        fig.colorbar(sm, ax=ax1, label="Quantitative Trait")

        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.set_title("Colored by Quantitative Trait")

        ### --- Plot 2: Colored by Sex ---
        ax2 = axes[1]
        sns.scatterplot(
            x="PC1",
            y="PC2",
            data=cohort,
            hue="sex",
            palette="Set2",
            alpha=0.7,
            ax=ax2
        )
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("Colored by Sex")
        ax2.legend(title="Sex", loc='upper right', bbox_to_anchor=(1.35, 1.05))

        ### --- Plot 3: Colored by Environmental Index ---
        ax3 = axes[2]
        sns.scatterplot(
            x="PC1",
            y="PC2",
            data=cohort,
            hue="env_index",
            palette="viridis",
            alpha=0.7,
            legend=False,
            ax=ax3
        )
        # Add colorbar manually for continuous variable
        norm = Normalize(cohort["env_index"].min(), cohort["env_index"].max())
        sm = ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax3, label="Env Index")

        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.set_title("Colored by Environmental Index")

        # Set a Super Title for the entire figure
        fig.suptitle("PCA Bi-Plot Comparison (PC1 vs PC2)", fontsize=16, y=1.02)

        # 2. Display the entire figure in Streamlit
        st.pyplot(fig, clear_figure=True)

        st.markdown("---")

    display_pca_comparison(cohort)

    def display_numerical_boxplots(cohort):
        """
        Generates and displays side-by-side box plots for all numerical variables
        in the cohort DataFrame to check for distribution and outliers.
        """
        st.header("Numerical Feature Outlier Check (Box Plots)")

        # 1. Define the numerical variables
        # (Using all numerical columns in the dummy data for this example)
        cohort_numerical_vars = cohort.select_dtypes(include=np.number).columns.tolist()
        cohort_n_numerical_cols = len(cohort_numerical_vars)

        # Check if there are numerical columns to plot
        if not cohort_numerical_vars:
            st.warning("No numerical columns found in the cohort DataFrame to display box plots.")
            return

        # 2. Setup the figure and axes
        # Use one row and N columns (one for each numerical variable)
        fig, axes = plt.subplots(
            1,
            cohort_n_numerical_cols,
            figsize=(4 * cohort_n_numerical_cols, 6)  # Adjusted height to 6 for better vertical fit
        )

        # Ensure axes is iterable even if there's only one column
        if cohort_n_numerical_cols == 1:
            axes = [axes]

        # 3. Create the plots in a loop
        for i, col in enumerate(cohort_numerical_vars):
            sns.boxplot(y=cohort[col], ax=axes[i], color='skyblue')
            axes[i].set_title(col, fontsize=12)
            axes[i].set_ylabel("")  # Remove y-axis label to keep things clean

        fig.suptitle("Outlier and Distribution Check (Box Plots)", fontsize=16, y=1.05)

        # Adjust layout to prevent overlapping titles/labels
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # 4. Display the Matplotlib figure in Streamlit
        st.pyplot(fig, clear_figure=True)

        st.markdown("---")

    display_numerical_boxplots(cohort)

    def display_outlier_zscore_analysis(cohort):
        """
        Performs Z-score standardization and counts outliers for numerical columns,
        displaying the results in Streamlit.
        """
        st.header("Outlier Analysis (Z-score Method)")

        # 1. Calculate Z-scores and store them in the DataFrame
        # Determine which columns to standardize based on the original code
        vars_to_analyze = [
            "age", "env_index", "quant_trait", "polygenic_score",
            "disease_prob", "PC1", "PC2"
        ]

        outlier_counts = {}

        for var in vars_to_analyze:
            # Check if the column exists to prevent errors
            if var in cohort.columns:
                # Calculate z-scores
                z_col_name = f"{var}_z"
                cohort[z_col_name] = zscore(cohort[var])

                # Identify rows with extreme |z| > 3
                outliers = cohort[cohort[z_col_name].abs() > 3]
                outlier_counts[var] = len(outliers)
            else:
                outlier_counts[var] = "N/A (Column Missing)"

        # 2. Display the results using st.dataframe for a clean table view
        st.subheader("Count of Extreme Outliers (|Z| > 3)")

        # Convert results to a DataFrame for display
        results_df = pd.DataFrame(
            outlier_counts.items(),
            columns=["Feature", "Outlier Count"]
        ).sort_values(by="Outlier Count", ascending=False)

        # Highlight features with outliers
        def highlight_outliers(s):
            # Check if the Outlier Count is greater than 0
            is_outlier = s['Outlier Count'] > 0

            # If there are outliers, apply the red background style to BOTH columns in that row.
            if is_outlier:
                # Return a list of styles for the columns in the row: [Feature_Style, Outlier_Count_Style]
                return ['background-color: #600000', 'background-color: #600000']
            else:
                # If no outliers, return an empty style string for both columns
                return ['', '']

        st.dataframe(results_df.style.apply(highlight_outliers, axis=1), hide_index=True)

        # 3. Optional: Display the total count of samples that have at least one outlier
        all_z_cols = [f"{var}_z" for var in vars_to_analyze if f"{var}_z" in cohort.columns]
        total_outliers = cohort[cohort[all_z_cols].abs().gt(3).any(axis=1)]
        st.info(
            f"**Total unique samples** containing at least one outlier across these features: **{len(total_outliers)}** out of {len(cohort)} samples.")

        st.markdown("---")

    display_outlier_zscore_analysis(cohort)


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
    st.title("Evaluating Classification, Linear, and Shrinkage Models for Predicting a Simulated Disease Status and Polygenic Trait Within A Population")

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
        ("LR Full + PCA", LinearRegression, features_full_pca, {}),
        ("Ridge Full - No PCA", RidgeCV, features_full_no_pca, {}),
        ("LR Full (PRS + Covariates - No PCA)", LinearRegression, features_full_no_pca, {}),
    ]

    # --- Run Models and Display Results ---
    tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Exploratory Data Analysis", "Classification (Disease Status)", "Regression (Quant Trait)"])

    # Store results for sidebar summary
    clf_results_display = []
    reg_results_display = []

    # =========================================================================
    # TAB 1: Introduction
    # =========================================================================
    with tab1:
        st.header("Project Introduction: Simulating Genetic and Environmental Influences")
        st.markdown(
            """
            In this project, we analyze a **simulated cohort** generated from a coalescent-based genetic model. Each row corresponds to one individual and includes both demographic covariates and genotype-derived predictors suitable for statistical learning.

            ### Data Generation: The `msprime` Pipeline
            The dataset used was generated using a population-genetic simulation pipeline built with **msprime**, a coalescent simulator designed for scalable and biologically realistic genomic data. Instead of relying on a pre-existing real-world dataset, this approach creates a fully synthetic cohort in which the **true effect sizes, covariates, and sources of noise are known**. This allows direct evaluation of how well different statistical learning methods recover the underlying structure of the data.

            The `msprime` pipeline produces a tree sequence under a Wright–Fisher coalescent model. Key steps include:
            * Extracting a diploid genotype matrix.
            * Designating a subset of variants as **causal**, each assigned an effect size.
            * Using these effect sizes to construct a standardized **polygenic score** for each individual.
            * Generating demographic and environmental covariates (**sex, age, env\_index**) to mimic realistic non-genetic influences.

            A continuous **quantitative trait (`quant_trait`)** is then created from a linear model combining the polygenic score, covariates, and Gaussian noise. This design ensures a dataset with controlled polygenic signal, demographic structure, and stochastic variation. Optional **principal components (PC1, PC2, …)** derived from genotype PCA provide covariates for controlling population structure.

            ### Dataset Components
            Overall, each row of the dataset includes:
            * **Demographic Covariates**: `sex`, `age`, `env_index`
            * **Genetic Predictors**: `polygenic_score`, optional PCA components (`PC1`, `PC2`, etc.)
            * **Response Variable**: `quant_trait` (continuous) and `disease_status` (binary derived from `quant_trait`).

            ### Project Goals
            The goals of this project are to use statistical learning methods to:
            * Quantify how much of the variation in the quantitative trait is explained by the polygenic score and covariates.
            * Evaluate whether controlling for population structure via principal components improves prediction.
            * Compare classical linear models and shrinkage methods.
            * Assess how well the fitted models recover the true underlying effect sizes used during data generation.
            """
        )

    # =========================================================================
    # TAB 2: EDA
    # =========================================================================
    with tab2:
        st.header("EDA")
        display_eda(cohort)

    # =========================================================================
    # TAB 3: CLASSIFICATION
    # =========================================================================
    with tab3:
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

            if name == "Logistic Regression":
                comment = f"""
                        The Logistic Regression model is a moderately effective classifier for predicting Disease Status. With a 
                        Test Accuracy of approximately $68.4\%$ and a Test AUC of $0.758$, the model is stable (no overfitting) 
                        and demonstrates useful predictive power over random guessing.
                        """
            elif name == "LDA":
                comment = f"""
                        The Linear Discriminant Analysis (LDA) model is a moderately effective classifier for predicting Disease 
                        Status. With a Test Accuracy of approximately $68.5\%$ and a Test AUC of $0.758$, the model is stable 
                        (no overfitting) and demonstrates useful predictive power over random guessing.
                        """
            else:
                comment = f"""
                        The Linear SVM model is a moderately effective classifier for predicting Disease Status. With a Test 
                        Accuracy of approximately $68.4\%$ and a Test AUC of $0.758$, the model is stable (no overfitting) and 
                        demonstrates useful predictive power over random guessing.
                        """

            st.markdown(comment)

            # Display Feature Coefficients
            if hasattr(model, 'coef_'):
                coef_df = pd.DataFrame(model.coef_[0], index=features, columns=['Coefficient'])
                st.write("**Feature Coefficients** (Impact on log-odds):")
                st.dataframe(coef_df.sort_values(by='Coefficient', ascending=False), use_container_width=True)

            clf_results_display.append(
                {'Model': name, 'Test Accuracy': f"{test_acc:.4f}", 'Test AUC': f"{test_auc:.4f}"})
            st.markdown("---")

    # =========================================================================
    # TAB 4: REGRESSION
    # =========================================================================
    with tab4:
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

            if name == "LR Full + PCA":
                comment = f"""
                        The Linear Regression (LR Full + PCA) model demonstrates a strong linear relationship between the full 
                        set of predictors (including the Principal Components) and the True Values, achieving a Test $R^2$ of 
                        $0.567$. This means the full set of features accounts for approximately $56.7\%$ of the variance in the 
                        predicted trait, achieving performance almost identical to the model using raw covariates. The model is 
                        highly stable with low error (Test RMSE approx $0.644$) and shows no significant signs of bias 
                        or non-linearity, as confirmed by the Residuals vs Fitted plot (which is randomly scattered around zero) 
                        and the Residual Distribution (which is perfectly normally distributed). The QQ Plot also shows the 
                        residuals closely follow the theoretical line, confirming the model adheres perfectly to the assumption 
                        of normally distributed errors.
                        """
            elif name == "Ridge Full - No PCA":
                comment = f"""
                        The Ridge Regression model demonstrates a strong linear relationship between the predictors and the True 
                        Values, achieving a Test $R^2$ of $0.566$. This means the model accounts for approximately $56.6\%$ of the 
                        variance in the predicted trait. The model is highly stable with low error (Test RMSE approx $0.644$) and 
                        shows no significant signs of bias or non-linearity, as confirmed by the Residuals vs Fitted plot (which 
                        is randomly scattered around zero) and the Residual Distribution (which is perfectly normally distributed). 
                        The QQ Plot also shows the residuals closely follow the theoretical line, confirming the model adheres 
                        perfectly to the assumption of normally distributed errors, while the alpha regularization helps mitigate 
                        the effects of multicollinearity (VIF $> 10$) observed in the original features.
                        """
            else:
                comment = f"""
                        The Linear Regression Full Model (PRS + covariates) demonstrates a stronger linear relationship between 
                        the full set of predictors and the True Values, achieving a Test $R^2$ of $0.566$. This means the Polygenic 
                        Score and covariates together account for about $56.6\%$ of the variance in the predicted trait, an 
                        improvement over the baseline model. The model is highly stable with low error (Test RMSE approx $0.644$) 
                        and shows no significant signs of bias or non-linearity, as confirmed by the Residuals vs Fitted plot 
                        (which is randomly scattered around zero) and the Residual Distribution (which is perfectly normally 
                        distributed). The QQ Plot also shows the residuals closely follow the theoretical line, indicating the 
                        model adheres perfectly to the assumption of normally distributed errors.
                        """

            st.markdown(comment)

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