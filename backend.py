# Re-import after code execution environment reset
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Sample function to compute ATE using Propensity Score Matching (PSM)
def compute_ate_with_psm(data_path, treatment_col, outcome_col, confounders, control_value):
    # Load dataset
    df = pd.read_csv(data_path)

    # Step 1: Create binary treatment column
    treatment_values = df[treatment_col].unique()
    treat_value = [v for v in treatment_values if v != control_value][0]
    df['is_treated'] = (df[treatment_col] == treat_value).astype(int)

    # Step 2: Propensity score model
    X = pd.get_dummies(df[confounders], drop_first=True)
    y = df['is_treated']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    df['propensity_score'] = model.predict_proba(X)[:, 1]

    # Step 3: Matching
    treated = df[df['is_treated'] == 1]
    control = df[df['is_treated'] == 0]
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    matched_control = control.iloc[indices.flatten()]

    # Step 4: ATE computation
    treated_outcome = treated[outcome_col].values
    control_outcome = matched_control[outcome_col].values
    ate = np.mean(treated_outcome - control_outcome)

    # Step 5: Visualization
    plt.figure(figsize=(8, 5))
    sns.histplot(treated_outcome - control_outcome, kde=True, bins=30, color='skyblue')
    plt.axvline(ate, color='red', linestyle='--', label=f'ATE = {ate:.4f}')
    plt.title(f'ATE Estimate using PSM: {treat_value} vs {control_value}')
    plt.xlabel('Outcome Difference (Treated - Control)')
    plt.ylabel('Frequency')
    plt.legend()
    plot_path = "/mnt/data/ate_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return ate, plot_path
