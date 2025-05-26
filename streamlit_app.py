import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Hypothesis-Driven Causal Explanation", layout="wide")

st.title("Hypothesis-Driven Causal Explanation")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of uploaded data", df.head())

    with st.form("ate_form"):
        treatment_col = st.selectbox("Select Treatment Column", df.columns)
        outcome_col = st.selectbox("Select Outcome Column", df.columns)
        confounders = st.multiselect("Select Confounder Columns", df.columns.drop([treatment_col, outcome_col]))
        
        treatment_values = df[treatment_col].dropna().unique()
        control_value = st.selectbox("Select Control Value (baseline group)", treatment_values)

        submitted = st.form_submit_button("Compute ATE")
        
        if submitted:
            try:
                # Prepare data
                treat_value = [v for v in treatment_values if v != control_value][0]
                df['is_treated'] = (df[treatment_col] == treat_value).astype(int)

                X = pd.get_dummies(df[confounders], drop_first=True)
                y = df['is_treated']

                # Propensity Score Estimation
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)
                df['propensity_score'] = model.predict_proba(X)[:, 1]

                # Matching
                treated = df[df['is_treated'] == 1]
                control = df[df['is_treated'] == 0]
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(control[['propensity_score']])
                distances, indices = nn.kneighbors(treated[['propensity_score']])
                matched_control = control.iloc[indices.flatten()]

                # ATE
                treated_outcome = treated[outcome_col].values
                control_outcome = matched_control[outcome_col].values
                ate = np.mean(treated_outcome - control_outcome)

                # Plot
                st.write(f"### Average Treatment Effect (ATE): `{ate:.4f}`")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(treated_outcome - control_outcome, kde=True, bins=30, color='skyblue', ax=ax)
                ax.axvline(ate, color='red', linestyle='--', label=f'ATE = {ate:.4f}')
                ax.set_title(f'Distribution of Outcome Differences ({treat_value} vs {control_value})')
                ax.set_xlabel('Change in Outcome (Treated - Control)')
                ax.set_ylabel('Count')
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error computing ATE: {e}")
