import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from mlxtend.regressor import StackingRegressor

st.set_page_config(page_title="G-Computation Hypothesis Explorer", layout="wide")
st.title("🔍 Hypothesis-Driven Explanation using G-Computation")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview", df.head())

    st.markdown("---")

    # Step 1: Select treatment column
    treatment_col = st.selectbox("1. Select Treatment Column", df.columns)

    # Step 2: Select rest in a form
    treatment_values = df[treatment_col].dropna().unique()

    with st.form("gcomp_form"):
        outcome_col = st.selectbox("2. Select Outcome Column", df.columns.drop(treatment_col))
        confounders = st.multiselect("3. Select Confounders", df.columns.drop([treatment_col, outcome_col]))

        treatment_shift = st.number_input(
            f"4. Shift value to apply to {treatment_col}",
            value=0.0,
            help="This value will be added to the treatment (e.g., -100 means reduce credit amount by 100)"
        )


        submitted = st.form_submit_button("Compute Causal Effect")

    if submitted:
        try:
            num_features = [col for col in confounders if df[col].dtype in [np.float64, np.int64]]
            cat_features = [col for col in confounders if df[col].dtype == object or df[col].dtype.name == 'category']
            all_features = confounders + [treatment_col]

            preprocessor = ColumnTransformer(transformers=[
                ('num', StandardScaler(), num_features + [treatment_col]),
                ('cat', OneHotEncoder(drop='first'), cat_features)
            ])

            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            knn = KNeighborsRegressor(n_neighbors=5)
            mlp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)

            super_learner = StackingRegressor(
                regressors=[rf, gb, knn, mlp],
                meta_regressor=LinearRegression()
            )

            model = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("superlearner", super_learner)
            ])

            model.fit(df[all_features], df[outcome_col])
            df['risk_now'] = model.predict(df[all_features])

            # Counterfactual
            df_cf = df.copy()
            df_cf[treatment_col] = df_cf[treatment_col] + treatment_shift
            df['risk_counterfactual'] = model.predict(df_cf[all_features])

            # Causal Effect
            df['causal_effect'] = df['risk_counterfactual'] - df['risk_now']
            average_effect = df['causal_effect'].mean()

            # Plot
            st.write(f"### Average Causal Effect: `{average_effect:.4f}`")


            st.markdown("## Hypothesis Evaluation")
            st.markdown(f"**Hypothesis:** Changing `{treatment_col}` by `{treatment_shift}`, in the presence of confounders, `{confounders}`, affects the model's prediction (`{outcome_col}`).")

            # Evidence For / Against based on thresholds
            evidence_threshold = 0.05  # can adjust this as needed
            direction = "positive" if average_effect > 0 else "negative"

            if abs(average_effect) > evidence_threshold:
                st.success(f"**Evidence For:** The average effect was `{average_effect:.3f}` ({direction} impact).")
                st.markdown(f"The model responds noticeably to changes in `{treatment_col}`, suggesting this feature influences predictions.")
            else:
                st.warning(f"**Evidence Against:** The average effect was `{average_effect:.3f}`, which is relatively small.")
                st.markdown(f"This suggests that changing `{treatment_col}` by `{treatment_shift}` has little influence on model predictions.")


            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['causal_effect'], bins=30, kde=True, color='skyblue', ax=ax)
            ax.axvline(average_effect, color='red', linestyle='--', label=f"Avg Effect = {average_effect:.4f}")
            title_text = f"{'+' if treatment_shift >= 0 else ''}{treatment_shift} shift in {treatment_col}"
            ax.set_title(f"Distribution of Causal Effect\n({title_text})")
            ax.set_xlabel("Change in Predicted Risk")
            ax.set_ylabel("Number of Observations")
            ax.legend()
            st.pyplot(fig)

            st.caption("This shows the change in model output when the treatment variable is perturbed while holding confounders fixed.")

        except Exception as e:
            st.error(f"Error: {e}")
