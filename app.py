import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ---- Page setup ----
st.set_page_config(page_title="Credit Default Risk Predictor", layout="wide")
st.title("🏦 Credit Default Risk Predictor")
st.write(
    "A machine learning tool that estimates an applicant's probability of loan default, "
    "with a transparent, feature-by-feature explanation of every prediction — "
    "built on the Home Credit Default Risk dataset (307,511 applicants)."
)
with st.expander("ℹ️ About this project"):
    st.write("""
    This tool is built on the **Home Credit Default Risk** dataset (307,511 loan applicants).
    
    **Model**: XGBoost classifier, tuned via RandomizedSearchCV  
    **Performance**: AUC-ROC 0.7687, KS-Statistic 0.4046 on held-out test data  
    **Key features**: External credit bureau scores (EXT_SOURCE_1/2/3) combined with 
    engineered features from applicants' prior loan history (refusal rates, payment 
    shortfalls, active loan ratios)
    
    Every prediction is paired with a SHAP explanation showing exactly which factors 
    drove that specific result — built with real-world credit risk deployment 
    requirements in mind, where model decisions need to be explainable, not just accurate.
    """)

# ---- Load model and reference data (cached so it only loads once) ----
@st.cache_resource
def load_model():
    with open('best_xgb_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_reference_data():
    X_test = pd.read_csv('X_test_sample.csv')
    return X_test

model = load_model()
X_test_full = load_reference_data()

# ---- Build a "template" row using median/mode values from the real data ----
# This gives every non-exposed feature a sensible default instead of 0 or NaN
default_row = X_test_full.median(numeric_only=True)

# ---- Sidebar: user inputs (the ~10 features that actually drive predictions) ----
st.sidebar.header("Applicant Details")
st.sidebar.subheader("Quick Start")
example = st.sidebar.selectbox(
    "Load an example applicant",
    ["Custom (use sliders below)", "Low-Risk Example", "High-Risk Example", "Borderline Example"]
)

example_profiles = {
    "Low-Risk Example": {
        "ext_1": 0.85, "ext_2": 0.90, "ext_3": 0.80,
        "credit": 300000, "goods": 280000, "years_emp": 12.0,
        "refusal": 0.0, "shortfall": 0.0, "gender": "Female", "car": "Yes"
    },
    "High-Risk Example": {
        "ext_1": 0.05, "ext_2": 0.02, "ext_3": 0.08,
        "credit": 700000, "goods": 650000, "years_emp": 0.5,
        "refusal": 0.8, "shortfall": 5000.0, "gender": "Male", "car": "No"
    },
    "Borderline Example": {
        "ext_1": 0.35, "ext_2": 0.40, "ext_3": 0.30,
        "credit": 500000, "goods": 480000, "years_emp": 3.0,
        "refusal": 0.2, "shortfall": 500.0, "gender": "Male", "car": "Yes"
    }
}

defaults = example_profiles.get(example, None)

ext_source_1 = st.sidebar.slider("External Credit Score 1", 0.0, 1.0, 
    defaults["ext_1"] if defaults else 0.5, 0.01)
ext_source_2 = st.sidebar.slider("External Credit Score 2", 0.0, 1.0, 
    defaults["ext_2"] if defaults else 0.5, 0.01)
ext_source_3 = st.sidebar.slider("External Credit Score 3", 0.0, 1.0, 
    defaults["ext_3"] if defaults else 0.5, 0.01)

amt_credit = st.sidebar.number_input("Loan Amount ($)", min_value=0, 
    value=defaults["credit"] if defaults else 500000, step=10000)
amt_goods_price = st.sidebar.number_input("Goods Price ($)", min_value=0, 
    value=defaults["goods"] if defaults else 450000, step=10000)

days_employed_years = st.sidebar.slider("Years Employed", 0.0, 40.0, 
    defaults["years_emp"] if defaults else 5.0, 0.5)

refusal_rate = st.sidebar.slider("Prior Loan Refusal Rate", 0.0, 1.0, 
    defaults["refusal"] if defaults else 0.0, 0.05)
avg_payment_shortfall = st.sidebar.number_input("Avg. Payment Shortfall ($)", 
    value=defaults["shortfall"] if defaults else 0.0, step=100.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"], 
    index=0 if (defaults and defaults["gender"] == "Male") else 1 if defaults else 0)
own_car = st.sidebar.selectbox("Owns a Car", ["Yes", "No"], 
    index=0 if (defaults and defaults["car"] == "Yes") else 1 if defaults else 0)
# ---- Assemble the full feature row for prediction ----
input_row = default_row.copy()

input_row['EXT_SOURCE_1'] = ext_source_1
input_row['EXT_SOURCE_2'] = ext_source_2
input_row['EXT_SOURCE_3'] = ext_source_3
input_row['AMT_CREDIT'] = amt_credit
input_row['AMT_GOODS_PRICE'] = amt_goods_price
input_row['DAYS_EMPLOYED'] = -days_employed_years * 365  # matches the dataset's negative-days convention
input_row['REFUSAL_RATE'] = refusal_rate
input_row['AVG_PAYMENT_SHORTFALL'] = avg_payment_shortfall

if 'CODE_GENDER_M' in input_row.index:
    input_row['CODE_GENDER_M'] = 1.0 if gender == "Male" else 0.0
if 'FLAG_OWN_CAR_Y' in input_row.index:
    input_row['FLAG_OWN_CAR_Y'] = 1.0 if own_car == "Yes" else 0.0

input_df = pd.DataFrame([input_row])[X_test_full.columns]  # enforce correct column order

# ---- Predict ----
if st.sidebar.button("Predict Risk", type="primary"):
    proba = model.predict_proba(input_df)[:, 1][0]
    threshold = 0.0846  # your Day 2 KS-optimal threshold
    prediction = "HIGH RISK" if proba >= threshold else "LOW RISK"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Default Probability", f"{proba*100:.2f}%")
    with col2:
        if prediction == "HIGH RISK":
            st.error(f"Classification: {prediction}")
        else:
            st.success(f"Classification: {prediction}")

    st.caption(f"Classification threshold: {threshold*100:.2f}% (optimized via KS-statistic, Week 3 Day 2)")

    # ---- SHAP explanation for this specific prediction ----
    st.subheader("Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns.tolist()
    )
    shap.plots.waterfall(shap_explanation, max_display=10, show=False)
    st.pyplot(fig)
else:
    st.info("Adjust applicant details in the sidebar, then click **Predict Risk** to see results.")
st.divider()
st.subheader("How This Model Makes Decisions")
st.write(
    "Across the full test set, these are the features with the greatest overall "
    "influence on the model's predictions:"
)

@st.cache_data
def get_global_importance():
    sample = X_test_full.sample(n=500, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)
    importance = pd.DataFrame({
        'feature': sample.columns,
        'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False).head(10)
    return importance

importance_df = get_global_importance()

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(importance_df['feature'][::-1], importance_df['mean_abs_shap'][::-1], color='#1f77b4')
ax2.set_xlabel('Mean |SHAP value|')
ax2.set_title('Top 10 Global Feature Importance')
plt.tight_layout()
st.pyplot(fig2)
    