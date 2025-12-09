# linkedin_app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import altair as alt

# ---------------------------------------------------
# Utility: cleaning + model training (matches notebook)
# ---------------------------------------------------

def clean_sm(x):
    """Return 1 if x == 1, else 0 (binary LinkedIn indicator)."""
    return np.where(x == 1, 1, 0)


def load_and_clean_data(csv_path="social_media_usage.csv"):
    """
    Load Pew data and create cleaned dataframe.

    - Target from web1b (LinkedIn usage in write-up)
    - Same missingness rules as notebook
    """
    s = pd.read_csv(csv_path)
    ss = s.copy()

    # Target: LinkedIn usage
    ss["sm_li"] = clean_sm(ss["web1b"])

    # income: 1–9 valid, >9 missing
    ss["income"] = ss["income"].where(ss["income"] <= 9, np.nan)

    # education: 1–8 valid, >8 missing
    ss["education"] = ss["educ2"].where(ss["educ2"] <= 8, np.nan)

    # parent: 1–2 valid, >2 missing → binary
    ss["parent"] = np.where(
        ss["par"] == 1, 1,
        np.where(ss["par"] == 2, 0, np.nan)
    )

    # married: 1–6 valid, >6 missing → binary
    ss["married"] = np.where(
        ss["marital"] == 1, 1,
        np.where(ss["marital"].isin([2, 3, 4, 5, 6]), 0, np.nan)
    )

    # female: 2 = female, 1 = male, others missing
    ss["female"] = np.where(
        ss["gender"] == 2, 1,
        np.where(ss["gender"] == 1, 0, np.nan)
    )

    # age: <= 98 valid, >98 missing
    ss["age"] = ss["age"].where(ss["age"] <= 98, np.nan)

    # Keep only needed columns and drop missing
    cols_keep = ["sm_li", "income", "education", "parent", "married", "female", "age"]
    ss = ss[cols_keep].dropna().copy()

    return ss


@st.cache_data
def get_clean_data():
    return load_and_clean_data()


@st.cache_resource
def train_model(ss):
    """
    Train logistic regression and return:
    - model
    - metrics dict
    - train/test splits (for reference)
    """
    X = ss[["income", "education", "parent", "married", "female", "age"]]
    y = ss["sm_li"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=987,
        stratify=y
    )

    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report_dict": report_dict,
        "report_text": report_text,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }

    return model, metrics, X_train, X_test, y_train, y_test


# ---------------------------------------------------
# App layout
# ---------------------------------------------------

st.set_page_config(page_title="LinkedIn Predictor", layout="wide")

st.markdown("# LinkedIn Usage Prediction App")
st.write(
    """
Welcome to the **LinkedIn Usage Prediction App**.

Use the controls in the sidebar to describe a person, and the model will:
1. Predict **whether they are likely to be a LinkedIn user**.
2. Estimate the **probability** that they use LinkedIn.
3. Show **additional visual insights** from the data.
"""
)

# Load data and model
ss = get_clean_data()
model, metrics, X_train, X_test, y_train, y_test = train_model(ss)

# ---------------------------------------------------
# Sidebar – interactive inputs
# ---------------------------------------------------

st.sidebar.markdown("## Build a profile")

income_options = {
    1: "< $10,000",
    2: "$10k–$20k",
    3: "$20k–$30k",
    4: "$30k–$40k",
    5: "$40k–$50k",
    6: "$50k–$75k",
    7: "$75k–$100k",
    8: "$100k–$150k",
    9: "≥ $150k",
}
income_choice = st.sidebar.selectbox(
    "Household income bracket:",
    options=list(income_options.keys()),
    format_func=lambda x: f"{x}: {income_options[x]}"
)

edu_options = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate / GED",
    4: "Some college, no degree",
    5: "Associate degree",
    6: "Bachelor’s degree",
    7: "Some grad school",
    8: "Graduate / professional degree",
}
education_choice = st.sidebar.selectbox(
    "Highest education level:",
    options=list(edu_options.keys()),
    format_func=lambda x: f"{x}: {edu_options[x]}"
)

parent_choice = st.sidebar.radio(
    "Parent of a child under 18 at home?",
    options=[1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

married_choice = st.sidebar.radio(
    "Married?",
    options=[1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

female_choice = st.sidebar.radio(
    "Female?",
    options=[1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

age_choice = st.sidebar.slider(
    "Age:",
    min_value=18,
    max_value=98,
    value=42,
    step=1
)

st.sidebar.markdown("---")
run_prediction = st.sidebar.button("Predict LinkedIn usage")

# Single-row DataFrame for user profile
user_df = pd.DataFrame({
    "income":   [income_choice],
    "education":[education_choice],
    "parent":   [parent_choice],
    "married":  [married_choice],
    "female":   [female_choice],
    "age":      [age_choice]
})

# ---------------------------------------------------
# Live prediction
# ---------------------------------------------------

st.markdown("## Live Prediction")

if run_prediction:
    proba = model.predict_proba(user_df)[0, 1]
    pred_class = int(proba >= 0.5)
    label = "LinkedIn user" if pred_class == 1 else "Not a LinkedIn user"

    # --- Highlighted prediction result ---
    if pred_class == 1:
        st.markdown(
            """
            <div style="
                padding: 20px;
                border-radius: 10px;
                background-color: #e6f4ea;
                border-left: 8px solid #2e7d32;
            ">
                <h1 style="color:#2e7d32; margin-bottom:5px;">
                    Classified as a LinkedIn User
                </h1>
                <h3 style="margin-top:0;">
                    Predicted probability: {:.1f}%
                </h3>
            </div>
            """.format(proba * 100),
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="
                padding: 20px;
                border-radius: 10px;
                background-color: #fdecea;
                border-left: 8px solid #c62828;
            ">
                <h1 style="color:#c62828; margin-bottom:5px;">
                    Classified as NOT a LinkedIn User
                </h1>
                <h3 style="margin-top:0;">
                    Predicted probability of LinkedIn use: {:.1f}%
                </h3>
            </div>
            """.format(proba * 100),
            unsafe_allow_html=True
        )

    prob_df = pd.DataFrame(
        {"Outcome": ["LinkedIn user", "Not a LinkedIn user"],
         "Probability": [proba, 1 - proba]}
    )
    bar = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x="Outcome",
            y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Outcome", alt.Tooltip("Probability", format=".3f")]
        )
    )
    st.altair_chart(bar, use_container_width=True)
else:
    st.info("Use the sidebar to set a profile and click **Predict LinkedIn usage**.")

# ---------------------------------------------------
# Probability vs age for this profile
# ---------------------------------------------------

st.markdown("---")
st.markdown("## How does age change the probability for this profile?")

age_grid = np.arange(18, 99)
demo_df = pd.DataFrame({
    "income":   [income_choice]    * len(age_grid),
    "education":[education_choice] * len(age_grid),
    "parent":   [parent_choice]    * len(age_grid),
    "married":  [married_choice]   * len(age_grid),
    "female":   [female_choice]    * len(age_grid),
    "age":      age_grid
})
demo_df["prob"] = model.predict_proba(demo_df)[:, 1]

age_plot = (
    alt.Chart(demo_df)
    .mark_line()
    .encode(
        x=alt.X("age", title="Age"),
        y=alt.Y("prob", title="Predicted probability of LinkedIn use"),
        tooltip=["age", alt.Tooltip("prob", format=".3f")]
    )
    .interactive()
)
st.altair_chart(age_plot, use_container_width=True)
st.caption(
    "This shows how the predicted probability of LinkedIn use changes with age "
    "for the selected income / education / parent / marital / gender profile."
)

# ---------------------------------------------------
# Marketing dashboard: patterns in the data
# ---------------------------------------------------

st.markdown("---")
st.markdown("## Marketing dashboard: patterns in the data")

col1, col2 = st.columns(2)

# 1) LinkedIn use by income
with col1:
    st.markdown("### LinkedIn use by income group")
    income_stats = (
        ss.groupby("income")["sm_li"]
        .mean()
        .reset_index()
        .rename(columns={"sm_li": "linkedin_rate"})
    )
    income_plot = (
        alt.Chart(income_stats)
        .mark_bar()
        .encode(
            x=alt.X("income:O", title="Income group (1=lowest, 9=highest)"),
            y=alt.Y("linkedin_rate:Q", title="The proportion of people who use LinkedIn"),
            tooltip=["income", alt.Tooltip("linkedin_rate", format=".2f")]
        )
    )
    st.altair_chart(income_plot, use_container_width=True)

# 2) LinkedIn use by education
with col2:
    st.markdown("### LinkedIn use by education level")
    edu_stats = (
        ss.groupby("education")["sm_li"]
        .mean()
        .reset_index()
        .rename(columns={"sm_li": "linkedin_rate"})
    )
    edu_plot = (
        alt.Chart(edu_stats)
        .mark_bar()
        .encode(
            x=alt.X("education:O", title="Education level (1–8)"),
            y=alt.Y("linkedin_rate:Q", title="The proportion of people who use LinkedIn"),
            tooltip=["education", alt.Tooltip("linkedin_rate", format=".2f")]
        )
    )
    st.altair_chart(edu_plot, use_container_width=True)

st.caption(
    "These charts help the marketing team see which income and education "
    "segments are most likely to use LinkedIn, supporting targeted campaigns."
)

# Extra visualization: LinkedIn use by age group
st.markdown("### LinkedIn use by age group")

age_bins = [18, 29, 39, 49, 59, 69, 79, 89, 99]
age_labels = ["18–29", "30–39", "40–49", "50–59", "60–69", "70–79", "80–89", "90–98"]

ss_age = ss.copy()
ss_age["age_group"] = pd.cut(
    ss_age["age"], bins=age_bins, labels=age_labels, right=True
)

age_stats = (
    ss_age.groupby("age_group")["sm_li"]
    .mean()
    .reset_index()
    .rename(columns={"sm_li": "linkedin_rate"})
)

age_plot = (
    alt.Chart(age_stats)
    .mark_bar()
    .encode(
        x=alt.X("age_group:O", title="Age group"),
        y=alt.Y("linkedin_rate:Q", title="The proportion of people who use LinkedIn"),
        tooltip=["age_group", alt.Tooltip("linkedin_rate", format=".2f")]
    )
)
st.altair_chart(age_plot, use_container_width=True)
st.caption(
    "This chart shows how LinkedIn adoption varies by age group, "
    "which can help tailor campaigns to different age segments."
)
