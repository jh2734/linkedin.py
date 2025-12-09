import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import altair as alt


# ---------------------------------------------------
# Data cleaning + model training
# ---------------------------------------------------

def clean_sm(x):
    """Return 1 if x == 1, else 0."""
    return np.where(x == 1, 1, 0)


def load_and_clean_data(csv_path="social_media_usage.csv"):
    """Load Pew data and create cleaned ss dataframe."""
    s = pd.read_csv(csv_path)

    ss = s.copy()

    # Target: LinkedIn usage from web1h (1=yes, else 0)
    ss["sm_li"] = clean_sm(ss["web1h"])

    # income: 1–9 valid, 98/99 missing
    ss["income"] = ss["income"].where(ss["income"] <= 9, np.nan)

    # education: 1–8 valid, 98/99 missing
    ss["education"] = ss["educ2"].where(ss["educ2"] <= 8, np.nan)

    # parent: par (1 yes, 2 no, 8/9 missing)
    ss["parent"] = np.where(ss["par"] == 1, 1,
                     np.where(ss["par"] == 2, 0, np.nan))

    # married: marital (1 married vs everyone else)
    ss["married"] = np.where(ss["marital"] == 1, 1,
                      np.where(ss["marital"].isin([2, 3, 4, 5, 6]), 0, np.nan))

    # female: gender (2 = female, 1 = male, others missing)
    ss["female"] = np.where(ss["gender"] == 2, 1,
                     np.where(ss["gender"] == 1, 0, np.nan))

    # age: numeric, <= 97 valid, 98 / 97+ missing
    ss["age"] = ss["age"].where(ss["age"] <= 97, np.nan)

    # Keep only needed columns and drop missing
    ss = ss[["sm_li", "income", "education", "parent", "married", "female", "age"]]
    ss = ss.dropna().copy()

    return ss


@st.cache_data
def get_clean_data():
    return load_and_clean_data()


@st.cache_resource
def train_model(ss):
    """Train logistic regression and return model + test accuracy."""
    X = ss[["income", "education", "parent", "married", "female", "age"]]
    y = ss["sm_li"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc


# ---------------------------------------------------
# Streamlit main
# ---------------------------------------------------

st.set_page_config(page_title="My Streamlit LinkedIn App!!!", layout="wide")

st.markdown("# 💼 My Streamlit LinkedIn Data Science App!!!")
st.write(
    """
This app uses a **logistic regression model** trained on survey data to predict 
whether someone is likely to be a LinkedIn user, and with what probability.  
Use the controls to build a profile and see how the prediction changes.
"""
)

ss = get_clean_data()
model, test_acc = train_model(ss)

# ---------------------------------------------------
# Sidebar: interactive inputs (inspired by your examples)
# ---------------------------------------------------

st.sidebar.markdown("## ✏️ Build a profile")

######## Example 1 style: selectbox + manual conversion for education

educ_text = st.sidebar.selectbox(
    "Education level",
    options=[
        "Less than high school",
        "High school incomplete",
        "High school graduate / GED",
        "Some college, no degree",
        "Associate degree",
        "Bachelor’s degree",
        "Some graduate school",
        "Graduate / professional degree"
    ]
)

# Convert text to numeric educ2-style code
if educ_text == "Less than high school":
    education = 1
elif educ_text == "High school incomplete":
    education = 2
elif educ_text == "High school graduate / GED":
    education = 3
elif educ_text == "Some college, no degree":
    education = 4
elif educ_text == "Associate degree":
    education = 5
elif educ_text == "Bachelor’s degree":
    education = 6
elif educ_text == "Some graduate school":
    education = 7
else:
    education = 8

######## Example 2 style: sliders

income = st.sidebar.slider(
    label="Income (1=lowest, 9=highest)",
    min_value=1,
    max_value=9,
    value=7
)

age = st.sidebar.slider(
    label="Age",
    min_value=18,
    max_value=97,
    value=42
)

######## Example 3 style: number_input + labels (parent / married / female)

with st.sidebar:
    parent = st.number_input("Parent (0=no, 1=yes)", 0, 1, value=0)
    married = st.number_input("Married (0=no, 1=yes)", 0, 1, value=1)
    female = st.number_input("Female (0=no, 1=yes)", 0, 1, value=1)

# Create human-readable labels
# Income bracket label
if income <= 3:
    inc_label = "low income"
elif 3 < income < 7:
    inc_label = "middle income"
else:
    inc_label = "high income"

# Degree label (college grad = education >= 6)
if education >= 6:
    deg_label = "college graduate"
else:
    deg_label = "non-college graduate"

# Marital label
mar_label = "married" if married == 1 else "non-married"

# Gender label
gender_label = "female" if female == 1 else "not female"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Click to get prediction")
run_prediction = st.sidebar.button("Predict LinkedIn usage")


# ---------------------------------------------------
# Build user row for the model
# ---------------------------------------------------

user_df = pd.DataFrame({
    "income":   [income],
    "education":[education],
    "parent":   [parent],
    "married":  [married],
    "female":   [female],
    "age":      [age]
})


# ---------------------------------------------------
# Main panel: prediction + visuals
# ---------------------------------------------------

st.markdown("## 🔍 Live prediction")

# Natural-language summary (like Example 3)
st.write(
    f"This person is **{mar_label}**, **{deg_label}**, "
    f"**{gender_label}**, and in a **{inc_label}** bracket, age **{age}**."
)

if run_prediction:
    # Predict probability and class
    proba = model.predict_proba(user_df)[0, 1]
    pred_class = int(proba >= 0.5)

    label = "LinkedIn user ✅" if pred_class == 1 else "Not a LinkedIn user ❌"

    st.markdown(f"**Predicted class:** {label}")
    st.markdown(f"**Estimated probability of LinkedIn use:** `{proba:.3f}`")

    # Probability bar chart
    prob_df = pd.DataFrame(
        {"Outcome": ["LinkedIn user", "Not a LinkedIn user"],
         "Probability": [proba, 1 - proba]}
    )

    prob_chart = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x="Outcome",
            y="Probability",
            tooltip=["Outcome", alt.Tooltip("Probability", format=".3f")]
        )
    )

    st.altair_chart(prob_chart, use_container_width=True)

else:
    st.info("Use the sidebar to set the profile and click **Predict LinkedIn usage**.")


# ---------------------------------------------------
# Extra: show model accuracy for marketing team
# ---------------------------------------------------

st.markdown("---")
st.markdown("## 🧪 Model quality for the marketing team")

st.write(
    f"The logistic regression model achieves a test set accuracy of "
    f"**{test_acc:.3f}** when predicting LinkedIn usage from income, education, "
    "parent status, marital status, gender, and age."
)
st.caption(
    "Higher accuracy means the model's predictions are more reliable for targeting campaigns."
)