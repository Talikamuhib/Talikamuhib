"""
ADHD Appointment No-Show (DNA) Prediction — Streamlit App
=========================================================

An interactive machine-learning app that predicts the risk that a referral will
result in a "Did Not Attend" (DNA) appointment across an ADHD care pathway.

Story: the Tableau dashboard surfaced a ~17% DNA rate. This app turns that
insight into action — predicting *which* referrals are most at risk so a service
can target reminders and engagement interventions.

Data: fully synthetic (data/adhd_pathway_data.csv). Not real patients.
For portfolio / methodology demonstration only — not for clinical use.

Run:
    pip install -r requirements-app.txt
    streamlit run apps/adhd_noshow_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------ config
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "adhd_pathway_data.csv"

# NOTE: engagement_score is intentionally excluded — in the source data it is
# derived from the appointment outcome (target leakage) and is not knowable
# before the appointment. We only use features available at referral time.
CATEGORICAL = ["referral_source", "clinic", "priority", "age_band", "gender"]
NUMERIC = ["wait_days"]
TARGET = "is_dna"

ACCENT = "#2dd4bf"

st.set_page_config(
    page_title="ADHD No-Show Prediction",
    page_icon="🧠",
    layout="wide",
)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Binary target: did the patient not attend?
    df[TARGET] = (df["appointment_outcome"] == "DNA").astype(int)
    return df


@st.cache_resource(show_spinner=False)
def train_model(model_name: str):
    df = load_data()
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )

    if model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:  # Logistic Regression
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
    }
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
    cm = confusion_matrix(y_test, preds)
    report = classification_report(
        y_test, preds, target_names=["Attend", "DNA"], output_dict=True
    )

    # Feature importance (works for both, via a fitted model on encoded names)
    feat_names = pipe.named_steps["pre"].get_feature_names_out()
    if model_name == "Random Forest":
        importances = pipe.named_steps["clf"].feature_importances_
    else:
        importances = np.abs(pipe.named_steps["clf"].coef_[0])
    imp_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )
    imp_df["feature"] = imp_df["feature"].str.replace("cat__", "").str.replace(
        "num__", ""
    )

    return pipe, metrics, roc_df, cm, report, imp_df


# ------------------------------------------------------------------ UI
df = load_data()

st.title("🧠 ADHD Appointment No-Show (DNA) Prediction")
st.markdown(
    "Predicting which referrals are most likely to **Did Not Attend (DNA)** "
    "across an ADHD care pathway — turning dashboard insight into targeted action."
)
st.caption(
    "Synthetic demonstration data (1,200 referrals). "
    "For portfolio / methodology demonstration only — not for clinical use."
)

with st.sidebar:
    st.header("⚙️ Model")
    model_name = st.radio(
        "Algorithm", ["Logistic Regression", "Random Forest"], index=0
    )
    st.divider()
    st.markdown(
        "**Why this project?**\n\n"
        "The Tableau dashboard found a ~17% DNA rate. This model predicts *who* "
        "is at risk so reminders and engagement can be targeted — reducing wasted "
        "clinical capacity."
    )

pipe, metrics, roc_df, cm, report, imp_df = train_model(model_name)

# ---- KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
c2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
c3.metric("Precision (DNA)", f"{metrics['precision']:.1%}")
c4.metric("Recall (DNA)", f"{metrics['recall']:.1%}")

st.divider()

tab_predict, tab_perf, tab_data = st.tabs(
    ["🔮 Predict a referral", "📊 Model performance", "🗂️ Data overview"]
)

# ---- Predict tab
with tab_predict:
    st.subheader("Estimate DNA risk for a new referral")
    col_a, col_b = st.columns(2)
    with col_a:
        referral_source = st.selectbox(
            "Referral source", sorted(df["referral_source"].unique())
        )
        clinic = st.selectbox("Clinic", sorted(df["clinic"].unique()))
        priority = st.selectbox("Priority", sorted(df["priority"].unique()))
    with col_b:
        age_band = st.selectbox("Age band", sorted(df["age_band"].unique()))
        gender = st.selectbox("Gender", sorted(df["gender"].unique()))
        wait_days = st.slider("Wait days (referral → appointment)", 3, 250, 84)

    input_df = pd.DataFrame(
        [
            {
                "referral_source": referral_source,
                "clinic": clinic,
                "priority": priority,
                "age_band": age_band,
                "gender": gender,
                "wait_days": wait_days,
            }
        ]
    )
    risk = pipe.predict_proba(input_df)[0, 1]

    st.markdown("### Predicted DNA risk")
    st.progress(min(int(risk * 100), 100))
    if risk >= 0.30:
        st.error(f"⚠️ High risk — {risk:.0%}. Recommend proactive reminder / outreach.")
    elif risk >= 0.18:
        st.warning(f"Moderate risk — {risk:.0%}. Standard reminder advised.")
    else:
        st.success(f"Low risk — {risk:.0%}.")

    st.caption(
        "Risk is a probability from the trained model. Thresholds are illustrative "
        "and would be tuned with a service in practice."
    )

# ---- Performance tab
with tab_perf:
    left, right = st.columns(2)
    with left:
        st.subheader("ROC curve")
        st.line_chart(roc_df, x="False Positive Rate", y="True Positive Rate")
        st.caption(f"ROC-AUC = {metrics['roc_auc']:.3f}")
    with right:
        st.subheader("Top predictive features")
        st.bar_chart(imp_df, x="feature", y="importance", horizontal=True)

    st.subheader("Confusion matrix (test set)")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Attend", "Actual: DNA"],
        columns=["Predicted: Attend", "Predicted: DNA"],
    )
    st.dataframe(cm_df, width="stretch")

    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")

# ---- Data tab
with tab_data:
    st.subheader("Outcome distribution")
    st.bar_chart(df["appointment_outcome"].value_counts())
    st.subheader("DNA rate by priority")
    dna_by_priority = df.groupby("priority")[TARGET].mean().round(3)
    st.bar_chart(dna_by_priority)
    st.subheader("Sample of the data")
    st.dataframe(df.head(50), width="stretch")

st.divider()
st.caption(
    "Built by Taliqa Muhib · Python · Scikit-learn · Streamlit · "
    "Synthetic data, not for clinical use."
)
