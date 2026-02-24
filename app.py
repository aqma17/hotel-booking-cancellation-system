import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG (LOCAL FILES)
# =========================
MODEL_FILE = "best_model_tuned.pkl"
FEATURE_FILE = "Exp4_feature_cols.json"

# Simple UI dropdown options (safe defaults)
CATEGORICAL_OPTIONS = {
    "deposit_type": ["No Deposit", "Non Refund", "Refundable"],
    "market_segment": ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Groups", "Complementary", "Aviation"],
    "arrival_date_month": ["January","February","March","April","May","June","July","August","September","October","November","December"],
    "country": ["PRT", "GBR", "ESP", "FRA", "DEU", "IRL", "ITA", "NLD", "BEL", "BRA", "USA", "Other"],
    "reserved_room_type": ["A","B","C","D","E","F","G","H","L","P"],
    "assigned_room_type": ["A","B","C","D","E","F","G","H","I","K","L","P"],
    "customer_type": ["Transient", "Transient-Party", "Contract", "Group"],
    "distribution_channel": ["Direct", "Corporate", "TA/TO", "GDS"],
}

NUMERIC_HINTS = {
    "total_of_special_requests": (0, 10, 0),
    "arrival_date_week_number": (1, 53, 1),
    "arrival_date_day_of_month": (1, 31, 1),
    "res_status_year": (2014, 2026, 2017),
}

THRESHOLD_HIGH = 0.30     # high-risk threshold (30%)
BATCH_LIMIT = 200         # max rows for batch


# =========================
# HELPERS
# =========================
@st.cache_resource
def load_assets():
    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"Missing {FEATURE_FILE}. Put it in the same folder as app.py")

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Missing {MODEL_FILE}. Put it in the same folder as app.py")

    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    model = joblib.load(MODEL_FILE)
    return model, feature_cols


def align_input_df(df_in: pd.DataFrame, feature_cols: list):
    df = df_in.copy()

    # add missing columns as NaN
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan

    # drop extra columns
    extras = [c for c in df.columns if c not in feature_cols]
    if extras:
        df = df.drop(columns=extras)

    # reorder
    df = df[feature_cols]
    return df, missing, extras


def predict_df(model, df_aligned: pd.DataFrame):
    pred = model.predict(df_aligned)

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_aligned)[:, 1]

    return pred, proba


def template_csv(feature_cols: list) -> bytes:
    return pd.DataFrame(columns=feature_cols).to_csv(index=False).encode("utf-8")


def render_value_input(feat: str):
    # categorical dropdown
    if feat in CATEGORICAL_OPTIONS:
        opts = CATEGORICAL_OPTIONS[feat]
        choice = st.selectbox(feat, ["(blank)"] + opts, index=0)
        if choice == "(blank)":
            return np.nan
        if choice == "Other":
            custom = st.text_input(f"{feat} (custom)", value="")
            return custom.strip() if custom.strip() else np.nan
        return choice

    # numeric widget
    if feat in NUMERIC_HINTS:
        mn, mx, default = NUMERIC_HINTS[feat]
        return st.number_input(feat, min_value=mn, max_value=mx, value=default, step=1)

    # fallback text
    raw = st.text_input(feat, value="")
    if raw.strip() == "":
        return np.nan
    try:
        return float(raw)
    except:
        return raw


# =========================
# PAGE
# =========================
st.set_page_config(page_title="Hotel Booking Cancellation Prediction", layout="wide")
st.title("🏨 Hotel Booking Cancellation Prediction System")
st.caption("Predict hotel booking cancellations (Cancel / Not Cancel) with probability output.")

try:
    model, FEATURE_COLS = load_assets()
except Exception as e:
    st.error(str(e))
    st.stop()

st.write(f"**Model file:** `{MODEL_FILE}`")
st.write(f"**Feature set:** {len(FEATURE_COLS)} features")

tab1, tab2 = st.tabs(["🧍 Individual Prediction", "📊 Batch Prediction"])

# =========================
# TAB 1: Individual
# =========================
with tab1:
    st.subheader("Individual Booking Prediction")

    mode = st.radio("Choose input mode:", ["Manual Form", "Upload 1-row CSV"], horizontal=True)

    if mode == "Manual Form":
        st.info("Fill in values. You may leave blanks if unknown.")

        row = {}
        cols_per_row = 3
        chunks = [FEATURE_COLS[i:i+cols_per_row] for i in range(0, len(FEATURE_COLS), cols_per_row)]

        for chunk in chunks:
            cols = st.columns(cols_per_row)
            for i, feat in enumerate(chunk):
                with cols[i]:
                    row[feat] = render_value_input(feat)

        if st.button("Predict (Individual) ✅"):
            df_one = pd.DataFrame([row])
            df_one, missing, extras = align_input_df(df_one, FEATURE_COLS)

            pred, proba = predict_df(model, df_one)
            label = "Canceled ❌" if int(pred[0]) == 1 else "Not Canceled ✅"

            st.write(f"Class Prediction: **{label}**")

            if proba is not None:
                prob = float(proba[0])
                prob_pct = prob * 100

                if prob >= THRESHOLD_HIGH:
                    st.error("🔴 High Risk Booking")
                else:
                    st.success("🟢 Low Risk Booking")

                st.write(f"Cancellation Probability: **{prob_pct:.2f}%**")
                st.caption(f"High Risk threshold = {THRESHOLD_HIGH*100:.0f}%")
            else:
                st.info("This model does not output probability (predict_proba not available).")

            with st.expander("Show aligned input row"):
                st.dataframe(df_one, use_container_width=True)

    else:
        st.info("Upload a CSV with **exactly 1 row**. Missing columns will be auto-added as NaN.")
        up = st.file_uploader("Upload 1-row CSV", type=["csv"])

        st.download_button(
            "Download Template CSV",
            data=template_csv(FEATURE_COLS),
            file_name="individual_template.csv",
            mime="text/csv"
        )

        if up is not None:
            df_up = pd.read_csv(up)

            if len(df_up) != 1:
                st.error(f"Your file has {len(df_up)} rows. Please upload exactly 1 row.")
            else:
                df_aligned, missing, extras = align_input_df(df_up, FEATURE_COLS)

                if missing:
                    st.warning(f"Missing columns auto-added as NaN: {missing}")
                if extras:
                    st.warning(f"Extra columns ignored: {extras}")

                if st.button("Predict (Uploaded) ✅"):
                    pred, proba = predict_df(model, df_aligned)
                    label = "Canceled ❌" if int(pred[0]) == 1 else "Not Canceled ✅"
                    st.write(f"Class Prediction: **{label}**")

                    if proba is not None:
                        prob = float(proba[0])
                        prob_pct = prob * 100

                        if prob >= THRESHOLD_HIGH:
                            st.error("🔴 High Risk Booking")
                        else:
                            st.success("🟢 Low Risk Booking")

                        st.write(f"Cancellation Probability: **{prob_pct:.2f}%**")
                    else:
                        st.info("No probability output (predict_proba not available).")

                    with st.expander("Show aligned row used for prediction"):
                        st.dataframe(df_aligned, use_container_width=True)


# =========================
# TAB 2: Batch
# =========================
with tab2:
    st.subheader("Batch Prediction (up to 200 rows)")
    st.info("Upload a CSV file. We will process the first 200 rows only.")

    up2 = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch")

    st.download_button(
        "Download Batch Template CSV",
        data=template_csv(FEATURE_COLS),
        file_name="batch_template.csv",
        mime="text/csv"
    )

    if "batch_out" not in st.session_state:
        st.session_state["batch_out"] = None

    if up2 is not None:
        df_up = pd.read_csv(up2)
        df_aligned, missing, extras = align_input_df(df_up, FEATURE_COLS)
        df_aligned = df_aligned.head(BATCH_LIMIT).copy()

        if missing:
            st.warning(f"Missing columns auto-added as NaN: {missing}")
        if extras:
            st.warning(f"Extra columns ignored: {extras}")

        if st.button("Predict (Batch) ✅"):
            pred, proba = predict_df(model, df_aligned)

            df_out = df_aligned.copy()
            df_out["predicted_cancel"] = pred.astype(int)

            if proba is not None:
                df_out["cancel_probability"] = proba.astype(float)
                df_out["cancel_probability_pct"] = (df_out["cancel_probability"] * 100).round(2)
                df_out["risk_label"] = np.where(
                    df_out["cancel_probability"] >= THRESHOLD_HIGH, "High Risk", "Low Risk"
                )
            else:
                df_out["cancel_probability"] = np.nan
                df_out["cancel_probability_pct"] = np.nan
                df_out["risk_label"] = "N/A"

            st.session_state["batch_out"] = df_out
            st.success(f"Batch prediction completed. Rows processed: {len(df_out)}")

    df_out = st.session_state.get("batch_out", None)

    if df_out is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total bookings", f"{len(df_out):,}")
        c2.metric("Predicted cancel rate", f"{df_out['predicted_cancel'].mean()*100:.2f}%")
        if df_out["cancel_probability"].notna().any():
            c3.metric("Avg cancel probability", f"{df_out['cancel_probability'].mean()*100:.2f}%")
        else:
            c3.metric("Avg cancel probability", "N/A")

        st.markdown("### Preview (Top 30)")
        st.dataframe(df_out.head(30), use_container_width=True)

        if df_out["cancel_probability"].notna().any():
            st.markdown("### Risk Distribution (Probability %)")
            bins = [0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
            labels = ["0–10%", "10–20%", "20–30%", "30–50%", "50–70%", "70–100%"]
            bucket = pd.cut(df_out["cancel_probability"], bins=bins, labels=labels, include_lowest=True)
            bucket_counts = bucket.value_counts().reindex(labels).fillna(0).astype(int)
            st.bar_chart(pd.DataFrame({"Bookings": bucket_counts.values}, index=labels), height=250)

            st.markdown("### Top 10 Highest-Risk Bookings")
            top_risk = df_out.sort_values("cancel_probability", ascending=False).head(10)
            st.dataframe(top_risk, use_container_width=True)

        st.download_button(
            "⬇ Download Predictions CSV",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )
