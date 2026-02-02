import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="CrediX — Loan Decision Support",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------
# Styling (light, professional)
# ---------------------------
st.markdown(
    """
<style>
:root { --card-bg: #ffffff; --text: #111827; --muted: #6b7280; --line: #e5e7eb; }
html, body, [class*="css"]  { color: var(--text); }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
hr { border: none; border-top: 1px solid var(--line); margin: 1rem 0 1.5rem 0; }

.hero {
  background: linear-gradient(135deg,#0f172a 0%,#1e293b 55%,#0f172a 100%);
  border-radius: 16px;
  padding: 26px 26px;
  color: #ffffff;
  box-shadow: 0 14px 34px rgba(0,0,0,0.18);
  margin-bottom: 18px;
}

.hero h1 {
  font-size: 38px;
  margin: 0;
  font-weight: 900;
  letter-spacing: 0.4px;
  color: #ffffff !important;
  text-shadow: 0 2px 6px rgba(0,0,0,0.6),0 0 18px rgba(255,255,255,0.15);
}

.hero p {
  margin: 8px 0 0 0;
  color: rgba(255,255,255,0.92);
  line-height: 1.45;
  font-size: 15px;
  text-shadow: 0 1px 6px rgba(0,0,0,0.35);
}

.card {
  background: var(--card-bg);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 10px 24px rgba(17,24,39,0.06);
}
.card h3 { margin: 0 0 10px 0; font-size: 18px; }
.caption { color: var(--muted); font-size: 0.92rem; margin-top: 2px; }

.badge {
  display: inline-block;
  padding: 10px 14px;
  border-radius: 999px;
  font-weight: 800;
  letter-spacing: 0.8px;
  font-size: 14px;
  color: #ffffff;
}

.small-metric {
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px 14px;
  background: #fbfdff;
}
/* Force readable text inside our custom metric cards */
.small-metric { color: #111827 !important; }
.small-metric .k { color: #6b7280 !important; }
.small-metric .v { color: #111827 !important; }
.small-metric p { color: inherit !important; opacity: 1 !important; }
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Load artifacts (XGBoost pipeline)
# ---------------------------
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("artifacts/xgb_pipeline.joblib")
    feature_cols = joblib.load("artifacts/feature_columns.joblib")
    return pipeline, feature_cols

pipeline, feature_cols = load_artifacts()

# ---------------------------
# Labels / options
# ---------------------------
EDU_OPTIONS = ["Graduate", "Not Graduate"]
SE_OPTIONS = ["Yes", "No"]

LABELS = {
    "no_of_dependents": "Number of Dependents",
    "education": "Education Level",
    "self_employed": "Self-employed Status",
    "income_annum": "Annual Income (AUD)",
    "loan_amount": "Requested Loan Amount (AUD)",
    "loan_term": "Loan Term (Months)",
    "cibil_score": "Credit Score (CIBIL)",
    "residential_assets_value": "Residential Assets Value (AUD)",
    "commercial_assets_value": "Commercial Assets Value (AUD)",
    "luxury_assets_value": "Luxury Assets Value (AUD)",
    "bank_asset_value": "Bank Asset Value (AUD)",
}

PRETTY = {
    "no_of_dependents": "Dependents",
    "education": "Education",
    "self_employed": "Self-employed",
    "income_annum": "Annual Income",
    "loan_amount": "Loan Amount",
    "loan_term": "Loan Term (Months)",
    "cibil_score": "Credit Score",
    "residential_assets_value": "Residential Assets",
    "commercial_assets_value": "Commercial Assets",
    "luxury_assets_value": "Luxury Assets",
    "bank_asset_value": "Bank Assets",
}

# ---------------------------
# Helper functions
# ---------------------------
def get_step(p, candidates):
    """Return the first matching named step from a Pipeline, else None."""
    for k in candidates:
        if hasattr(p, "named_steps") and k in p.named_steps:
            return p.named_steps[k]
    return None

def badge_html(label: str, is_positive: bool) -> str:
    color = "#16a34a" if is_positive else "#dc2626"
    return f'<span class="badge" style="background:{color};">{label}</span>'

def metric_box(title: str, value: str) -> str:
    return f"""
<div class="small-metric" style="color:#111827 !important;">
  <p class="k" style="color:#6b7280 !important; margin:0;">{title}</p>
  <p class="v" style="color:#111827 !important; margin:2px 0 0 0;">{value}</p>
</div>
"""

def prettify_feature_name(name: str) -> str:
    # Handles ColumnTransformer names like num__income_annum / cat__education_Graduate
    if "__" in name:
        _, rest = name.split("__", 1)
        if "_" in rest:
            base = rest.split("_", 1)[0]
            if base in PRETTY:
                base, level = rest.split("_", 1)
                return f"{PRETTY.get(base, base)} = {level}"
        return PRETTY.get(rest, rest)
    return PRETTY.get(name, name)

@st.cache_resource
def get_shap_explainer(_pipeline):
    """
    Cache SHAP explainer.
    IMPORTANT: _pipeline starts with underscore so Streamlit does NOT hash it.
    """
    model = get_step(_pipeline, ["model", "classifier", "xgb", "xgboost"])
    if model is None:
        if hasattr(_pipeline, "steps") and len(_pipeline.steps) > 0:
            model = _pipeline.steps[-1][1]
        else:
            raise ValueError("Could not locate model step in the pipeline.")

    return shap.TreeExplainer(model)

def shap_explain_text(_pipeline, X_df: pd.DataFrame, direction="Approved", top_k=6):
    preprocess = get_step(_pipeline, ["preprocess", "preprocessor", "prep", "transform"])
    if preprocess is None:
        if hasattr(_pipeline, "steps") and len(_pipeline.steps) >= 2:
            preprocess = _pipeline.steps[0][1]
        else:
            raise ValueError("Could not locate preprocessing step in the pipeline.")

    X_trans = preprocess.transform(X_df)

    try:
        feat_names = preprocess.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(X_trans.shape[1])], dtype=object)

    explainer = get_shap_explainer(_pipeline)
    exp = explainer(X_trans)  # shap.Explanation

    values = exp.values
    is_3d = isinstance(values, np.ndarray) and values.ndim == 3

    # UI mapping stated: 0 = Approved, 1 = Rejected
    cls = 0 if direction == "Approved" else 1

    if is_3d:
        sv_row = values[0, :, cls]
    else:
        sv_row = values[0]

    sv_row = np.asarray(sv_row).reshape(-1).astype(float)

    top_idx = np.argsort(np.abs(sv_row))[::-1][:top_k]
    top_features = [prettify_feature_name(str(feat_names[i])) for i in top_idx]
    top_vals = [float(sv_row[i]) for i in top_idx]

    table = pd.DataFrame({"Feature": top_features, "SHAP Impact": top_vals})

    positives = [(f, v) for f, v in zip(top_features, top_vals) if v > 0]
    negatives = [(f, v) for f, v in zip(top_features, top_vals) if v < 0]
    other = "Rejected" if direction == "Approved" else "Approved"

    lines = []
    lines.append(f"**Explanation orientation:** {direction}")
    lines.append("")
    if positives:
        lines.append(f"✅ **Top factors supporting {direction}:**")
        for f, _ in positives[:3]:
            lines.append(f"- {f}")
    else:
        lines.append(f"✅ **Top factors supporting {direction}:** (none among the top factors)")
    lines.append("")
    if negatives:
        lines.append(f"⚠️ **Top factors pushing toward {other}:**")
        for f, _ in negatives[:3]:
            lines.append(f"- {f}")
    else:
        lines.append(f"⚠️ **Top factors pushing toward {other}:** (none among the top factors)")

    return table, "\n".join(lines)

def map_probabilities(_pipeline, proba_vec):
    """
    Robustly map proba to P(Approved) and P(Rejected) using model.classes_ when available.
    Approved = 0, Rejected = 1.
    """
    model = get_step(_pipeline, ["model", "classifier", "xgb", "xgboost"])
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(_pipeline, "classes_"):
        classes = _pipeline.classes_

    proba_vec = np.asarray(proba_vec).reshape(-1)

    # Default fallback: assume [Approved, Rejected]
    p_app = float(proba_vec[0]) if len(proba_vec) > 0 else 0.0
    p_rej = float(proba_vec[1]) if len(proba_vec) > 1 else 0.0

    if classes is not None:
        classes_list = list(classes)
        if 0 in classes_list:
            p_app = float(proba_vec[classes_list.index(0)])
        if 1 in classes_list:
            p_rej = float(proba_vec[classes_list.index(1)])

    return p_app, p_rej

# ---------------------------
# Session state
# ---------------------------
if "has_prediction" not in st.session_state:
    st.session_state.has_prediction = False
if "pred_result" not in st.session_state:
    st.session_state.pred_result = {}
if "last_input_df" not in st.session_state:
    st.session_state.last_input_df = None
if "direction" not in st.session_state:
    st.session_state.direction = "Approved"

# ---------------------------
# Header / Hero
# ---------------------------
st.markdown(
    """
<div class="hero">
  <h1>CrediX — Loan Decision Support</h1>
  <p>
    A lightweight decision-support prototype using a trained XGBoost pipeline and explainable predictions (SHAP).
    Use this tool to explore decisions and key contributing factors.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("Model & notes", expanded=False):
    st.write("- Final model: XGBoost pipeline")
    st.write("- Mapping: 0 = Approved, 1 = Rejected")
    st.write("- SHAP explanations: top features by absolute SHAP impact")
    st.caption("This is a decision-support tool. Final decisions may require additional policy and compliance checks.")

# ---------------------------
# Input + Output layout
# ---------------------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown(
        '<div class="card"><h3>Applicant details</h3><p class="caption">Enter values, then run a prediction.</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    with st.form("loan_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            no_of_dependents = st.number_input(LABELS["no_of_dependents"], min_value=0, value=0, step=1)
            education = st.selectbox(LABELS["education"], EDU_OPTIONS)
            self_employed = st.selectbox(LABELS["self_employed"], SE_OPTIONS)

        with c2:
            income_annum = st.number_input(LABELS["income_annum"], min_value=0, value=500000, step=50000)
            loan_amount = st.number_input(LABELS["loan_amount"], min_value=0, value=1000000, step=50000)
            loan_term = st.number_input(LABELS["loan_term"], min_value=1, value=12, step=1)

        with c3:
            cibil_score = st.number_input(LABELS["cibil_score"], min_value=0, value=700, step=1)
            residential_assets_value = st.number_input(LABELS["residential_assets_value"], min_value=0, value=0, step=50000)
            commercial_assets_value = st.number_input(LABELS["commercial_assets_value"], min_value=0, value=0, step=50000)
            luxury_assets_value = st.number_input(LABELS["luxury_assets_value"], min_value=0, value=0, step=50000)
            bank_asset_value = st.number_input(LABELS["bank_asset_value"], min_value=0, value=0, step=50000)

        col_btn1, col_btn2 = st.columns([0.25, 0.75])
        with col_btn1:
            submitted = st.form_submit_button("Predict")
        with col_btn2:
            st.caption("Tip: Use realistic values to see more meaningful explanations.")

    # Build input dataframe (keep training column order)
    user = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }
    input_df = pd.DataFrame([user])[feature_cols]

    if submitted:
        try:
            pred = int(pipeline.predict(input_df)[0])
            proba_vec = pipeline.predict_proba(input_df)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        proba_app, proba_rej = map_probabilities(pipeline, proba_vec)

        st.session_state.has_prediction = True
        st.session_state.last_input_df = input_df.copy()
        st.session_state.pred_result = {
            "pred": pred,
            "proba_app": float(proba_app),
            "proba_rej": float(proba_rej),
        }

    st.write("")
    st.markdown(
        '<div class="card"><h3>Input summary</h3><p class="caption">Review the submitted inputs.</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    display_df = input_df.rename(columns={k: PRETTY.get(k, k) for k in input_df.columns})
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with right:
    st.markdown(
        '<div class="card"><h3>Decision & confidence</h3><p class="caption">Prediction results and probabilities.</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if st.session_state.has_prediction and st.session_state.last_input_df is not None:
        pred = int(st.session_state.pred_result["pred"])
        proba_app = float(st.session_state.pred_result["proba_app"])
        proba_rej = float(st.session_state.pred_result["proba_rej"])

        approved = pred == 0
        st.markdown(badge_html("APPROVED" if approved else "REJECTED", approved), unsafe_allow_html=True)
        st.write("")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(metric_box("P(Approved)", f"{proba_app*100:.2f}%"), unsafe_allow_html=True)
            st.progress(int(round(proba_app * 100)))
        with m2:
            st.markdown(metric_box("P(Rejected)", f"{proba_rej*100:.2f}%"), unsafe_allow_html=True)
            st.progress(int(round(proba_rej * 100)))

        st.write("")
        st.markdown(
            '<div class="card"><h3>Explainability (SHAP)</h3><p class="caption">Top contributing factors for the selected outcome direction.</p></div>',
            unsafe_allow_html=True,
        )
        st.write("")

        direction = st.radio(
            "Explain toward:",
            options=["Approved", "Rejected"],
            horizontal=True,
            key="direction",
        )

        with st.spinner("Computing explanation..."):
            table, explanation_text = shap_explain_text(
                pipeline,
                st.session_state.last_input_df,
                direction=direction,
                top_k=6,
            )

        st.markdown(explanation_text)
        st.write("")
        st.dataframe(
            table.assign(**{"|Impact|": table["SHAP Impact"].abs()})
            .sort_values("|Impact|", ascending=False)
            .drop(columns=["|Impact|"]),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("Run a prediction to see the decision and SHAP explanation.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption(
    "Decision-support prototype. Predictions depend on training data and assumptions; use with appropriate governance and policy checks."
)
