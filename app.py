import pandas as pd
import streamlit as st

# -------------------------
# 1) Page config (MUST be first Streamlit call)
# -------------------------
st.set_page_config(
    page_title="Diagnosis Assistant",
    page_icon="🩺",
    layout="centered"
)

# -------------------------
# 2) Load dataset (cached)
# -------------------------
@st.cache_data(ttl=60 * 60 * 24)
def load_mapping():
    """
    Loads symptom → disease associations from HSDN Combined-Output.tsv.
    'weight' is PubMed co-occurrence count (association), NOT diagnosis probability.
    """
    url = "https://raw.githubusercontent.com/LeoBman/HSDN/master/Combined-Output.tsv"
    df = pd.read_csv(url, sep="\t", header=0)

    df = df.rename(columns={
        "MeSH Symptom Term": "symptom",
        "MeSH Disease Term": "disease",
        "PubMed occurrence": "weight",
    })

    df = df[["symptom", "disease", "weight"]].copy()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)

    df["symptom"] = df["symptom"].astype(str).str.strip()
    df["disease"] = df["disease"].astype(str).str.strip()
    df = df[(df["symptom"] != "") & (df["disease"] != "")]
    return df

# -------------------------
# 3) Disease → specialist mapping & constants
# -------------------------
disease_to_specialist = {
    "Common Cold": "General Physician",
    "Influenza": "General Physician / Infectious Disease Specialist",
    "Sinusitis": "ENT Specialist",
    "Asthma": "Pulmonologist",
    "Bronchitis": "Pulmonologist / General Physician",
    "Migraine Disorders": "Neurologist",
    "Gastroesophageal Reflux": "General Physician / Gastroenterologist",
    "Gastroenteritis": "General Physician / Gastroenterologist",
    "Dengue": "Internal Medicine Specialist",
    "Malaria": "Infectious Disease Specialist",
    "Chickenpox": "General Physician / Dermatologist",
    "Pneumonia": "Pulmonologist",
    "COVID-19": "Pulmonologist / General Physician",
    "Hypertension": "Cardiologist",
    "Diabetes Mellitus": "Endocrinologist",
    "Brain Diseases": "Neurologist",
    "Brain Neoplasms": "Neurologist / Neurosurgeon / Oncologist",
}

# List of common conditions for filtering
COMMON_CONDITIONS = list(disease_to_specialist.keys())

# -------------------------
# 4) Load data
# -------------------------
try:
    mapping_df = load_mapping()
except Exception as e:
    st.error("Dataset could not be loaded. Check your internet connection and try again.")
    st.code(str(e))
    st.stop()

symptoms = sorted(mapping_df["symptom"].unique().tolist())
symptom_set = set(symptoms)

# Disease popularity across the dataset
disease_popularity = mapping_df.groupby("disease")["symptom"].nunique()

# -------------------------
# 5) UI
# -------------------------
st.title("🩺 Diagnosis Assistant")
st.write(
    "Select symptoms and get **possible associated conditions** based on biomedical literature. "
   
)

# -------------------------
# 6) Sidebar settings
# -------------------------
st.sidebar.header("Settings")

top_k = st.sidebar.slider("Top results", 3, 15, 5, 1)

# Added missing min_weight slider
min_weight = st.sidebar.slider("Minimum PubMed occurrence (weight)", 0, 100, 0, 5)

# Require stronger evidence
min_coverage = st.sidebar.slider("Minimum symptom match (coverage)", 1, 5, 2, 1)
min_score = st.sidebar.slider("Minimum raw score", 0, 5000, 200, 50)

# Penalize too-general diseases
use_popularity_penalty = st.sidebar.checkbox("Penalize very general diseases", value=True)
beta = st.sidebar.slider("Penalty strength (beta)", 0.0, 2.0, 1.0, 0.1)
alpha = st.sidebar.slider("Coverage boost (alpha)", 1.0, 2.5, 1.5, 0.1)

# Added missing toggle switches for filtering modes
common_only = st.sidebar.checkbox("Show common conditions only", value=False)
demo_mode = st.sidebar.checkbox("Enable safe demo mode (filter serious terms)", value=True)
show_evidence = st.sidebar.checkbox("Show raw literature evidence tables", value=False)

# -------------------------
# 7) Symptom selection
# -------------------------
selected_symptoms = st.multiselect(
    "Which symptoms are you experiencing? Select all that apply:",
    options=symptoms
)

# -------------------------
# 8) Suggestion logic
# -------------------------
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom!")
        st.stop()

    subset = mapping_df[mapping_df["symptom"].isin(selected_symptoms)].copy()

    if min_weight > 0:
        subset = subset[subset["weight"] >= min_weight]

    if subset.empty:
        st.warning("No matching entries found. Try lowering the minimum weight.")
        st.stop()
        
    # Base scores
    sum_scores = subset.groupby("disease")["weight"].sum()

    # Coverage = how many selected symptoms each disease matches
    coverage = subset.groupby("disease")["symptom"].nunique()

    # Combine: total weight * coverage boost
    final_scores = (sum_scores * (coverage ** float(alpha))).sort_values(ascending=False)

    # Remove "diseases" that are actually symptom terms
    final_scores = final_scores[~final_scores.index.isin(symptom_set)]
    final_scores = final_scores[~final_scores.index.isin(selected_symptoms)]

    # Require stronger evidence: filter by coverage + min_score
    cov_aligned = coverage.reindex(final_scores.index).fillna(0).astype(int)
    final_scores = final_scores[cov_aligned >= int(min_coverage)]
    final_scores = final_scores[final_scores >= float(min_score)]

    # Penalize too-general diseases
    if use_popularity_penalty and len(final_scores) > 0:
        pop = disease_popularity.reindex(final_scores.index).fillna(1).astype(float)
        penalized = final_scores / (pop ** float(beta))
        final_scores = penalized.sort_values(ascending=False)

    # Common conditions only
    if common_only and len(final_scores) > 0:
        final_scores = final_scores[final_scores.index.isin(COMMON_CONDITIONS)]

    # Demo-mode: hide rare/serious terms
    if demo_mode and len(final_scores) > 0:
        block_keywords = [
            "neoplasm", "cancer", "tumor", "malignant", "carcinoma",
            "leukemia", "lymphoma", "sarcoma", "metast", "oncolog",
        ]
        patt = "|".join(block_keywords)
        final_scores = final_scores[~final_scores.index.str.lower().str.contains(patt)]

    if final_scores.empty:
        st.warning(
            "No results after filters.\n\n"
            "Try: lower Minimum raw score, reduce Minimum coverage, or disable Common-only/Demo mode."
        )
        st.stop()
        
    top = final_scores.head(top_k)

    # Normalize for display
    max_score = float(top.max()) if len(top) else 1.0
    if max_score <= 0:
        max_score = 1.0

    st.subheader("Suggested associated conditions (literature-based)")
    for disease, score in top.items():
        norm = float(score) / max_score
        cov = int(coverage.get(disease, 0))
        doctor = disease_to_specialist.get(disease, "General Physician")

        st.markdown(
            f"**• {disease}** \n"
            f"Matched symptoms: `{cov}/{len(selected_symptoms)}` \n"
            f"Suggested doctor: **{doctor}**"
        )

    st.divider()

    if show_evidence:
        st.subheader("Evidence (rows used for scoring)")
        evidence = subset[subset["disease"].isin(top.index)].copy()
        evidence = evidence.sort_values(["disease", "weight"], ascending=[True, False])
        st.dataframe(evidence.head(100), use_container_width=True, hide_index=True)

    st.divider()
    st.info("Disclaimer: Consult a qualified doctor for medical advice.")
