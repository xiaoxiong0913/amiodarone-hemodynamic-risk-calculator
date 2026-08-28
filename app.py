from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


FEATURES = (
    "lactate_last_24h",
    "invasive_ventilation_at_t0",
    "map_mean_6h",
    "urine_output_6h_ml",
    "prior_vasoactive_6to1h",
    "bilirubin_total_last_24h",
    "map_sd_6h",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_model():
    assets = Path(__file__).resolve().parent / "model"
    model_path = assets / "amiodarone_xgboost_model.joblib"
    metadata = json.loads((assets / "model_metadata.json").read_text(encoding="utf-8"))
    if metadata["model_sha256"] != sha256_file(model_path):
        raise RuntimeError("Model checksum mismatch")
    if tuple(metadata["features"]) != FEATURES:
        raise RuntimeError("Feature contract mismatch")
    packaged = joblib.load(model_path)
    if packaged.get("model_key") != "xgboost" or tuple(packaged.get("features", ())) != FEATURES:
        raise RuntimeError("Frozen model contract mismatch")
    pipeline = packaged["pipeline"]
    if tuple(getattr(pipeline, "feature_names_in_", ())) != FEATURES:
        raise RuntimeError("Pipeline feature order mismatch")
    return pipeline, metadata


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_verified_model()


def case_frame(
    lactate: float | None,
    invasive_ventilation: bool,
    mean_map: float,
    urine_output: float | None,
    prior_vasoactive: bool,
    bilirubin: float | None,
    map_variability: float | None,
) -> pd.DataFrame:
    values = [[
        np.nan if lactate is None else float(lactate),
        int(invasive_ventilation),
        float(mean_map),
        np.nan if urine_output is None else float(urine_output),
        int(prior_vasoactive),
        np.nan if bilirubin is None else float(bilirubin),
        np.nan if map_variability is None else float(map_variability),
    ]]
    return pd.DataFrame(values, columns=list(FEATURES))


def predict_probability(model, **values) -> float:
    return float(model.predict_proba(case_frame(**values))[:, 1][0])


def classify_risk(probability: float, threshold: float) -> str:
    return "High risk" if probability >= threshold else "Low risk"


def optional_number(label: str, key: str, minimum: float, maximum: float, default: float, step: float, missing_label: str):
    left, right = st.columns((3.2, 1.0), vertical_alignment="bottom")
    missing = right.checkbox(missing_label, key=f"{key}_missing")
    value = left.number_input(label, min_value=minimum, max_value=maximum, value=default, step=step, key=key, disabled=missing)
    return None if missing else float(value)


def main() -> None:
    st.set_page_config(page_title="Amiodarone hemodynamic risk calculator", page_icon="♥", layout="wide")
    try:
        model, metadata = cached_model()
    except Exception as exc:
        st.error(f"Model verification failed: {exc}")
        st.stop()

    st.title("Early hemodynamic intolerance after intravenous amiodarone")
    st.write("Enter seven variables available at intravenous amiodarone initiation to estimate the 2-hour risk.")
    entry, result = st.columns((1.25, 0.75), gap="large")
    with entry:
        with st.form("amiodarone_risk_form"):
            lactate = optional_number("Lactate, mmol/L", "lactate", 0.1, 30.0, 2.0, 0.1, "Not measured")
            invasive = st.selectbox("Invasive mechanical ventilation at initiation", ("No", "Yes")) == "Yes"
            mean_map = st.number_input("Mean MAP during the preceding 6 hours, mmHg", min_value=40.0, max_value=140.0, value=75.0, step=1.0)
            urine = optional_number("Urine output during the preceding 6 hours, mL", "urine", 0.0, 10000.0, 300.0, 10.0, "Missing")
            vasoactive = st.selectbox("Vasoactive drug exposure 6–1 hours before initiation", ("No", "Yes")) == "Yes"
            bilirubin = optional_number("Total bilirubin, mg/dL", "bilirubin", 0.0, 50.0, 0.8, 0.1, "Not measured")
            map_sd = optional_number("MAP standard deviation during the preceding 6 hours, mmHg", "map_sd", 0.0, 50.0, 8.0, 0.5, "Missing")
            calculate = st.form_submit_button("Calculate predicted risk", type="primary", use_container_width=True)

    with result:
        st.subheader("Estimated risk")
        if calculate:
            probability = predict_probability(
                model,
                lactate=lactate,
                invasive_ventilation=invasive,
                mean_map=mean_map,
                urine_output=urine,
                prior_vasoactive=vasoactive,
                bilirubin=bilirubin,
                map_variability=map_sd,
            )
            threshold = float(metadata["threshold"])
            st.metric("Predicted 2-hour risk", f"{probability:.1%}")
            if classify_risk(probability, threshold) == "High risk":
                st.error(f"High-risk group (threshold: {threshold:.1%})")
            else:
                st.success(f"Low-risk group (threshold: {threshold:.1%})")
        else:
            st.info("Enter patient data and select Calculate predicted risk.")
        st.caption(f"Risk-classification threshold: {float(metadata['threshold']):.1%}")
        st.markdown("**Observed event rates at this threshold**")
        st.dataframe(pd.DataFrame(metadata["observed_event_rates"]), hide_index=True, use_container_width=True)

    st.divider()
    st.caption("Model developed in MIMIC-IV and externally validated in eICU. Use the estimate to support individualized monitoring at treatment initiation.")


if __name__ == "__main__":
    main()
