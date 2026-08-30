from __future__ import annotations

import hashlib
import html
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
    "hr_min_6h",
    "bilirubin_total_last_24h",
)

INPUT_RANGES = {
    "lactate_last_24h": (0.1, 30.0),
    "invasive_ventilation_at_t0": (0.0, 1.0),
    "map_mean_6h": (40.0, 140.0),
    "urine_output_6h_ml": (0.0, 10000.0),
    "hr_min_6h": (20.0, 220.0),
    "bilirubin_total_last_24h": (0.0, 50.0),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_model():
    assets = Path(__file__).resolve().parent / "model"
    model_path = assets / "amiodarone_random_forest_model.joblib"
    metadata = json.loads((assets / "model_metadata.json").read_text(encoding="utf-8"))
    if metadata["model_sha256"] != sha256_file(model_path):
        raise RuntimeError("Model checksum mismatch")
    if tuple(metadata["features"]) != FEATURES or metadata.get("model_key") != "random_forest":
        raise RuntimeError("Feature contract mismatch")
    packaged = joblib.load(model_path)
    if packaged.get("model_key") != metadata["model_key"] or tuple(packaged.get("features", ())) != FEATURES:
        raise RuntimeError("Model contract mismatch")
    pipeline = packaged["pipeline"]
    if tuple(getattr(pipeline, "feature_names_in_", ())) != FEATURES:
        raise RuntimeError("Pipeline feature order mismatch")
    return pipeline, metadata


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_verified_model()


def case_frame(
    lactate: float | None,
    invasive_ventilation: int | None,
    mean_map: float,
    urine_output: float | None,
    minimum_heart_rate: float | None,
    bilirubin: float | None,
) -> pd.DataFrame:
    values = [[
        np.nan if lactate is None else float(lactate),
        np.nan if invasive_ventilation is None else int(invasive_ventilation),
        float(mean_map),
        np.nan if urine_output is None else float(urine_output),
        np.nan if minimum_heart_rate is None else float(minimum_heart_rate),
        np.nan if bilirubin is None else float(bilirubin),
    ]]
    return pd.DataFrame(values, columns=list(FEATURES))


def predict_probability(model, **values) -> float:
    frame = case_frame(**values)
    for feature, (minimum, maximum) in INPUT_RANGES.items():
        value = frame.iloc[0][feature]
        if pd.notna(value) and not minimum <= float(value) <= maximum:
            raise ValueError(f"{feature} must be between {minimum:g} and {maximum:g}")
    return float(model.predict_proba(frame)[:, 1][0])


def classify_risk(probability: float, threshold: float) -> str:
    return "High risk" if probability >= threshold else "Low risk"


def optional_number(
    label: str,
    key: str,
    minimum: float,
    maximum: float,
    default: float,
    step: float,
) -> float | None:
    missing_key = f"{key}_missing"
    missing = bool(st.session_state.get(missing_key, False))
    value = st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=step,
        key=key,
        disabled=missing,
    )
    missing = st.toggle("Missing", key=missing_key)
    return None if missing else float(value)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #102a2f;
            --muted: #607277;
            --line: #dbe4e6;
            --surface: #ffffff;
            --canvas: #f4f7f7;
            --accent: #0f6973;
            --accent-dark: #0b525b;
            --accent-soft: #eaf4f4;
            --high: #a63a32;
            --high-soft: #fff2f0;
            --low: #28755d;
            --low-soft: #edf8f3;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(180deg, #eef4f4 0, #f7f9f9 210px, var(--canvas) 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .app-header {
            max-width: 850px;
            margin-bottom: 1.65rem;
        }

        .app-header h1 {
            color: var(--ink);
            font-size: clamp(2rem, 3.4vw, 3.05rem);
            font-weight: 680;
            letter-spacing: -0.042em;
            line-height: 1.04;
            margin: 0 0 0.85rem;
        }

        .app-header p {
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.6;
            margin: 0;
            max-width: 680px;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > [data-testid="stVerticalBlock"]
            > [data-testid="stElementContainer"]:first-child .panel-title
        ) {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid var(--line);
            border-radius: 14px;
            box-shadow: 0 14px 34px rgba(31, 67, 72, 0.06);
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(
            > div > [data-testid="stVerticalBlock"]
            > [data-testid="stElementContainer"]:first-child .panel-title
        ) > div {
            padding: 0.35rem 0.45rem;
        }

        .panel-title {
            color: var(--ink);
            font-size: 1.08rem;
            font-weight: 680;
            letter-spacing: -0.015em;
            margin: 0 0 0.18rem;
        }

        .panel-copy {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
            margin: 0 0 0.65rem;
        }

        [data-testid="stForm"] {
            border: 0;
            padding: 0;
        }

        [data-testid="stWidgetLabel"] p,
        label p {
            color: #314b50;
            font-size: 0.82rem;
            font-weight: 560;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background: #fbfcfc;
            border-color: #cfdadc;
            border-radius: 8px;
            min-height: 2.65rem;
            transition: border-color 150ms ease, box-shadow 150ms ease, background 150ms ease;
        }

        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {
            background: #ffffff;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(15, 105, 115, 0.12);
        }

        [data-testid="stNumberInput"],
        [data-testid="stSelectbox"] {
            margin-bottom: 0.1rem;
        }

        [data-testid="stToggle"] {
            margin-top: -0.42rem;
            margin-bottom: -0.2rem;
        }

        [data-testid="stToggle"] label {
            flex-direction: row-reverse;
            justify-content: flex-end;
            gap: 0.45rem;
        }

        [data-testid="stToggle"] p {
            color: #75878b;
            font-size: 0.74rem;
            font-weight: 500;
        }

        [data-testid="stFormSubmitButton"] button {
            background: var(--accent);
            border: 1px solid var(--accent);
            border-radius: 8px;
            box-shadow: none;
            color: #ffffff;
            font-size: 0.9rem;
            font-weight: 650;
            min-height: 2.85rem;
            margin-top: 0.35rem;
            transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
        }

        [data-testid="stFormSubmitButton"] button:hover {
            background: var(--accent-dark);
            border-color: var(--accent-dark);
            color: #ffffff;
            transform: translateY(-1px);
        }

        [data-testid="stFormSubmitButton"] button:focus {
            box-shadow: 0 0 0 3px rgba(15, 105, 115, 0.18);
        }

        .result-shell {
            min-height: 158px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 0.45rem 0 0.75rem;
            animation: result-in 280ms ease-out both;
        }

        .result-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-bottom: 0.15rem;
        }

        .result-value {
            color: var(--ink);
            font-size: 3.85rem;
            font-weight: 680;
            letter-spacing: -0.065em;
            line-height: 1;
            margin: 0.15rem 0 0.85rem;
        }

        .risk-state {
            align-items: center;
            border-radius: 8px;
            display: flex;
            font-size: 0.88rem;
            font-weight: 650;
            justify-content: space-between;
            padding: 0.72rem 0.82rem;
        }

        .risk-state.high {
            background: var(--high-soft);
            color: var(--high);
        }

        .risk-state.low {
            background: var(--low-soft);
            color: var(--low);
        }

        .risk-state .threshold {
            font-size: 0.75rem;
            font-weight: 560;
            opacity: 0.8;
        }

        .empty-result {
            align-items: flex-start;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 158px;
            padding: 0.45rem 0 0.75rem;
        }

        .empty-result strong {
            color: #314b50;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }

        .empty-result span {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.55;
        }

        .threshold-note {
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.76rem;
            margin: 0.2rem 0 0.9rem;
            padding-top: 0.8rem;
        }

        .benchmark-title {
            color: #314b50;
            font-size: 0.82rem;
            font-weight: 650;
            margin: 0 0 0.45rem;
        }

        .benchmark-table {
            border-bottom: 1px solid var(--line);
            border-top: 1px solid var(--line);
            margin-bottom: 0.25rem;
        }

        .benchmark-row {
            display: grid;
            grid-template-columns: 1.2fr 1fr 1fr;
            column-gap: 0.7rem;
            padding: 0.56rem 0.1rem;
        }

        .benchmark-row + .benchmark-row {
            border-top: 1px solid #edf1f2;
        }

        .benchmark-row.header {
            color: #7b8b8e;
            font-size: 0.7rem;
            font-weight: 650;
            padding-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.045em;
        }

        .benchmark-row.data {
            color: #314b50;
            font-size: 0.81rem;
            font-variant-numeric: tabular-nums;
        }

        .evidence-note {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.55;
            margin: 1.15rem 0 0;
            padding-top: 1rem;
            border-top: 1px solid var(--line);
        }

        @keyframes result-in {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (min-width: 901px) {
            section.main [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2)
            [data-testid="stVerticalBlockBorderWrapper"] {
                position: sticky;
                top: 1rem;
            }
        }

        @media (max-width: 900px) {
            .block-container {
                padding: 1.35rem 1rem 1.5rem;
            }

            .app-header {
                margin-bottom: 1.15rem;
            }

            .app-header h1 {
                font-size: 2rem;
                line-height: 1.08;
            }

            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }

            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                width: 100% !important;
            }

            .result-value {
                font-size: 3.2rem;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_benchmark_table(metadata: dict) -> None:
    rows = "".join(
        (
            '<div class="benchmark-row data">'
            f"<span>{html.escape(row['Cohort'])}</span>"
            f"<span>{html.escape(row['Low risk'])}</span>"
            f"<span>{html.escape(row['High risk'])}</span>"
            "</div>"
        )
        for row in metadata["observed_event_rates"]
    )
    st.markdown(
        (
            '<div class="benchmark-title">Observed event rates at this threshold</div>'
            '<div class="benchmark-table">'
            '<div class="benchmark-row header">'
            "<span>Cohort</span><span>Low risk</span><span>High risk</span>"
            "</div>"
            f"{rows}</div>"
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Amiodarone hemodynamic risk calculator",
        page_icon="⚕",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    try:
        model, metadata = cached_model()
    except Exception as exc:
        st.error(f"Model verification failed: {exc}")
        st.stop()

    st.markdown(
        """
        <div class="app-header">
            <h1>Early hemodynamic intolerance after intravenous amiodarone</h1>
            <p>Enter six variables available at treatment initiation to estimate
            the probability of hemodynamic intolerance within 2 hours.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    entry, result = st.columns((1.18, 0.82), gap="large")
    with entry:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Patient data</div>'
                '<div class="panel-copy">Measurements should reflect the period immediately before amiodarone initiation.</div>',
                unsafe_allow_html=True,
            )
            with st.form("amiodarone_risk_form"):
                row_1 = st.columns(2, gap="medium")
                with row_1[0]:
                    lactate = optional_number("Lactate, mmol/L", "lactate", 0.1, 30.0, 2.0, 0.1)
                with row_1[1]:
                    mean_map = st.number_input(
                        "Mean MAP during preceding 6 h, mmHg",
                        min_value=40.0,
                        max_value=140.0,
                        value=75.0,
                        step=1.0,
                    )

                row_2 = st.columns(2, gap="medium")
                with row_2[0]:
                    urine = optional_number(
                        "Urine output during preceding 6 h, mL",
                        "urine",
                        0.0,
                        10000.0,
                        300.0,
                        10.0,
                    )
                with row_2[1]:
                    minimum_hr = optional_number(
                        "Minimum heart rate during preceding 6 h, beats/min",
                        "minimum_hr",
                        20.0,
                        220.0,
                        65.0,
                        1.0,
                    )

                row_3 = st.columns(2, gap="medium")
                with row_3[0]:
                    invasive_label = st.selectbox(
                        "Invasive ventilation at initiation", ("No", "Yes", "Unknown")
                    )
                    invasive = {"No": 0, "Yes": 1, "Unknown": None}[invasive_label]
                with row_3[1]:
                    bilirubin = optional_number(
                        "Total bilirubin, mg/dL",
                        "bilirubin",
                        0.0,
                        50.0,
                        0.8,
                        0.1,
                    )
                calculate = st.form_submit_button(
                    "Calculate predicted risk",
                    type="primary",
                    use_container_width=True,
                )

    with result:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Estimated risk</div>'
                f'<div class="panel-copy">Final {html.escape(metadata["model"])} model with the MIMIC-IV-derived threshold.</div>',
                unsafe_allow_html=True,
            )
            threshold = float(metadata["threshold"])
            if calculate:
                probability = predict_probability(
                    model,
                    lactate=lactate,
                    invasive_ventilation=invasive,
                    mean_map=mean_map,
                    urine_output=urine,
                    minimum_heart_rate=minimum_hr,
                    bilirubin=bilirubin,
                )
                risk = classify_risk(probability, threshold)
                risk_class = "high" if risk == "High risk" else "low"
                st.markdown(
                    (
                        '<div class="result-shell">'
                        f'<span class="machine-output" data-probability="{probability:.17g}" data-threshold="{threshold:.17g}" style="display:none"></span>'
                        '<div class="result-label">PREDICTED 2-HOUR RISK</div>'
                        f'<div class="result-value">{probability:.1%}</div>'
                        f'<div class="risk-state {risk_class}">'
                        f"<span>{risk}</span>"
                        f'<span class="threshold">threshold {threshold:.1%}</span>'
                        "</div></div>"
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="empty-result">'
                    "<strong>Ready to calculate</strong>"
                    "<span>Complete the patient data and calculate the model estimate.</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="threshold-note">Risk-classification threshold: {threshold:.1%}</div>',
                unsafe_allow_html=True,
            )
            render_benchmark_table(metadata)

    st.markdown(
        '<div class="evidence-note">'
        "Model developed in MIMIC-IV and externally validated in eICU. "
        f"Analysis version {html.escape(metadata['analysis_version'])}; model SHA256 {html.escape(metadata['model_sha256'][:12])}…. "
        "Prospective clinical utility has not been evaluated."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
