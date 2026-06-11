from heatmap_utils import generate_heatmap
from metadata_utils import get_patient_metadata
from model_utils import predict_xray
from report_generator import generate_report
from clinical_reasoning import calculate_risk, PatientRecord

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="NeuroScan Edge",
    layout="wide"
)

st.title("🩺 NeuroScan Edge")

st.caption("AI-powered chest X-ray triage system for prioritizing radiologist review.")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    metadata = get_patient_metadata(getattr(uploaded_file, "name", None))
    patient_id = metadata.get("patient_id") if metadata else getattr(uploaded_file, "name", "local_upload")
    age = metadata.get("age") if metadata else None
    gender = metadata.get("gender") if metadata else None
    view_position = metadata.get("view_position") if metadata else None

    image = Image.open(uploaded_file)

    uploaded_file.seek(0)
    prediction = predict_xray(uploaded_file)

    top_finding = prediction["display_label"]
    top_score = prediction["display_score"]
    selected_findings = prediction["selected_findings"]
    selected_scores = prediction["selected_scores"]
    no_findings = prediction["no_findings"]

    findings_only = [f for f in selected_findings] if selected_findings else []
    confidences = [s for _, s in selected_scores] if selected_scores else []

    risk_score, urgency, reason = calculate_risk(
        findings_only,
        confidences,
        age=age,
        gender=gender,
        view_position=view_position,
    )

    uploaded_file.seek(0)
    heatmap = generate_heatmap(uploaded_file)

    st.subheader("Patient Information")
    st.write(f"Patient ID: {patient_id}")
    st.write(f"Age: {age if age is not None else 'Unknown'}")
    st.write(f"Gender: {gender if gender else 'Unknown'}")
    st.write(f"View Position: {view_position if view_position else 'Unknown'}")

    if metadata is None:
        st.warning("No NIH metadata found for this image. Clinical reasoning will proceed using available imaging inputs only.")

    st.subheader("Case Summary")
    summary_1, summary_2, summary_3, summary_4 = st.columns(4)
    summary_1.metric("Top Finding", top_finding)
    summary_2.metric("Top Confidence", f"{top_score * 100:.1f}%")
    summary_3.metric("Detected Findings", len(selected_findings) if not no_findings else 0)
    summary_4.metric("Urgency", urgency)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original X-ray")
        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.subheader("AI Attention Heatmap")
        st.image(
            heatmap,
            use_container_width=True
        )

    st.subheader("Top Candidate Findings")
    if no_findings:
        st.write("No significant findings detected above the configured confidence threshold.")
    else:
        for finding, score in selected_scores:
            st.write(f"**{finding}** — {score * 100:.1f}%")

    st.subheader("Clinical Risk Assessment")
    st.write(f"Risk Score: {risk_score}")
    st.write(f"Urgency: {urgency}")
    st.write(f"Reason: {reason}")

    st.subheader("📄 Preliminary Triage Report")
    patient = PatientRecord(
        patient_id=patient_id,
        filename=getattr(uploaded_file, "name", "uploaded_image"),
        age=age,
        gender=gender,
        view_position=view_position,
        findings=findings_only,
        scores=confidences,
        risk_score=risk_score,
        urgency=urgency,
        reason=reason,
    )

    report = generate_report(
        selected_findings,
        selected_scores,
        urgency,
        patient_id=patient_id,
        age=age,
        gender=gender,
        view_position=view_position,
        risk_score=risk_score,
        reason=reason,
    )
    st.markdown(report)