import streamlit as st
import pandas as pd
import time

from PIL import Image
from pdf_utils import create_pdf_report
from metadata_utils import get_patient_metadata
from model_utils import predict_xray
from queue_utils import get_urgency
from clinical_reasoning import calculate_risk
from heatmap_utils import generate_heatmap
from report_generator import generate_report

st.set_page_config(
    page_title="NeuroScan Edge",
    layout="wide"
)

st.title("🩺 NeuroScan Edge")

hero_placeholder = st.empty()

uploaded_files = st.file_uploader(
    "Upload Multiple Chest X-rays",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)


def render_landing_metrics(
    scans_processed=0,
    high_priority_cases=0,
    medium_priority_cases=0,
    low_priority_cases=0,
    average_risk_score=0.0,
    average_latency=0.0
):
    with hero_placeholder:
        st.markdown(
            "### 🩺 NeuroScan Edge\n\n"
            "AI-powered chest X-ray triage system for prioritizing radiologist review."
        )

        c1, c2, c3 = st.columns([1, 1, 1])

        c1.metric("Total Scans", scans_processed)
        c2.metric("High Risk", high_priority_cases)
        c3.metric("Medium Risk", medium_priority_cases)

        c4, c5 = st.columns([1, 1])
        c4.metric("Low Risk", low_priority_cases)
        c5.metric("Average Risk Score", f"{average_risk_score:.1f}")
        st.write(f"Average Inference Time: {average_latency:.2f}s")

        st.divider()


render_landing_metrics()

if uploaded_files:

    st.markdown("### NIH metadata is loaded automatically from ChestXray14 entries when available.")

    queue_data = []
    progress = st.progress(0)
    total_files = len(uploaded_files)
    missing_metadata = False

    for idx, file in enumerate(uploaded_files):
        file.seek(0)
        metadata = get_patient_metadata(getattr(file, "name", None))
        if metadata is None:
            metadata = {
                "patient_id": None,
                "age": None,
                "gender": None,
                "view_position": None,
            }
            missing_metadata = True

        start_time = time.time()
        prediction = predict_xray(file)

        latency = round(time.time() - start_time, 3)
        top_finding = prediction.get("display_label")
        top_score = float(prediction.get("display_score", 0.0))
        selected_findings = prediction.get("selected_findings", [])
        selected_scores = prediction.get("selected_scores", [])
        confidences = [s for _, s in selected_scores] if selected_scores else []

        risk_score, urgency_simple, reason = calculate_risk(
            selected_findings,
            confidences,
            age=metadata.get("age"),
            gender=metadata.get("gender"),
            view_position=metadata.get("view_position"),
        )

        urgency = get_urgency(
            selected_findings,
            top_score,
            age=metadata.get("age"),
            gender=metadata.get("gender"),
            view_position=metadata.get("view_position"),
        )

        queue_data.append({
            "Patient ID": metadata.get("patient_id") or file.name,
            "Filename": file.name,
            "Finding": top_finding,
            "Score": round(top_score * 100, 1),
            "Detected Findings": ", ".join(selected_findings) if selected_findings else "No Significant Findings",
            "Selected Findings": selected_findings,
            "Selected Scores": selected_scores,
            "Urgency": urgency,
            "Latency (s)": latency,
            "File": file,
            "Results": prediction.get("predictions"),
            "Age": metadata.get("age"),
            "Gender": metadata.get("gender"),
            "View Position": metadata.get("view_position"),
            "Risk Score": risk_score,
            "Risk Reason": reason,
        })

        progress.progress((idx + 1) / total_files)

    if missing_metadata:
        st.warning(
            "Some uploads could not be matched to NIH ChestXray14 metadata. "
            "Those cases are still processed with image-only inference."
        )

    df = pd.DataFrame(queue_data)

    urgency_order = {
        "🔴 HIGH": 0,
        "🟡 MEDIUM": 1,
        "🟢 LOW": 2,
    }

    df["priority"] = df["Urgency"].map(urgency_order).fillna(3)
    df = df.sort_values(["Risk Score", "priority", "Score"], ascending=[False, True, False]).reset_index(drop=True)

    high_count = len(df[df["Urgency"] == "🔴 HIGH"])
    medium_count = len(df[df["Urgency"] == "🟡 MEDIUM"])
    low_count = len(df[df["Urgency"] == "🟢 LOW"])

    avg_latency = round(df["Latency (s)"].mean(), 2) if len(df) > 0 else 0.0
    avg_risk_score = round(df["Risk Score"].mean(), 1) if len(df) > 0 else 0.0

    render_landing_metrics(
        scans_processed=len(df),
        high_priority_cases=high_count,
        medium_priority_cases=medium_count,
        low_priority_cases=low_count,
        average_risk_score=avg_risk_score,
        average_latency=avg_latency,
    )

    st.subheader("📋 Radiologist Priority Queue")

    export_df = df[["Patient ID", "Filename", "Age", "Gender", "View Position", "Finding", "Score", "Risk Score", "Urgency", "Latency (s)"]]
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Queue CSV",
        data=csv_bytes,
        file_name="neuroscan_queue.csv",
        mime="text/csv",
    )

    st.dataframe(
        export_df,
        hide_index=True,
        use_container_width=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Scans", len(df))

    with c2:
        st.metric("High Priority", high_count)

    with c3:
        st.metric("Medium Priority", medium_count)

    with c4:
        st.metric("Low Priority", low_count)

    st.divider()

    st.subheader("🔍 Patient Review")

    selected_patient = st.selectbox("Select Scan", df["Filename"])
    selected_case = next(item for item in queue_data if item["Filename"] == selected_patient)

    selected_file = selected_case["File"]
    selected_file.seek(0)
    image = Image.open(selected_file)

    selected_file.seek(0)
    heatmap = generate_heatmap(selected_file)

    selected_findings = selected_case.get("Selected Findings", [])
    finding_count = len(selected_findings)

    # -------------------------
    # Case Metrics
    # -------------------------

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Top Finding", selected_case["Finding"])
    metric_2.metric("Top Confidence", f"{selected_case['Score']:.1f}%")
    metric_3.metric("Detected Findings", finding_count)
    metric_4.metric("Urgency", selected_case["Urgency"])

    st.write("**Detected findings:**", ", ".join(selected_findings) if selected_findings else "No Significant Findings")
    st.divider()

    # -------------------------
    # Images
    # -------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original X-ray")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("AI Attention Heatmap")
        st.image(heatmap, use_container_width=True)

    # -------------------------
    # Findings + Urgency
    # -------------------------

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top Findings")
        if selected_case.get("Selected Scores"):
            for finding, score in selected_case["Selected Scores"]:
                st.write(f"**{finding}** — {score * 100:.1f}%")
        else:
            st.write("No significant findings detected above threshold.")

    with col4:
        st.subheader("Urgency Level")
        st.markdown(f"# {selected_case['Urgency']}")
        st.write("Top Finding:", selected_case["Finding"])
        st.write("Confidence Score:", selected_case["Score"])
        st.write("Inference Time:", f"{selected_case['Latency (s)']} sec")
        st.divider()
        st.subheader("Patient Information")
        st.write(f"Patient ID: {selected_case.get('Patient ID', 'N/A')}")
        st.write(f"Age: {selected_case.get('Age', 'N/A')}")
        st.write(f"Gender: {selected_case.get('Gender', 'N/A')}")
        st.write(f"View Position: {selected_case.get('View Position', 'N/A')}")
        st.divider()
        st.subheader("Clinical Risk Assessment")
        st.write(f"Risk Score: {selected_case.get('Risk Score', 'N/A')}")
        st.write(f"Reason: {selected_case.get('Risk Reason', 'N/A')}")

    # -------------------------
    # Gemini Report
    # -------------------------

    st.subheader(
        "📄 NeuroScan Edge Preliminary Report"
    )

    with st.spinner(
        "Generating radiology note..."
    ):

        try:

            report = generate_report(
                selected_case.get("Selected Findings", []),
                selected_case.get("Selected Scores", []),
                selected_case["Urgency"],
                age=selected_case.get("Age"),
                gender=selected_case.get("Gender"),
                view_position=selected_case.get("View Position"),
                risk_score=selected_case.get("Risk Score"),
                reason=selected_case.get("Risk Reason"),
            )
            
            pdf_buffer = create_pdf_report(
                finding=selected_case["Finding"],
                score=selected_case["Score"],
                urgency=selected_case["Urgency"],
                report_text=report,
                age=selected_case.get("Age"),
                gender=selected_case.get("Gender"),
                view_position=selected_case.get("View Position"),
                risk_score=selected_case.get("Risk Score"),
                reason=selected_case.get("Risk Reason"),
                xray_image=image,
                heatmap_image=heatmap,
                inference_time=selected_case["Latency (s)"],
            )

            st.markdown(report)
            st.download_button(
                label="📄 Download Patient Report",
                data=pdf_buffer,
                file_name=f"{selected_case['Patient ID']}_report.pdf",
                mime="application/pdf"
            )

        except Exception as e:

            st.error(
                f"Report generation failed: {e}"
            )