def generate_report(
    selected_findings,
    selected_scores,
    urgency,
    patient_id=None,
    age=None,
    gender=None,
    view_position=None,
    risk_score=None,
    reason=None,
):
    """
    Generate a calibrated preliminary note using thresholded findings.

    Parameters:
    -----------
    selected_findings : list[str]
        Detected findings above the configured confidence threshold.
    selected_scores : list[tuple[str, float]]
        Detected findings with their confidence scores.
    urgency : str
        Urgency level for the case.

    Returns:
    --------
    str : Clinical-style preliminary note
    """

    urgency_clean = (urgency or "").replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")

    metadata_lines = []
    if patient_id:
        metadata_lines.append(f"Patient ID: {patient_id}")
    if age is not None:
        metadata_lines.append(f"Age: {age}")
    if gender:
        metadata_lines.append(f"Gender: {gender}")
    if view_position:
        metadata_lines.append(f"View Position: {view_position}")

    if not selected_findings or selected_findings == ["No Significant Findings"]:
        findings_section = (
            "No significant abnormalities were identified above the configured confidence threshold."
        )
    else:
        findings_text = []
        for label, score in selected_scores:
            findings_text.append(f"{label} ({score * 100:.1f}% confidence)")

        if len(findings_text) == 1:
            findings_section = f"{findings_text[0]} was detected above threshold."
        else:
            findings_section = (
                "Findings above threshold include "
                + ", ".join(findings_text[:-1])
                + ", and "
                + findings_text[-1]
                + "."
            )

    # Risk assessment
    risk_lines = []
    if risk_score is not None:
        risk_lines.append(f"Clinical risk score: {risk_score}")
    if urgency_clean:
        risk_lines.append(f"Triage urgency: {urgency_clean}")
    if reason:
        risk_lines.append(f"Reason: {reason}")

    # Build the report
    report_parts = []
    if metadata_lines:
        report_parts.append("Patient Information:\n" + "; ".join(metadata_lines))

    report_parts.append("\nAI Findings:\n" + findings_section)

    if risk_lines:
        report_parts.append("\nClinical Risk Assessment:\n" + "; ".join(risk_lines))

    report_parts.append(
        "\nThis assessment is AI-generated and not a medical diagnosis. Radiologist review is required."
    )

    return "\n\n".join(report_parts)