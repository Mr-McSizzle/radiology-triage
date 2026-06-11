from io import BytesIO
from datetime import datetime
import numpy as np

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Image as RLImage
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def _convert_numpy_to_pil(np_array):
    """Convert numpy array to PIL Image."""
    from PIL import Image
    
    if isinstance(np_array, np.ndarray):
        # Handle different dtypes
        if np_array.dtype != np.uint8:
            if np_array.max() <= 1.0:
                np_array = (np_array * 255).astype(np.uint8)
            else:
                np_array = np_array.astype(np.uint8)
        
        return Image.fromarray(np_array)
    return np_array


def create_pdf_report(
    finding,
    score,
    urgency,
    report_text,
    patient_id=None,
    age=None,
    gender=None,
    view_position=None,
    risk_score=None,
    reason=None,
    xray_image=None,
    heatmap_image=None,
    inference_time=None
):
    """
    Create a professional radiology triage report PDF.
    
    Parameters:
    -----------
    finding : str
        Top finding from the model
    score : float
        Confidence score
    urgency : str
        Urgency level (e.g., "🔴 HIGH")
    report_text : str
        AI-generated clinical note
    xray_image : PIL.Image or np.ndarray, optional
        Original X-ray image
    heatmap_image : PIL.Image or np.ndarray, optional
        GradCAM heatmap image
    inference_time : float, optional
        Inference latency in seconds
    """
    
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=(8.5 * inch, 11 * inch),
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch
    )

    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=3,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#2e5c8a'),
        spaceAfter=6,
        spaceBefore=4,
        fontName='Helvetica-Bold',
        borderColor=colors.HexColor('#2e5c8a'),
        borderWidth=0.5,
        borderPadding=4
    )

    clinical_style = ParagraphStyle(
        'Clinical',
        parent=styles['BodyText'],
        fontSize=10,
        leading=12,
        spaceAfter=6,
        alignment=TA_LEFT
    )

    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER,
        spaceAfter=1
    )

    content = []

    # ========================================
    # HEADER
    # ========================================
    
    content.append(
        Paragraph(
            "🩺 NEUROSCAN EDGE RADIOLOGY TRIAGE REPORT",
            title_style
        )
    )
    
    content.append(Spacer(1, 3))

    # Patient and timestamp info
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Patient metadata for header
    patient_label = f"Report_{now.strftime('%Y%m%d_%H%M%S')}"
    demographics_parts = [f"<b>Report:</b> {patient_label}", f"<b>Generated:</b> {timestamp}"]
    if patient_id:
        demographics_parts.append(f"<b>Patient ID:</b> {patient_id}")
    if age is not None:
        demographics_parts.append(f"<b>Age:</b> {age}")
    if gender:
        demographics_parts.append(f"<b>Gender:</b> {gender}")
    if view_position:
        demographics_parts.append(f"<b>View Position:</b> {view_position}")

    patient_info = "<br/>".join(demographics_parts)
    content.append(
        Paragraph(patient_info, styles['Normal'])
    )

    content.append(Spacer(1, 6))
    
    # Separator line
    content.append(
        Paragraph(
            "=" * 60,
            styles['Normal']
        )
    )

    # ========================================
    # IMAGES SECTION (Side by Side)
    # ========================================

    content.append(Spacer(1, 6))
    content.append(
        Paragraph(
            "IMAGING",
            section_style
        )
    )

    # Create table for side-by-side images
    image_data = []
    
    xray_cell = None
    heatmap_cell = None

    if xray_image is not None:
        xray_image_pil = _convert_numpy_to_pil(xray_image)
        xray_buffer = BytesIO()
        xray_image_pil.save(xray_buffer, format='PNG')
        xray_buffer.seek(0)
        
        try:
            xray_cell = RLImage(
                xray_buffer,
                width=2.8 * inch,
                height=2.8 * inch
            )
        except:
            xray_cell = Paragraph("[X-ray]", styles['Normal'])
    else:
        xray_cell = Paragraph("[X-ray]", styles['Normal'])

    if heatmap_image is not None:
        heatmap_image_pil = _convert_numpy_to_pil(heatmap_image)
        heatmap_buffer = BytesIO()
        heatmap_image_pil.save(heatmap_buffer, format='PNG')
        heatmap_buffer.seek(0)
        
        try:
            heatmap_cell = RLImage(
                heatmap_buffer,
                width=2.8 * inch,
                height=2.8 * inch
            )
        except:
            heatmap_cell = Paragraph("[Heatmap]", styles['Normal'])
    else:
        heatmap_cell = Paragraph("[Heatmap]", styles['Normal'])

    image_table = Table(
        [[xray_cell, heatmap_cell]],
        colWidths=[3 * inch, 3 * inch]
    )
    image_table.setStyle(
        TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ])
    )

    content.append(image_table)

    # ========================================
    # TRIAGE SUMMARY SECTION
    # ========================================

    content.append(Spacer(1, 6))
    content.append(
        Paragraph(
            "TRIAGE SUMMARY",
            section_style
        )
    )

    # Clean urgency display
    urgency_clean = urgency.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")

    summary_data = [
        [
            Paragraph("<b>Top Finding:</b>", styles['Normal']),
            Paragraph(str(finding), styles['Normal'])
        ],
        [
            Paragraph("<b>Confidence Score:</b>", styles['Normal']),
            Paragraph(f"{score * 100:.1f}%" if isinstance(score, float) else str(score), styles['Normal'])
        ],
        [
            Paragraph("<b>Urgency:</b>", styles['Normal']),
            Paragraph(urgency_clean, styles['Normal'])
        ]
    ]

    if inference_time is not None:
        summary_data.append([
            Paragraph("<b>Inference Time:</b>", styles['Normal']),
            Paragraph(f"{inference_time} sec", styles['Normal'])
        ])

    summary_table = Table(summary_data, colWidths=[2.5 * inch, 2.5 * inch])
    summary_table.setStyle(
        TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ])
    )

    content.append(summary_table)

    if patient_id or age is not None or gender or view_position:
        content.append(Spacer(1, 6))
        content.append(
            Paragraph(
                "PATIENT INFORMATION",
                section_style
            )
        )
        patient_info_lines = []
        if patient_id:
            patient_info_lines.append(f"Patient ID: {patient_id}")
        if age is not None:
            patient_info_lines.append(f"Age: {age}")
        if gender:
            patient_info_lines.append(f"Gender: {gender}")
        if view_position:
            patient_info_lines.append(f"View Position: {view_position}")
        content.append(Paragraph("; ".join(patient_info_lines), clinical_style))

    # ========================================
    # AI PRELIMINARY NOTE SECTION
    # ========================================

    content.append(Spacer(1, 6))
    content.append(
        Paragraph(
            "AI PRELIMINARY NOTE",
            section_style
        )
    )

    content.append(
        Paragraph(
            report_text,
            clinical_style
        )
    )

    # Clinical Risk Assessment section
    if risk_score is not None or reason:
        content.append(Spacer(1, 6))
        content.append(
            Paragraph(
                "CLINICAL RISK ASSESSMENT",
                section_style
            )
        )
        risk_lines = []
        if risk_score is not None:
            risk_lines.append(f"Risk score: {risk_score}")
        if urgency_clean:
            risk_lines.append(f"Urgency: {urgency_clean}")
        if reason:
            risk_lines.append(f"Reason: {reason}")

        content.append(Paragraph("; ".join(risk_lines), clinical_style))

    # ========================================
    # FOOTER
    # ========================================

    content.append(Spacer(1, 8))
    content.append(
        Paragraph(
            "=" * 60,
            styles['Normal']
        )
    )

    content.append(Spacer(1, 4))

    content.append(
        Paragraph(
            "Generated by NeuroScan Edge",
            footer_style
        )
    )

    content.append(
        Paragraph(
            "AMD Edge AI Demonstration",
            footer_style
        )
    )

    content.append(Spacer(1, 2))
    
    disclaimer = (
        "<b>DISCLAIMER:</b> This assessment is AI-generated and is <b>NOT</b> a clinical diagnosis. "
        "All reports must be reviewed by a qualified radiologist before clinical action."
    )
    
    content.append(
        Paragraph(
            disclaimer,
            footer_style
        )
    )

    # Build the PDF
    doc.build(content)

    buffer.seek(0)

    return buffer