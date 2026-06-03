"""
utils/visualization.py
=======================
Visualization helpers: PDF report generation, metric plots.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def generate_pdf_report(entry: dict, output_dir: str) -> str:
    """
    Generate a professional PDF report for a prediction result.

    Args:
        entry: Prediction result dict
        output_dir: Directory to save the PDF

    Returns:
        Path to generated PDF
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Image as RLImage, Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        pdf_path = os.path.join(output_dir, f"RP_Report_{entry['id']}.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        story = []

        # Header style
        title_style = ParagraphStyle(
            'Title', parent=styles['Title'],
            fontSize=20, textColor=colors.HexColor('#0a2540'),
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            'Subtitle', parent=styles['Normal'],
            fontSize=11, textColor=colors.HexColor('#4a6fa5'),
            spaceAfter=4
        )
        heading_style = ParagraphStyle(
            'Heading', parent=styles['Heading2'],
            fontSize=13, textColor=colors.HexColor('#0a2540'),
            spaceBefore=12, spaceAfter=6
        )
        body_style = ParagraphStyle(
            'Body', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#333333'),
            spaceAfter=4, leading=14
        )

        # ── Title ──
        story.append(Paragraph("RetinaScan AI", title_style))
        story.append(Paragraph("Retinitis Pigmentosa Detection Report", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=colors.HexColor('#4a6fa5'), spaceAfter=12))

        # ── Patient / Scan Info ──
        story.append(Paragraph("Scan Information", heading_style))
        info_data = [
            ['Report ID', entry['id']],
            ['Timestamp', entry['timestamp']],
            ['File Name', entry['filename']],
            ['AI Model', 'ResNet50 Transfer Learning'],
        ]
        info_table = Table(info_data, colWidths=[5*cm, 12*cm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0fe')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#0a2540')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (1, 0), (1, -1), [colors.white, colors.HexColor('#f5f8ff')]),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#c0d0e8')),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#c0d0e8')),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 12))

        # ── Prediction Result ──
        story.append(Paragraph("Prediction Result", heading_style))
        risk_color = colors.HexColor('#dc2626') if entry['label'] == 'Retinitis Pigmentosa' else colors.HexColor('#16a34a')
        result_data = [
            ['Diagnosis', entry['label']],
            ['Confidence Score', f"{entry['confidence']}%"],
            ['Raw Model Score', f"{entry['raw_score']}"],
            ['Risk Level', entry.get('risk_level', 'N/A')],
        ]
        result_table = Table(result_data, colWidths=[5*cm, 12*cm])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0fe')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#0a2540')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('TEXTCOLOR', (1, 0), (1, 0), risk_color),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (1, 0), (1, -1), [colors.white, colors.HexColor('#f5f8ff')]),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#c0d0e8')),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#c0d0e8')),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(result_table)
        story.append(Spacer(1, 12))

        # ── Clinical Interpretation ──
        story.append(Paragraph("Clinical Interpretation", heading_style))
        story.append(Paragraph(entry.get('interpretation', 'N/A'), body_style))
        story.append(Spacer(1, 8))

        # ── Images ──
        img_path = entry.get('image_url', '').lstrip('/')
        overlay_path = entry.get('overlay_url', '').lstrip('/')

        if img_path and os.path.exists(img_path):
            story.append(Paragraph("Retinal Image Analysis", heading_style))
            images_data = []
            try:
                orig_img = RLImage(img_path, width=7*cm, height=7*cm)
                images_data.append(orig_img)
            except Exception:
                pass
            if overlay_path and os.path.exists(overlay_path):
                try:
                    overlay_img = RLImage(overlay_path, width=7*cm, height=7*cm)
                    images_data.append(overlay_img)
                except Exception:
                    pass
            if images_data:
                captions = ['Original Retinal Image', 'Grad-CAM Heatmap Overlay']
                img_table = Table([images_data, captions[:len(images_data)]],
                                  colWidths=[8.5*cm]*len(images_data))
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTSIZE', (0, 1), (-1, 1), 8),
                    ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#4a6fa5')),
                ]))
                story.append(img_table)

        story.append(Spacer(1, 16))

        # ── Disclaimer ──
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor('#c0d0e8'), spaceAfter=8))
        disclaimer_style = ParagraphStyle(
            'Disclaimer', parent=styles['Normal'],
            fontSize=8, textColor=colors.HexColor('#888888'),
            leading=11
        )
        story.append(Paragraph(
            "<b>DISCLAIMER:</b> This report is generated by an AI system and is intended for "
            "research and screening purposes only. It does not constitute a medical diagnosis. "
            "Please consult a qualified ophthalmologist or retinal specialist for clinical evaluation "
            "and treatment decisions. The AI model achieves high accuracy on test data but is not "
            "infallible. Always rely on professional medical judgment.", disclaimer_style))

        doc.build(story)
        logger.info(f"PDF report saved: {pdf_path}")
        return pdf_path

    except ImportError:
        logger.warning("reportlab not installed. Generating text report instead.")
        return _generate_text_report(entry, output_dir)


def _generate_text_report(entry: dict, output_dir: str) -> str:
    """Fallback plain-text report when reportlab is not available."""
    txt_path = os.path.join(output_dir, f"RP_Report_{entry['id']}.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("       RETINASCAN AI — RP DETECTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Report ID    : {entry['id']}\n")
        f.write(f"Timestamp    : {entry['timestamp']}\n")
        f.write(f"File Name    : {entry['filename']}\n\n")
        f.write("-" * 40 + "\n")
        f.write("PREDICTION RESULT\n")
        f.write("-" * 40 + "\n")
        f.write(f"Diagnosis    : {entry['label']}\n")
        f.write(f"Confidence   : {entry['confidence']}%\n")
        f.write(f"Raw Score    : {entry['raw_score']}\n")
        f.write(f"Risk Level   : {entry.get('risk_level', 'N/A')}\n\n")
        f.write("-" * 40 + "\n")
        f.write("INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"{entry.get('interpretation', 'N/A')}\n\n")
        f.write("=" * 60 + "\n")
        f.write("DISCLAIMER: AI screening tool only. Consult a specialist.\n")
    return txt_path
