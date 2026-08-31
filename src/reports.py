import os
import tempfile
import time
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(results, summary_stats):
    """
    Generates a structured PDF inspection report detailing mango ripeness analysis.
    results: List of dicts with keys: 'filename', 'morph_pred', 'morph_conf', 'color_pred', 'color_conf',
             'texture_pred', 'texture_conf', 'geom_pred', 'geom_conf', 'final_pred'
    summary_stats: Dict with summary metrics.
    Returns: Raw bytes of the generated PDF document.
    """
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f"Mango_Ripeness_Report_{os.getpid()}_{int(time.time())}.pdf")
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Professional Styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#D97706'),
        spaceAfter=6
    )
    
    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []
    
    # Title & Header
    story.append(Paragraph("Mango Ripeness Quality Inspection Report", title_style))
    story.append(Paragraph("Automated Classical Image Processing Evaluation Suite — BMDS2133 Final Integration", body_style))
    story.append(Spacer(1, 10))
    
    # Resolve summary statistics
    total_val = summary_stats.get('total', summary_stats.get('total_assessed', len(results)))
    unripe_val = summary_stats.get('unripe', summary_stats.get('consensus_unripe', 0))
    ripe_val = summary_stats.get('fully_ripe', summary_stats.get('consensus_ripe', 0))
    overripe_val = summary_stats.get('overripe', summary_stats.get('consensus_overripe', 0))
    avg_conf = summary_stats.get('avg_confidence', 0.0)
    
    # Determine dominant
    dom = summary_stats.get('dominant')
    if not dom:
        class_counts = {'Unripe': unripe_val, 'Fully Ripe': ripe_val, 'Overripe': overripe_val}
        dom = max(class_counts.keys(), key=lambda k: class_counts[k]) if class_counts else 'N/A'
    
    # Executive Summary Table
    story.append(Paragraph("Executive Summary & Batch KPIs", h2_style))
    summary_data = [
        [Paragraph("Inspection Metric", table_header_style), Paragraph("Evaluated Metric Value", table_header_style)],
        [Paragraph("Total Ingested Images", table_text_style), Paragraph(str(total_val), table_text_style)],
        [Paragraph("Unripe Mangoes (Count / %)", table_text_style), Paragraph(f"{unripe_val} ({(unripe_val/total_val*100):.1f}%)" if total_val > 0 else "0", table_text_style)],
        [Paragraph("Fully Ripe Mangoes (Count / %)", table_text_style), Paragraph(f"{ripe_val} ({(ripe_val/total_val*100):.1f}%)" if total_val > 0 else "0", table_text_style)],
        [Paragraph("Overripe Mangoes (Count / %)", table_text_style), Paragraph(f"{overripe_val} ({(overripe_val/total_val*100):.1f}%)" if total_val > 0 else "0", table_text_style)],
        [Paragraph("Dominant Maturity Class", table_text_style), Paragraph(str(dom).upper(), table_text_style)],
        [Paragraph("Average Decision Confidence", table_text_style), Paragraph(f"{avg_conf:.1f}%" if avg_conf else "N/A", table_text_style)],
        [Paragraph("Integrated Team Modules", table_text_style), Paragraph("Morphology (Herman), Color (Siew Feng), Texture (Kai Bin), Geometry (Wei Kang)", table_text_style)]
    ]
    
    t_summary = Table(summary_data, colWidths=[2.8 * inch, 3.7 * inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F766E')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('TOPPADDING', (0,0), (-1,0), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F8FAFC'), colors.HexColor('#FFFFFF')]),
        ('TOPPADDING', (0,1), (-1,-1), 4),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 14))
    
    # Detailed Diagnostics Table
    story.append(Paragraph("Detailed Diagnostic Matrix (Per-Sample Multi-Model Decisions)", h2_style))
    
    headers = [
        Paragraph("Sample ID / Name", table_header_style),
        Paragraph("Morphology (Herman)", table_header_style),
        Paragraph("Color (Siew Feng)", table_header_style),
        Paragraph("Texture (Kai Bin)", table_header_style),
        Paragraph("Geometry (Wei Kang)", table_header_style),
        Paragraph("Final Decision", table_header_style)
    ]
    
    table_data = [headers]
    for r in results:
        m_str = r.get('morph_pred', '-') if isinstance(r.get('morph_pred'), str) and '(' in r.get('morph_pred', '') else (f"{r.get('morph_pred', '-')} ({r.get('morph_conf', 0):.1f}%)" if r.get('morph_pred') and r.get('morph_pred') != '-' else "-")
        c_str = r.get('color_pred', '-') if isinstance(r.get('color_pred'), str) and '(' in r.get('color_pred', '') else (f"{r.get('color_pred', '-')} ({r.get('color_conf', 0):.1f}%)" if r.get('color_pred') and r.get('color_pred') != '-' else "-")
        t_str = r.get('texture_pred', '-') if isinstance(r.get('texture_pred'), str) and '(' in r.get('texture_pred', '') else (f"{r.get('texture_pred', '-')} ({r.get('texture_conf', 0):.1f}%)" if r.get('texture_pred') and r.get('texture_pred') != '-' else "-")
        g_str = r.get('geom_pred', '-') if isinstance(r.get('geom_pred'), str) and '(' in r.get('geom_pred', '') else (f"{r.get('geom_pred', '-')} ({r.get('geom_conf', 0):.1f}%)" if r.get('geom_pred') and r.get('geom_pred') != '-' else "-")
        
        final_str = str(r.get('final_pred', 'N/A')).upper()
        
        table_data.append([
            Paragraph(str(r.get('filename', 'Sample')), table_text_style),
            Paragraph(str(m_str), table_text_style),
            Paragraph(str(c_str), table_text_style),
            Paragraph(str(t_str), table_text_style),
            Paragraph(str(g_str), table_text_style),
            Paragraph(f"<b>{final_str}</b>", table_text_style)
        ])
        
    t_details = Table(table_data, colWidths=[1.4 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch])
    t_details.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#D97706')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('TOPPADDING', (0,0), (-1,0), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,1), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 3.5),
    ]))
    
    story.append(t_details)
    
    # Build Document
    doc.build(story)
    
    # Read bytes and remove temp file
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    try:
        os.remove(pdf_path)
    except Exception:
        pass
        
    return pdf_bytes
