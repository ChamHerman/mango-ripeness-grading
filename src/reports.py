import os
import tempfile
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
    summary_stats: Dict with keys: 'total', 'unripe', 'fully_ripe', 'overripe', 'dominant'
    """
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, "Mango_Ripeness_Inspection_Report.pdf")
    
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
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#D97706'),
        spaceAfter=8
    )
    
    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=14,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155')
    )
    
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    story = []
    
    # Title & Header
    story.append(Paragraph("[Report] Mango Ripeness Quality Inspection Report", title_style))
    story.append(Paragraph("Automated Classical Image Processing Evaluation Suite (BMDS2133 - Mode A Comparative Study)", body_style))
    story.append(Spacer(1, 12))
    
    # Executive Summary Table
    story.append(Paragraph("[Executive Summary]", h2_style))
    summary_data = [
        [Paragraph("Inspection Metric", table_header_style), Paragraph("Evaluated Metric Value", table_header_style)],
        [Paragraph("Total Ingested Images", table_text_style), Paragraph(str(summary_stats.get('total', len(results))), table_text_style)],
        [Paragraph("Unripe Mangoes (Count)", table_text_style), Paragraph(str(summary_stats.get('unripe', 0)), table_text_style)],
        [Paragraph("Fully Ripe Mangoes (Count)", table_text_style), Paragraph(str(summary_stats.get('fully_ripe', 0)), table_text_style)],
        [Paragraph("Overripe Mangoes (Count)", table_text_style), Paragraph(str(summary_stats.get('overripe', 0)), table_text_style)],
        [Paragraph("Dominant Sample Class", table_text_style), Paragraph(str(summary_stats.get('dominant', 'N/A')).upper(), table_text_style)],
        [Paragraph("Evaluation Framework", table_text_style), Paragraph("Multi-Technique Classical Computer Vision Ensemble", table_text_style)]
    ]
    
    t_summary = Table(summary_data, colWidths=[3.0 * inch, 3.5 * inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F766E')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F8FAFC'), colors.HexColor('#FFFFFF')]),
        ('TOPPADDING', (0,1), (-1,-1), 5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 5),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 16))
    
    # Detailed Diagnostics Table
    story.append(Paragraph("[Detailed Diagnostic Matrix]", h2_style))
    
    headers = [
        Paragraph("Sample ID / Name", table_header_style),
        Paragraph("Morphology (Herman)", table_header_style),
        Paragraph("Color-Space (Siew Feng)", table_header_style),
        Paragraph("Texture (Kai Bin)", table_header_style),
        Paragraph("Geometry (Wei Kang)", table_header_style),
        Paragraph("Consensus Grade", table_header_style)
    ]
    
    table_data = [headers]
    for r in results:
        m_str = f"{r.get('morph_pred', '-')} ({r.get('morph_conf', 0):.1f}%)" if 'morph_pred' in r else "-"
        c_str = f"{r.get('color_pred', '-')} ({r.get('color_conf', 0):.1f}%)" if 'color_pred' in r else "-"
        t_str = f"{r.get('texture_pred', '-')} ({r.get('texture_conf', 0):.1f}%)" if 'texture_pred' in r else "-"
        g_str = f"{r.get('geom_pred', '-')} ({r.get('geom_conf', 0):.1f}%)" if 'geom_pred' in r else "-"
        
        table_data.append([
            Paragraph(r.get('filename', 'Sample'), table_text_style),
            Paragraph(m_str, table_text_style),
            Paragraph(c_str, table_text_style),
            Paragraph(t_str, table_text_style),
            Paragraph(g_str, table_text_style),
            Paragraph(f"<b>{r.get('final_pred', 'N/A').upper()}</b>", table_text_style)
        ])
        
    t_details = Table(table_data, colWidths=[1.4 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch])
    t_details.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#D97706')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,1), (-1,-1), 4),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
    ]))
    
    story.append(t_details)
    
    # Build Document
    doc.build(story)
    return pdf_path
