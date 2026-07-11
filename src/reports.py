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
    Generates a structured PDF report detailing mango ripeness analysis.
    results: List of dicts, each with keys: 'filename', 'color_pred', 'color_conf',
             'texture_pred', 'texture_conf', 'geom_pred', 'geom_conf', 'dl_pred', 'dl_conf', 'final_pred'
    """
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, "Mango_Ripeness_Report.pdf")
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles for Premium Look
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#FF8C00'),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#2E8B57'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#333333')
    )
    
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white
    )

    story = []
    
    # Title
    story.append(Paragraph("Mango Ripeness Analysis Report", title_style))
    story.append(Paragraph("Automated batch grading using Computer Vision & Deep Learning techniques.", body_style))
    story.append(Spacer(1, 15))
    
    # Executive Summary Table
    story.append(Paragraph("Executive Summary", h2_style))
    summary_data = [
        [Paragraph("Metric", table_header_style), Paragraph("Value", table_header_style)],
        [Paragraph("Total Images Processed", table_text_style), Paragraph(str(summary_stats['total']), table_text_style)],
        [Paragraph("Unripe Count", table_text_style), Paragraph(str(summary_stats['unripe']), table_text_style)],
        [Paragraph("Partially Ripe Count", table_text_style), Paragraph(str(summary_stats['partially_ripe']), table_text_style)],
        [Paragraph("Fully Ripe Count", table_text_style), Paragraph(str(summary_stats['fully_ripe']), table_text_style)],
        [Paragraph("Dominant Class", table_text_style), Paragraph(summary_stats['dominant'], table_text_style)]
    ]
    
    t_summary = Table(summary_data, colWidths=[3.0 * inch, 3.5 * inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E8B57')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('TOPPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#E0E0E0')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F9F9F9')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#F9F9F9'), colors.HexColor('#FFFFFF')]),
        ('TOPPADDING', (0,1), (-1,-1), 6),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 20))
    
    # Detailed Results Table
    story.append(Paragraph("Detailed Ripeness Evaluation Matrix", h2_style))
    
    headers = [
        Paragraph("Filename", table_header_style),
        Paragraph("Color prediction", table_header_style),
        Paragraph("Texture prediction", table_header_style),
        Paragraph("Geometry prediction", table_header_style),
        Paragraph("Deep Learning", table_header_style),
        Paragraph("Consensus Grade", table_header_style)
    ]
    
    table_data = [headers]
    for r in results:
        table_data.append([
            Paragraph(r['filename'], table_text_style),
            Paragraph(f"{r['color_pred']} ({r['color_conf']:.2f})", table_text_style),
            Paragraph(f"{r['texture_pred']} ({r['texture_conf']:.2f})", table_text_style),
            Paragraph(f"{r['geom_pred']} ({r['geom_conf']:.2f})", table_text_style),
            Paragraph(f"{r['dl_pred']} ({r['dl_conf']:.2f})", table_text_style),
            Paragraph(f"<b>{r['final_pred']}</b>", table_text_style)
        ])
        
    t_details = Table(table_data, colWidths=[1.5 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
    t_details.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#FF8C00')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D3D3D3')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F0F8FF')]),
        ('TOPPADDING', (0,1), (-1,-1), 5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 5),
    ]))
    
    story.append(t_details)
    
    # Build Document
    doc.build(story)
    return pdf_path
