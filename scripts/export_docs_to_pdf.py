#!/usr/bin/env python3
"""Convert NIDS-DL markdown docs to shareable PDFs."""

import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def md_to_paragraphs(md_text, styles):
    """Parse markdown text into ReportLab flowables."""
    flowables = []
    lines = md_text.split('\n')
    i = 0
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    title_style = ParagraphStyle('Title2', parent=styles['Title'],
                                  fontSize=20, spaceAfter=8,
                                  textColor=colors.HexColor('#1a237e'))
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
                               fontSize=16, spaceBefore=14, spaceAfter=6,
                               textColor=colors.HexColor('#283593'),
                               borderPad=2)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                               fontSize=13, spaceBefore=10, spaceAfter=4,
                               textColor=colors.HexColor('#1565c0'))
    h3_style = ParagraphStyle('H3', parent=styles['Heading3'],
                               fontSize=11, spaceBefore=8, spaceAfter=3,
                               textColor=colors.HexColor('#0277bd'))
    body_style = ParagraphStyle('Body2', parent=styles['Normal'],
                                 fontSize=9, leading=13, spaceAfter=3)
    bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],
                                   fontSize=9, leading=12, spaceAfter=2,
                                   leftIndent=16, bulletIndent=6)
    code_style = ParagraphStyle('Code', parent=styles['Code'],
                                 fontSize=7.5, leading=11, spaceAfter=4,
                                 backColor=colors.HexColor('#f5f5f5'),
                                 borderColor=colors.HexColor('#cccccc'),
                                 borderWidth=0.5, borderPad=4,
                                 leftIndent=8, fontName='Courier')

    def clean_inline(text):
        """Clean inline markdown: bold, italic, code, links."""
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = text.replace('✅', '[OK]').replace('⚠️', '[!]').replace('🔜', '[>]')
        text = text.replace('🟢', '[>]').replace('★', '*').replace('↑', '+')
        text = text.replace('✓', '[OK]').replace('→', '->')
        return text

    def flush_table():
        nonlocal table_rows, in_table
        if table_rows:
            col_widths = None
            data = []
            for ri, row in enumerate(table_rows):
                cells = [Paragraph(clean_inline(c.strip()),
                                   ParagraphStyle('TC', fontSize=8, leading=10,
                                                  fontName='Helvetica-Bold' if ri == 0 else 'Helvetica'))
                         for c in row]
                data.append(cells)
            t = Table(data, hAlign='LEFT', colWidths=col_widths)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#283593')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1),
                 [colors.HexColor('#f8f9fa'), colors.HexColor('#e8eaf6')]),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#9fa8da')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            flowables.append(t)
            flowables.append(Spacer(1, 6))
        table_rows = []
        in_table = False

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith('```'):
            if in_code_block:
                code_text = '\n'.join(code_lines)
                flowables.append(Paragraph(
                    code_text.replace('\n', '<br/>').replace(' ', '&nbsp;'),
                    code_style))
                code_lines = []
                in_code_block = False
            else:
                if in_table:
                    flush_table()
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
            i += 1
            continue

        # Table row
        if line.startswith('|') and line.endswith('|'):
            cells = [c for c in line.strip('|').split('|')]
            # Skip separator rows
            if all(re.match(r'^[-: ]+$', c.strip()) for c in cells if c.strip()):
                i += 1
                continue
            in_table = True
            table_rows.append(cells)
            i += 1
            continue
        elif in_table:
            flush_table()

        # Headings
        if line.startswith('# '):
            flowables.append(Paragraph(clean_inline(line[2:]), title_style))
        elif line.startswith('## '):
            flowables.append(HRFlowable(width='100%', thickness=1,
                                         color=colors.HexColor('#7986cb'), spaceAfter=2))
            flowables.append(Paragraph(clean_inline(line[3:]), h1_style))
        elif line.startswith('### '):
            flowables.append(Paragraph(clean_inline(line[4:]), h2_style))
        elif line.startswith('#### '):
            flowables.append(Paragraph(clean_inline(line[5:]), h3_style))
        # HR
        elif line.strip() == '---':
            flowables.append(Spacer(1, 4))
            flowables.append(HRFlowable(width='100%', thickness=0.5,
                                         color=colors.HexColor('#9fa8da')))
            flowables.append(Spacer(1, 4))
        # Bullets
        elif line.startswith('- ') or line.startswith('* '):
            text = line[2:]
            # Sub-bullet indentation already ignored, just use 2-space indent
            flowables.append(Paragraph(
                '• ' + clean_inline(text), bullet_style))
        elif re.match(r'^\d+\. ', line):
            text = re.sub(r'^\d+\. ', '', line)
            flowables.append(Paragraph(
                clean_inline(text), bullet_style))
        # Blockquote
        elif line.startswith('> '):
            q_style = ParagraphStyle('Quote', parent=body_style,
                                      leftIndent=12, textColor=colors.HexColor('#555555'),
                                      borderPadding=(2, 2, 2, 6),
                                      backColor=colors.HexColor('#fffde7'))
            flowables.append(Paragraph(clean_inline(line[2:]), q_style))
        # Empty line
        elif line.strip() == '':
            flowables.append(Spacer(1, 4))
        # Normal paragraph
        else:
            flowables.append(Paragraph(clean_inline(line), body_style))

        i += 1

    if in_table:
        flush_table()

    return flowables


def convert_md_to_pdf(md_path, pdf_path):
    print(f"Converting: {os.path.basename(md_path)} -> {os.path.basename(pdf_path)}")
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    story = md_to_paragraphs(md_text, styles)
    doc.build(story)
    size_kb = os.path.getsize(pdf_path) // 1024
    print(f"  -> Done: {size_kb} KB")


DOCS_DIR = r"d:\NIDS-DL-main\NIDS-DL-main\docs"
OUT_DIR = r"d:\NIDS-DL-main\NIDS-DL-main\docs\pdf_exports"
os.makedirs(OUT_DIR, exist_ok=True)

files = [
    ("PROJECT_REPORT.md", "NIDS_Project_Report.pdf"),
    ("MODEL_COMPARISON.md", "NIDS_Model_Comparison.pdf"),
    ("3rd_Review_Progress.md", "NIDS_3rd_Review.pdf"),
]

for md_name, pdf_name in files:
    md_path = os.path.join(DOCS_DIR, md_name)
    pdf_path = os.path.join(OUT_DIR, pdf_name)
    try:
        convert_md_to_pdf(md_path, pdf_path)
    except Exception as e:
        print(f"  ERROR on {md_name}: {e}")

print("\nAll PDFs saved to:", OUT_DIR)
