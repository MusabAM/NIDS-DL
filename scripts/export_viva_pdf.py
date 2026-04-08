#!/usr/bin/env python3
"""Convert VIVA_PREP.md to PDF."""
import os, re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)

def clean_inline(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = text.replace('✅','[OK]').replace('⚠️','[!]').replace('🔜','[>]')
    text = text.replace('🟢','[>]').replace('★','*').replace('↑','+')
    text = text.replace('✓','[OK]').replace('→','->').replace('💪','')
    return text

def convert(md_path, pdf_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle('T', fontSize=20, spaceAfter=8, alignment=1,
                              textColor=colors.HexColor('#1a237e'), fontName='Helvetica-Bold')
    h1_s = ParagraphStyle('H1', fontSize=15, spaceBefore=12, spaceAfter=5,
                           textColor=colors.HexColor('#283593'), fontName='Helvetica-Bold')
    h2_s = ParagraphStyle('H2', fontSize=12, spaceBefore=9, spaceAfter=4,
                           textColor=colors.HexColor('#1565c0'), fontName='Helvetica-Bold')
    h3_s = ParagraphStyle('H3', fontSize=10, spaceBefore=7, spaceAfter=3,
                           textColor=colors.HexColor('#0277bd'), fontName='Helvetica-Bold')
    h4_s = ParagraphStyle('H4', fontSize=9.5, spaceBefore=5, spaceAfter=2,
                           textColor=colors.HexColor('#01579b'), fontName='Helvetica-Bold')
    body_s = ParagraphStyle('B', fontSize=9, leading=13, spaceAfter=3)
    bullet_s = ParagraphStyle('BL', fontSize=9, leading=12, spaceAfter=2, leftIndent=14)
    code_s = ParagraphStyle('C', fontSize=7.5, leading=11, spaceAfter=4,
                             backColor=colors.HexColor('#f5f5f5'), leftIndent=6,
                             fontName='Courier', borderPad=3)
    q_s = ParagraphStyle('Q', fontSize=9.5, leading=13, spaceAfter=2,
                          textColor=colors.HexColor('#1a237e'), fontName='Helvetica-Bold',
                          backColor=colors.HexColor('#e8eaf6'), borderPad=3, leftIndent=6)
    a_s = ParagraphStyle('A', fontSize=9, leading=13, spaceAfter=5,
                          leftIndent=10, backColor=colors.HexColor('#f9fbe7'), borderPad=3)

    story = []
    lines = md_text.split('\n')
    i = 0
    in_code = False
    code_buf = []
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal table_rows, in_table
        if table_rows:
            data = []
            for ri, row in enumerate(table_rows):
                cells = [Paragraph(clean_inline(c.strip()),
                         ParagraphStyle('TC', fontSize=8, leading=10,
                         fontName='Helvetica-Bold' if ri==0 else 'Helvetica'))
                         for c in row]
                data.append(cells)
            t = Table(data, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#283593')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8f9fa'),colors.HexColor('#e8eaf6')]),
                ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#9fa8da')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
                ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
            ]))
            story.append(t)
            story.append(Spacer(1,6))
        table_rows.clear(); in_table = False

    prev_was_blank = False
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('```'):
            if in_table: flush_table()
            if in_code:
                story.append(Paragraph(
                    '<br/>'.join(l.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace(' ','&nbsp;') for l in code_buf),
                    code_s))
                code_buf.clear(); in_code = False
            else:
                in_code = True
            i += 1; continue
        if in_code:
            code_buf.append(line); i += 1; continue
        if line.startswith('|') and line.endswith('|'):
            cells = line.strip('|').split('|')
            if all(re.match(r'^[-: ]+$', c.strip()) for c in cells if c.strip()):
                i += 1; continue
            in_table = True; table_rows.append(cells); i += 1; continue
        elif in_table:
            flush_table()
        if line.startswith('# '):
            story.append(Paragraph(clean_inline(line[2:]), title_s))
        elif line.startswith('## '):
            story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#7986cb'), spaceAfter=2))
            story.append(Paragraph(clean_inline(line[3:]), h1_s))
        elif line.startswith('### '):
            story.append(Paragraph(clean_inline(line[4:]), h2_s))
        elif line.startswith('#### '):
            story.append(Paragraph(clean_inline(line[5:]), h3_s))
        elif line.startswith('##### '):
            story.append(Paragraph(clean_inline(line[6:]), h4_s))
        elif line.strip() == '---':
            story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#9fa8da'), spaceBefore=4, spaceAfter=4))
        elif line.startswith('> '):
            story.append(Paragraph(clean_inline(line[2:]), a_s))
        elif re.match(r'^\*\*Q\d+', line):
            story.append(Paragraph(clean_inline(line.lstrip('- *')), q_s))
        elif line.startswith('- ') or line.startswith('* '):
            story.append(Paragraph('• ' + clean_inline(line[2:]), bullet_s))
        elif re.match(r'^\d+\. ', line):
            story.append(Paragraph(clean_inline(re.sub(r'^\d+\. ', '', line)), bullet_s))
        elif line.strip() == '':
            if not prev_was_blank:
                story.append(Spacer(1, 4))
        else:
            story.append(Paragraph(clean_inline(line), body_s))
        prev_was_blank = (line.strip() == '')
        i += 1
    if in_table: flush_table()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                             rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    doc.build(story)
    print(f"Done: {os.path.basename(pdf_path)} ({os.path.getsize(pdf_path)//1024} KB)")

convert(
    r"d:\NIDS-DL-main\NIDS-DL-main\docs\VIVA_PREP.md",
    r"d:\NIDS-DL-main\NIDS-DL-main\docs\pdf_exports\NIDS_VIVA_PREP.pdf"
)
