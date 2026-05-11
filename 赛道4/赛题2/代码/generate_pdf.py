from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

BASE_DIR = "/mnt/storage2/zyc/CIM比赛/赛道4/赛题2"
DOC_DIR = os.path.join(BASE_DIR, "文档")

FONT_PATHS = [
    ('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', 'DroidSansFallback'),
    ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 'NotoSansCJK'),
]

FONT_AVAILABLE = False
for font_path, font_name in FONT_PATHS:
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont('Chinese', font_path))
            FONT_AVAILABLE = True
            print(f"Using font: {font_path}")
            break
        except Exception as e:
            print(f"Failed to load {font_path}: {e}")
            continue

if not FONT_AVAILABLE:
    print("Warning: No Chinese font found, PDF will have display issues")

def create_pdf():
    output_path = os.path.join(BASE_DIR, '技术报告.pdf')
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    font_main = 'Chinese' if FONT_AVAILABLE else 'Helvetica'

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title_CN',
                              fontName=font_main,
                              fontSize=22,
                              alignment=TA_CENTER,
                              spaceAfter=20,
                              spaceBefore=30))
    styles.add(ParagraphStyle(name='Subtitle',
                              fontName=font_main,
                              fontSize=14,
                              alignment=TA_CENTER,
                              spaceAfter=8))
    styles.add(ParagraphStyle(name='Chapter',
                              fontName=font_main,
                              fontSize=16,
                              alignment=TA_LEFT,
                              spaceAfter=12,
                              spaceBefore=18))
    styles.add(ParagraphStyle(name='Section',
                              fontName=font_main,
                              fontSize=13,
                              alignment=TA_LEFT,
                              spaceAfter=8,
                              spaceBefore=10))
    styles.add(ParagraphStyle(name='Body_CN',
                              fontName=font_main,
                              fontSize=11,
                              alignment=TA_JUSTIFY,
                              spaceAfter=6,
                              leading=16))

    story = []

    story.append(Paragraph('Analog CIM Design Based on FET Devices', styles['Title_CN']))
    story.append(Paragraph('赛道四：存算设计 - 赛题二', styles['Subtitle']))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph('团队 / Team: HDUer', styles['Subtitle']))
    story.append(Paragraph('成员 / Member: 1', styles['Subtitle']))
    story.append(Paragraph('日期 / Date: 2026-05-11', styles['Subtitle']))
    story.append(PageBreak())

    chapters = [
        ('01_器件原理.md', '第一章 FeFET器件的工作原理及特性介绍', 'Chapter 1: FeFET Device Principles'),
        ('02_工作区间.md', '第二章 工作区间选择的分析', 'Chapter 2: Work Region Analysis'),
        ('03_正负信号处理.md', '第三章 正负信号处理方案', 'Chapter 3: Sign Handling'),
        ('04_线性度分析.md', '第四章 线性度分析与补偿', 'Chapter 4: Linearity Analysis'),
        ('05_精度分析.md', '第五章 多比特精度分析与评估', 'Chapter 5: Precision Analysis'),
        ('06_总结讨论.md', '第六章 总结与讨论', 'Chapter 6: Summary'),
        ('参考文献.md', '参考文献', 'References'),
    ]

    for filename, title_cn, title_en in chapters:
        filepath = os.path.join(DOC_DIR, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        story.append(Paragraph(title_en, styles['Chapter']))
        story.append(Paragraph(title_cn, styles['Section']))
        story.append(Spacer(1, 0.3*cm))

        lines = content.split('\n')
        table_data = []

        for line in lines:
            line = line.rstrip()

            if line.startswith('```') or line.startswith('~~~'):
                continue

            if line.startswith('|'):
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if all(parts):
                    table_data.append(parts)
                continue
            elif table_data:
                if len(table_data) > 1:
                    col_count = len(table_data[0])
                    col_width = (17 * cm) / col_count
                    t = Table(table_data, colWidths=[col_width] * col_count)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.3, 0.5)),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('FONTNAME', (0, 0), (-1, -1), font_main),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 4),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 0.3*cm))
                table_data = []

            if line.startswith('# '):
                continue
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], styles['Section']))
            elif line.startswith('### '):
                story.append(Paragraph(line[4:], styles['Body_CN']))
            elif line.startswith('**') and line.endswith('**'):
                story.append(Paragraph(f"<b>{line.replace('**', '')}</b>", styles['Body_CN']))
            elif line.startswith('- '):
                story.append(Paragraph(f"• {line[2:]}", styles['Body_CN']))
            elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. '):
                story.append(Paragraph(line, styles['Body_CN']))
            elif line.strip() and not line.startswith('---') and not line.startswith('*'):
                story.append(Paragraph(line, styles['Body_CN']))

        story.append(PageBreak())

    doc.build(story)
    print(f"PDF saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_pdf()
