#!/usr/bin/env python3
"""REPORT.md -> REPORT.html -> REPORT.pdf (A4, 10pt, same page style as the proposal), then print the page count.
Needs the venv's `markdown` package and LibreOffice: /media/vlsi/NVMESSD/fused-moe/venv/bin/python scripts/report_pdf.py
"""
import re
import subprocess
from pathlib import Path

import markdown

W = Path('/home/vlsi/fused-moe')
CSS = '''
@page { size: 21cm 29.7cm; margin-left: 2cm; margin-right: 2cm; margin-top: 1.4cm; margin-bottom: 1.3cm; }
body { font-family: "Liberation Serif", serif; font-size: 10pt; line-height: 1.08; }
h1 { font-family: "Liberation Serif", serif; font-size: 13pt; font-weight: bold; text-align: center; margin: 0 0 2pt 0; }
h2 { font-size: 11pt; margin: 7pt 0 2pt 0; }
h3 { font-size: 10pt; margin: 5pt 0 2pt 0; }
p, li { margin: 0 0 3pt 0; text-align: justify; }
ul, ol { margin: 0 0 3pt 0; padding-left: 14pt; }
table { border-collapse: collapse; margin: 3pt auto; font-size: 9pt; }
th, td { border-top: 0.5pt solid #888; border-bottom: 0.5pt solid #888; padding: 1pt 4pt; }
img { width: 13.5cm; }
code { font-family: "Liberation Mono", monospace; font-size: 8.5pt; }
pre { font-size: 8pt; page-break-inside: avoid; }
'''


def sized_img(m):
    """LibreOffice ignores CSS image widths: give each figure explicit pixel sizes (13.5 cm wide at 96 dpi)."""
    from PIL import Image
    w, h = Image.open(W / m.group(2)).size
    width = 510
    return f'<img alt="{m.group(1)}" src="{m.group(2)}" width="{width}" height="{round(width * h / w)}" />'


def main():
    md = (W / 'REPORT.md').read_text()
    md = re.sub(r'<!--.*?-->\n?', '', md, flags=re.S)
    body = markdown.markdown(md, extensions=['tables', 'fenced_code'])
    body = re.sub(r'<img alt="([^"]*)" src="([^"]+)" />', sized_img, body)
    title = re.search(r'^# (.+)$', md, flags=re.M).group(1)
    html = f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head>\n<body>\n{body}\n</body></html>\n'
    (W / 'REPORT.html').write_text(html)
    # Writer's HTML import; the Writer/Web export pushed the first element onto a blank first page
    subprocess.run(['soffice', '--headless', '--infilter=HTML (StarWriter)', '--convert-to', 'pdf:writer_pdf_Export',
                    '--outdir', str(W), str(W / 'REPORT.html')],
                   check=True, capture_output=True, timeout=180)
    info = subprocess.run(['pdfinfo', str(W / 'REPORT.pdf')], capture_output=True, text=True).stdout
    print(next(line for line in info.splitlines() if line.startswith('Pages')))


if __name__ == '__main__':
    main()
