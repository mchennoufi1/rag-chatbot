import os
import pandas as pd
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


def jpeg_to_pdf(image_path: str, output_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    img.save(output_path, "PDF", resolution=100.0)
    return output_path


def csv_to_pdf(csv_path: str, output_path: str) -> str:
    df = pd.read_csv(csv_path)
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"Data: {os.path.basename(csv_path)}", styles["Title"]))
    elements.append(Spacer(1, 12))

    data = [df.columns.tolist()] + df.values.tolist()
    data = [[str(cell) for cell in row] for row in data]

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    elements.append(table)
    doc.build(elements)
    return output_path