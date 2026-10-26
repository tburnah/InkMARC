#!/usr/bin/env python3
"""Build the final InkMARC fallback-tracking technical report."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / ".analysis_tracking" / "final_122250"
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "InkMARC_Fallback_Surface_Tracking_Report_20260922.docx"
METRICS = ANALYSIS / "metrics.json"
TIMELINE = ANALYSIS / "timeline.png"
HAND_SHEET = ANALYSIS / "hand_interval_6frame.png"
ARCHITECTURE = ANALYSIS / "architecture.png"

FONT = "Calibri"
NAVY = "20384F"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
TEAL = "1D7773"
GOLD = "B07A15"
MUTED = "667085"
LIGHT_BLUE = "EAF2F8"
LIGHT_GRAY = "F2F4F7"
MID_GRAY = "D7DCE2"
WHITE = "FFFFFF"
BLACK = "000000"
RISK = "9B1C1C"


def set_run_font(run, *, size=None, bold=None, italic=None, color=None, name=FONT):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_table_geometry(table, widths_dxa):
    if sum(widths_dxa) != 9360:
        raise ValueError(f"Table widths must total 9360 DXA, got {sum(widths_dxa)}")
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), "9360")
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    layout = tbl_pr.first_child_found_in("w:tblLayout")
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths_dxa[index]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc, headers, rows, widths_dxa, *, numeric_columns=()):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for col, header in enumerate(headers):
        cell = table.rows[0].cells[col]
        set_cell_shading(cell, LIGHT_GRAY)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col in numeric_columns else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(str(header))
        set_run_font(run, size=9, bold=True, color=NAVY)
    set_repeat_table_header(table.rows[0])
    for row_data in rows:
        cells = table.add_row().cells
        for col, value in enumerate(row_data):
            p = cells[col].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col in numeric_columns else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(str(value))
            set_run_font(run, size=8.7, color=BLACK)
    set_table_geometry(table, widths_dxa)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def set_paragraph_shading(paragraph, fill, border=None):
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    p_pr.append(shd)
    if border:
        p_bdr = OxmlElement("w:pBdr")
        left = OxmlElement("w:left")
        left.set(qn("w:val"), "single")
        left.set(qn("w:sz"), "18")
        left.set(qn("w:space"), "8")
        left.set(qn("w:color"), border)
        p_bdr.append(left)
        p_pr.append(p_bdr)


def add_callout(doc, label, text, *, fill=LIGHT_BLUE, border=BLUE):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.16)
    p.paragraph_format.right_indent = Inches(0.08)
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.10
    p.paragraph_format.keep_together = True
    set_paragraph_shading(p, fill, border)
    lead = p.add_run(f"{label}: ")
    set_run_font(lead, size=10.5, bold=True, color=NAVY)
    body = p.add_run(text)
    set_run_font(body, size=10.5, color=BLACK)
    return p


def add_hyperlink(paragraph, text, url):
    relation_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relation_id)
    run = OxmlElement("w:r")
    run_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLUE)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    run_pr.append(color)
    run_pr.append(underline)
    run.append(run_pr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    value = OxmlElement("w:t")
    value.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, value, end])
    set_run_font(run, size=8.5, color=MUTED)


def add_numbering_definition(doc, *, bullet):
    numbering = doc.part.numbering_part.element
    abstract_ids = [int(node.get(qn("w:abstractNumId"))) for node in numbering.findall(qn("w:abstractNum"))]
    num_ids = [int(node.get(qn("w:numId"))) for node in numbering.findall(qn("w:num"))]
    abstract_id = max(abstract_ids, default=0) + 1
    num_id = max(num_ids, default=0) + 1
    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)
    level = OxmlElement("w:lvl")
    level.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "bullet" if bullet else "decimal")
    lvl_text = OxmlElement("w:lvlText")
    lvl_text.set(qn("w:val"), "-" if bullet else "%1.")
    suffix = OxmlElement("w:suff")
    suffix.set(qn("w:val"), "tab")
    p_pr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "720")
    tabs.append(tab)
    indent = OxmlElement("w:ind")
    indent.set(qn("w:left"), "720")
    indent.set(qn("w:hanging"), "360")
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:after"), "160")
    spacing.set(qn("w:line"), "280")
    spacing.set(qn("w:lineRule"), "auto")
    p_pr.extend([tabs, indent, spacing])
    level.extend([start, num_fmt, lvl_text, suffix, p_pr])
    if bullet:
        r_pr = OxmlElement("w:rPr")
        fonts = OxmlElement("w:rFonts")
        fonts.set(qn("w:ascii"), FONT)
        fonts.set(qn("w:hAnsi"), FONT)
        r_pr.append(fonts)
        level.append(r_pr)
    abstract.append(level)
    numbering.append(abstract)
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_ref = OxmlElement("w:abstractNumId")
    abstract_ref.set(qn("w:val"), str(abstract_id))
    num.append(abstract_ref)
    numbering.append(num)
    return num_id


def add_list_item(doc, text, num_id):
    p = doc.add_paragraph()
    num_pr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num = OxmlElement("w:numId")
    num.set(qn("w:val"), str(num_id))
    num_pr.extend([ilvl, num])
    p._p.get_or_add_pPr().append(num_pr)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.167
    run = p.add_run(text)
    set_run_font(run, size=11, color=BLACK)
    return p


def add_body(doc, text, *, bold_lead=None):
    p = doc.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        lead = p.add_run(bold_lead)
        set_run_font(lead, size=11, bold=True, color=NAVY)
        run = p.add_run(text[len(bold_lead):])
        set_run_font(run, size=11, color=BLACK)
    else:
        run = p.add_run(text)
        set_run_font(run, size=11, color=BLACK)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.keep_with_next = False
    run = p.add_run(text)
    set_run_font(run, size=9, italic=True, color=MUTED)
    return p


def add_figure(doc, path, caption, *, width=6.25):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_together = True
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    drawing = run._r.xpath(".//wp:docPr")
    if drawing:
        drawing[0].set("descr", caption)
        drawing[0].set("title", caption.split(".", 1)[0])
    add_caption(doc, caption)


def add_source_note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_run_font(run, size=8.5, italic=True, color=MUTED)
    return p


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for name, size, color, before, after in [
        ("Title", 30, NAVY, 0, 8),
        ("Subtitle", 14, MUTED, 0, 18),
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ]:
        style = styles[name]
        style.font.name = FONT
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
        style.font.size = Pt(size)
        style.font.bold = name != "Subtitle"
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    header = section.header
    p = header.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run("INKMARC  |  FALLBACK SURFACE TRACKING")
    set_run_font(run, size=8.5, bold=True, color=MUTED)
    footer = section.footer
    add_page_number(footer.paragraphs[0])


def create_architecture_figure(path):
    width, height = 1800, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 27)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 28)
    title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 34)
    draw.text((60, 35), "ARCore-independent surface tracking pipeline", font=title, fill="#20384F")

    boxes = [
        (70, 145, 330, 320, "Camera2", "Portrait preview\nYUV analysis\nH.264 recording"),
        (410, 145, 720, 320, "Visual front end", "Sparse pyramidal LK\nFB consistency\nPer-track reliability"),
        (800, 145, 1110, 320, "Plane estimator", "Gravity-fixed normal\nGyro/rotation prior\n3-parameter robust fit"),
        (1190, 145, 1510, 320, "Safety and output", "Coverage + innovation gates\nIMU hold and relock\nDisplay-time prediction"),
        (520, 460, 850, 640, "Occlusion evidence", "MediaPipe hand hull\nDynamic residual + shadow\nCombined feature mask"),
        (1000, 460, 1370, 640, "Synchronized evidence", "IMU + camera timing\nTracker telemetry + quad\nOffline annotated replay"),
    ]
    for x0, y0, x1, y1, heading, body in boxes:
        fill = "#EAF2F8" if y0 < 400 else "#F2F4F7"
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=fill, outline="#2E74B5", width=3)
        draw.text((x0 + 22, y0 + 18), heading, font=bold, fill="#20384F")
        draw.multiline_text((x0 + 22, y0 + 65), body, font=small, fill="#000000", spacing=8)

    def arrow(x0, y0, x1, y1):
        draw.line((x0, y0, x1, y1), fill="#1D7773", width=7)
        draw.polygon([(x1, y1), (x1 - 22, y1 - 13), (x1 - 22, y1 + 13)], fill="#1D7773")

    arrow(330, 232, 410, 232)
    arrow(720, 232, 800, 232)
    arrow(1110, 232, 1190, 232)
    draw.line((685, 460, 685, 355), fill="#1D7773", width=7)
    draw.polygon([(685, 335), (672, 359), (698, 359)], fill="#1D7773")
    draw.line((1185, 460, 1185, 355), fill="#1D7773", width=7)
    draw.polygon([(1185, 335), (1172, 359), (1198, 359)], fill="#1D7773")
    draw.text((70, 690), "Active live path: Camera2 -> masked sparse tracking -> constrained visual-inertial plane update -> overlay", font=regular, fill="#667085")
    image.save(path)


def heading(doc, text, level=1):
    return doc.add_heading(text, level=level)


def add_reference(doc, number, citation, url):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(5)
    run = p.add_run(f"[{number}] {citation} ")
    set_run_font(run, size=9.2, color=BLACK)
    add_hyperlink(p, f"Reference [{number}] full text", url)


def build_report():
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    create_architecture_figure(ARCHITECTURE)
    REPORTS.mkdir(parents=True, exist_ok=True)

    doc = Document()
    configure_styles(doc)
    configure_document(doc)
    bullet_id = add_numbering_definition(doc, bullet=True)
    number_id = add_numbering_definition(doc, bullet=False)

    # Cover
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(84)
    kicker = doc.add_paragraph()
    kicker.alignment = WD_ALIGN_PARAGRAPH.CENTER
    kicker.paragraph_format.space_after = Pt(14)
    run = kicker.add_run("TECHNICAL REPORT  |  FINAL OBSERVATIONAL EVALUATION")
    set_run_font(run, size=10, bold=True, color=GOLD)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("InkMARC ARCore-Independent\nHorizontal-Surface Tracking")
    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Implementation, measured performance, research lineage, and contribution")

    date = doc.add_paragraph()
    date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date.paragraph_format.space_after = Pt(28)
    run = date.add_run("Final run: 22 September 2026  |  Session InkMARC_20260922_122250")
    set_run_font(run, size=10.5, bold=True, color=DARK_BLUE)

    add_callout(
        doc,
        "Headline result",
        "After manual placement, the overlay remained available on 100% of 488 analyzed frames over 32.46 s, including the visible-hand interval. This continuity depended heavily on prediction and hold: only 50.0% of post-placement frames accepted a new constrained visual correction, while 47.1% used a protected inertial/predictive hold. Median end-to-end camera-to-result latency was 113.4 ms (p95 189.8 ms).",
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(24)
    run = p.add_run("Prepared as thesis-supporting evidence for the wider stylus-tip inference system")
    set_run_font(run, size=10, italic=True, color=MUTED)

    doc.add_page_break()

    heading(doc, "Executive summary", 1)
    add_body(
        doc,
        "InkMARC now runs on a phone that does not support ARCore by making Camera2 and the fallback tracker the primary execution path. ARCore remains compiled as an optional future enhancement, but it is disabled at runtime and marked optional in the Android manifest. Stylus-tip inference and depth inference are disabled for this evaluation; the active system is dedicated to horizontal-plane tracking, hand/dynamic occlusion handling, overlay rendering, and synchronized recording.",
    )
    add_body(
        doc,
        "The central design choice is to replace an unconstrained eight-degree-of-freedom image homography with a three-parameter translation-over-plane-distance estimate. Device orientation fixes camera rotation and gravity fixes the plane normal. Sparse visual correspondences therefore estimate only the residual translation term. The implementation combines this physical constraint with per-track reliability, weighted robust fitting, spatial support gates, rolling-shutter-aware orientation, innovation rejection, controlled relocking, and a prediction loop that projects the overlay toward display time.",
    )
    add_body(
        doc,
        "The final run demonstrates a materially more robust system than the earlier image-only variants: the overlay survives the hand passage and remains geometrically plausible in the annotated replay. The same evidence also exposes the remaining limitation. High processing and delivery latency, plus conservative gating, means the live output often advances by prediction rather than by a newly accepted visual update. The system is therefore best characterized as a successful ARCore-independent prototype with strong continuity and promising occlusion robustness, but not yet a validated high-accuracy pose tracker.",
    )

    heading(doc, "Key measured findings", 2)
    for text in [
        "Raw video: 35.96 s, 1,069 H.264 frames at 1280 x 720 with portrait orientation metadata (29.73 fps average).",
        "Tracker evidence: 574 analyzed camera rows; 488 post-placement rows spanning 32.46 s.",
        "Post-placement overlay availability: 488/488 analyzed rows (100%). Tracking-or-seeded state: 84.0%.",
        "Fresh constrained visual corrections: 244/488 rows (50.0%). Protected hold/prediction: 230/488 (47.1%). Longest hold: 2.45 s.",
        "Latency: 57.6 ms median camera delivery, 50.8 ms median processing, and 113.4 ms median total; total p95 was 189.8 ms.",
        "Accepted constrained fits: 2.19 px median plane-fit RMSE (p95 4.91 px), 0.740 median overall support ratio, and 31.26 median effective weighted support.",
        "Visible-hand interval (approximately 20-27 s): overlay availability remained 100%; the dynamic mask was active on 100% of analyzed rows, but fresh visual corrections fell to 38.1% and holds rose to 59.5%.",
        "MediaPipe hand masking was enabled but registered a non-zero semantic mask on only one analyzed row. The dynamic residual mask, not the landmark mask, supplied most occlusion protection in this recording.",
    ]:
        add_list_item(doc, text, bullet_id)

    heading(doc, "1. Scope and evaluation claim", 1)
    add_body(
        doc,
        "This report evaluates the implementation represented by the current source tree and the synchronized recording pair InkMARC_20260922_122250.mp4 / InkMARC_20260922_122250_sensors.csv. It also uses the completed offline replay experiments on InkMARC_20260921_183433 to document design decisions and negative results. No new tracker experiment is introduced here.",
    )
    add_callout(
        doc,
        "Validity boundary",
        "The session contains no independent camera-pose reference, fiducial target, motion-capture trajectory, or metrically surveyed rectangle. Accordingly, this report measures availability, latency, internal residuals, support, gating behavior, sensor quality, and qualitative overlay plausibility. It does not report absolute translation error, rotation error, drift in metres, or a ground-truth success rate.",
        fill="FFF7E6",
        border=GOLD,
    )

    heading(doc, "2. Current implementation", 1)
    heading(doc, "2.1 Runtime configuration", 2)
    add_table(
        doc,
        ["Subsystem", "Current setting", "Meaning for this report"],
        [
            ("ARCore", "Off", "No ARCore session is created; the renderer returns early."),
            ("Fallback tracker", "On", "Primary and only active surface-tracking path."),
            ("Session video / sensors", "On / on", "Raw portrait H.264 plus synchronized CSV telemetry."),
            ("Stylus inference", "Off", "No stylus-tip or touch inference influences the result."),
            ("DA3 depth", "Off", "No monocular depth estimate is used."),
            ("Hand occlusion mask", "On", "MediaPipe landmark mask is requested asynchronously."),
            ("Dynamic mask", "On", "IMU-compensated temporal residual rejects moving foreground/shadow."),
            ("Occlusion track memory", "Off", "Implementation retained, but disabled after neutral replay results."),
        ],
        [2400, 1500, 5460],
        numeric_columns=(1,),
    )
    add_source_note(doc, "Source: InkMarcActivity.kt, AndroidManifest.xml, and FallbackSurfaceTracker.kt in the evaluated source tree.")

    add_body(
        doc,
        "Although the conversation began around InkMarcRenderer.kt, that class is not the active rendering path in this configuration. With ENABLE_ARCORE=false, InkMarcActivity creates StandaloneCameraController and FallbackSurfaceTracker; the rectangle is drawn by FallbackOverlayView over the direct TextureView preview. The manifest declares android.hardware.camera.ar as not required and marks com.google.ar.core as optional, allowing installation on non-ARCore phones.",
    )

    heading(doc, "2.2 Data and control flow", 2)
    add_figure(
        doc,
        ARCHITECTURE,
        "Figure 1. Active implementation pipeline. Camera2 supplies the preview, YUV analysis images, and recording surface; visual and inertial evidence are fused in a constrained plane model before safety gates and output prediction.",
        width=6.35,
    )

    heading(doc, "2.3 Camera geometry, portrait mapping, and timing", 2)
    add_body(
        doc,
        "A single Camera2 capture session targets the TextureView preview, an ImageReader for YUV analysis, and MediaRecorder for H.264 output. The selected stream is shared across all three outputs. CameraCoordinateMapper maintains the explicit mapping between sensor-oriented analysis pixels and the portrait view, while the TextureView receives only an aspect-ratio correction and centre crop. This separation removed the earlier 90-degree axis error and portrait squashing without rotating the direct preview twice.",
    )
    add_body(
        doc,
        "For every image, the controller retrieves exposure time and SENSOR_ROLLING_SHUTTER_SKEW. CameraFrameTiming converts an image row into its exposure-centre timestamp. On the evaluated phone, exposure was 10.001 ms and full-frame rolling-shutter skew was 15.041 ms, giving a 12.521 ms offset from the first-row timestamp to the centre-exposure timestamp. The plane estimator can therefore use a different interpolated orientation for each correspondence row rather than treating the whole frame as instantaneous.",
    )

    heading(doc, "2.4 Gravity-constrained planar motion", 2)
    add_body(
        doc,
        "The game rotation vector supplies device orientation and the gyroscope supports short extrapolation. At placement, gravity defines the horizontal plane normal in the reference camera. For a calibrated camera observing that plane, the implementation uses:",
    )
    equation = doc.add_paragraph()
    equation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation.paragraph_format.space_before = Pt(5)
    equation.paragraph_format.space_after = Pt(8)
    run = equation.add_run("H = K (R + u n^T) K^-1,    where u = t / d")
    set_run_font(run, size=12, italic=True, color=NAVY)
    add_body(
        doc,
        "K is the camera intrinsic matrix, R is the IMU-derived relative camera rotation, n is the gravity-derived reference-plane normal, and u is the three-element translation divided by plane distance. The visual solver estimates u only. This converts a free projective warp into a physically meaningful, lower-dimensional problem while retaining the perspective change caused by moving the phone over the table.",
    )

    heading(doc, "2.5 Visual front end and feature reliability", 2)
    for text in [
        "Good-features-to-track detection supplies up to 90 active points from as many as 360 corner candidates, with full-frame support allowed so the rectangle can move partly off-screen.",
        "Pyramidal Lucas-Kanade flow uses the IMU-plane transform as its initial estimate. A forward-backward consistency check runs periodically, alongside LK error and frame-to-frame surface-consistency filtering.",
        "Each track maintains a static/reliability probability. Cauchy-shaped evidence combines LK error, forward-backward error, and disagreement with the inertial prediction. Reliability falls quickly on poor evidence and recovers slowly.",
        "The constrained fit uses the track reliabilities as weights. Effective support is the sum of accepted weights, and a 4 x 3 image grid verifies that weighted support is spatially distributed rather than concentrated in one patch.",
        "Low effective support, poor weighted coverage, high residual, implausible translation, or excessive innovation causes the system to retain a predicted plane instead of accepting the visual update.",
    ]:
        add_list_item(doc, text, bullet_id)

    heading(doc, "2.6 Relock, prediction, and display update", 2)
    add_body(
        doc,
        "A rejected visual measurement is not applied immediately. Consistent rejected hypotheses accumulate in PlaneRelockBuffer; an accepted proposal is blended over five frames and is cancelled if subsequent measurements disagree. Accepted translation-over-height terms update a filtered constant-velocity model. Prediction is capped at 140 ms and each component is bounded. A Choreographer callback reprojects the rectangle at display cadence, independently of the slower image-analysis callback.",
    )
    add_body(
        doc,
        "This design explains both the improvement and the residual lag. The final run's accepted measurements were already 122.2 ms old at the output time on the median frame (p95 190.5 ms). The predictor can cover most median delay, but its 140 ms cap cannot fully compensate the latency tail, and constant velocity cannot anticipate direction changes in translation.",
    )

    heading(doc, "2.7 Occlusion handling", 2)
    add_body(
        doc,
        "The semantic layer runs MediaPipe Hand Landmarker asynchronously at no more than 10 Hz. It fills the landmark convex hull, extends a corridor beyond the wrist to include the forearm, dilates the result, and retains the two most recent masks for up to 650 ms. The image-space mask is excluded from seeding, optical-flow survival, and feature replenishment.",
    )
    add_body(
        doc,
        "The geometric-photometric layer compensates the previous half-resolution grayscale image using the IMU-predicted plane motion. It removes global brightness change, thresholds the residual, filters components morphologically, dilates them, and carries a decaying mask forward. This catches clipped hands and cast shadows that a landmark detector can miss. The two masks are OR-combined before tracking.",
    )

    heading(doc, "2.8 Synchronized evidence pipeline", 2)
    add_body(
        doc,
        "MediaRecorder writes the raw camera stream at a requested 30 fps, 8 Mbit/s, with portrait orientation metadata. SessionSensorRecorder writes accelerometer, gyroscope, gravity, linear acceleration, magnetic field, rotation vector, and game rotation vector samples together with camera timing, view-space quadrilateral coordinates, tracker state, latency, residual, support, mask, and rejection diagnostics. Both use Android elapsed-realtime nanoseconds. The raw MP4 deliberately contains no overlay; tools/annotate_session.py reconstructs the annotated MP4 from synchronized CSV coordinates for analysis.",
    )

    doc.add_page_break()
    heading(doc, "3. Evaluation method", 1)
    heading(doc, "3.1 Final on-device session", 2)
    add_table(
        doc,
        ["Item", "Observed value"],
        [
            ("Session", "InkMARC_20260922_122250"),
            ("Raw video", "35.958 s; 1,069 frames; 1280 x 720; H.264; 29.73 fps average"),
            ("Phone view", "1220 x 2712 portrait; 90 degree analysis rotation; centre crop"),
            ("CSV", "38,295 data rows: 574 camera rows and 37,715 sensor rows"),
            ("Placement", "First valid overlay at 3.324 s"),
            ("Operating interval", "32.464 s and 488 analyzed camera rows after placement"),
            ("Hand interval", "Approximately 20-27 s, identified by direct video review"),
        ],
        [2600, 6760],
    )
    add_source_note(doc, "Source: ffprobe metadata, synchronized CSV schema 7, and direct inspection of the annotated replay.")
    add_body(
        doc,
        "Metrics were computed from recorded telemetry without rerunning or retuning the tracker. Quantiles use all finite values in the relevant rows. Post-placement availability asks whether a complete quadrilateral exists. A fresh visual correction is identified by the presence of constrained-fit telemetry; a held row is the implementation's heldLastGoodTransform flag. The visible-hand interval is the only manually delimited segment.",
    )

    heading(doc, "3.2 Interpretation of the metrics", 2)
    for text in [
        "Availability measures continuity, not correctness. A predicted quadrilateral can remain present while accumulating translational error.",
        "Reprojection error is an internal consistency measure on selected correspondences. Low residual can coexist with bias if the selected features share the same wrong motion.",
        "Corner displacement is not reported as pose error because deliberate phone motion also moves the overlay.",
        "A single recording and one device do not establish generalisation. The results should be reported as an observational case study plus controlled replay ablations.",
    ]:
        add_list_item(doc, text, bullet_id)

    heading(doc, "4. Results", 1)
    heading(doc, "4.1 Continuity, update source, and latency", 2)
    add_table(
        doc,
        ["Measure", "Median / count", "p95 / ratio", "Interpretation"],
        [
            ("Post-placement overlay", "488 / 488 frames", "100.0%", "No analyzed-frame disappearance after placement"),
            ("Tracking or seeded state", "410 / 488 frames", "84.0%", "State machine still entered loss/insufficient states"),
            ("Accepted visual corrections", "244 / 488 frames", "50.0%", "Half of post-placement frames updated from constrained vision"),
            ("Protected hold", "230 / 488 frames", "47.1%", "Prediction carried the overlay through rejected or missing vision"),
            ("Longest continuous hold", "2.45 s", "-", "Meaningful interval without a newly trusted visual correction"),
            ("Delivery latency", "57.65 ms", "80.14 ms", "Exposure centre to analysis callback"),
            ("Processing latency", "50.76 ms", "120.56 ms", "Fallback tracker and masks on callback thread"),
            ("Total latency", "113.41 ms", "189.77 ms", "Exposure centre to recorded tracker result"),
            ("Output measurement age", "122.18 ms", "190.45 ms", "Age used by display-time prediction on accepted fits"),
        ],
        [2650, 1700, 1400, 3610],
        numeric_columns=(1, 2),
    )
    add_source_note(doc, "Source: final run CSV. Quantiles are empirical; the 682.9 ms processing maximum occurs in the end-of-session latency spike and is not treated as steady-state performance.")
    add_callout(
        doc,
        "Primary interpretation",
        "The system now fails safe: it rejects weak visual updates and preserves a plausible plane estimate instead of accepting large arbitrary homographies. The cost is that translation responsiveness depends on a capped constant-velocity predictor during almost half of the post-placement analysis rows.",
    )

    heading(doc, "4.2 Visual support quality", 2)
    add_table(
        doc,
        ["Accepted-fit diagnostic", "Median", "p95", "Maximum"],
        [
            ("Active tracks (all camera rows)", "36", "78", "90"),
            ("Inliers (all camera rows)", "22", "74", "88"),
            ("Raw visual reprojection error", "0.774 px", "1.342 px", "4.723 px"),
            ("Constrained plane-fit RMSE", "2.194 px", "4.909 px", "6.326 px"),
            ("Overall weighted support ratio", "0.740", "0.913", "0.963"),
            ("Effective plane support", "31.26", "60.95", "72.43"),
            ("Weighted 4 x 3 coverage", "0.583", "0.833", "0.917"),
        ],
        [3900, 1700, 1700, 2060],
        numeric_columns=(1, 2, 3),
    )
    add_source_note(doc, "Source: final run CSV. Constrained diagnostics are available on 244 accepted visual-correction rows.")
    add_body(
        doc,
        "The accepted fits are internally clean: median raw correspondence error is below one pixel and the constrained plane fit remains below five pixels at the 95th percentile. The more important robustness signal is not the high inlier ratio alone, but the combination of 31.26 effective support and 58.3% median weighted grid coverage. These measures prevent many locally coherent foreground or shadow tracks from dominating the plane estimate.",
    )

    add_figure(
        doc,
        TIMELINE,
        "Figure 2. Final-run telemetry. Orange bands show held output; the dynamic mask rises sharply during the hand/shadow interval, visual support becomes intermittent, and total latency remains commonly above 100 ms.",
        width=6.35,
    )

    heading(doc, "4.3 Occlusion episode", 2)
    add_table(
        doc,
        ["Interval", "Rows", "Overlay available", "Held", "Fresh visual correction", "Dynamic mask active", "Median masked area"],
        [
            ("Pre-hand (3.32-20 s)", "218", "100%", "39.4%", "57.8%", "8.3%", "0.0%"),
            ("Hand visible (20-27 s)", "126", "100%", "59.5%", "38.1%", "100%", "28.9%"),
            ("Occlusion tail (27-31.5 s)", "81", "100%", "66.7%", "30.9%", "97.5%", "25.6%"),
            ("Post-hand (31.5-35.8 s)", "63", "100%", "23.8%", "71.4%", "39.7%", "1.0%"),
        ],
        [2100, 700, 1250, 900, 1430, 1390, 1590],
        numeric_columns=(1, 2, 3, 4, 5, 6),
    )
    add_source_note(doc, "Source: final run CSV; interval labels follow direct video inspection. The dynamic mask includes hand, shadow, and other photometric changes.")
    add_body(
        doc,
        "The overlay remains present while the hand crosses the tracked region. During the hand-visible interval the dynamic mask is active on every analyzed row and removes a median 28.9% of the frame. The tracker responds conservatively: accepted visual corrections fall from 57.8% before the hand to 38.1%, while holds rise from 39.4% to 59.5%. This is the intended failure mode. The system sacrifices responsiveness to avoid fitting the moving hand.",
    )
    add_body(
        doc,
        "MediaPipe does not explain this result. Across the complete run, the semantic hand-mask area is non-zero on only one of 574 camera rows (at approximately 30.81 s), even though the hand is clearly visible for several seconds. The dynamic residual mask is active on 248 rows (43.2%) and therefore provides nearly all observed foreground protection. This is a useful negative result: semantic masking is implemented, but it is not yet dependable for clipped or side-entering hands on this device.",
    )
    add_figure(
        doc,
        HAND_SHEET,
        "Figure 3. Synchronized annotated frames through the hand episode. The rectangle remains available as the hand enters, then returns to visually unobstructed table support after the hand leaves. The images demonstrate continuity and plausible plane geometry, not ground-truth positional accuracy.",
        width=5.0,
    )

    heading(doc, "4.4 Sensor and timing quality", 2)
    add_table(
        doc,
        ["Sensor / timing source", "Estimated rate", "Median interval", "Relevant observation"],
        [
            ("Gyroscope", "397.95 Hz", "2.513 ms", "Dense angular-rate support for short extrapolation"),
            ("Game rotation vector", "99.49 Hz", "10.052 ms", "Primary orientation stream"),
            ("Gravity", "99.47 Hz", "10.052 ms", "Median magnitude 9.8066 m/s^2; highly stable norm"),
            ("Accelerometer", "198.98 Hz", "5.026 ms", "Recorded for diagnostics; not integrated for free translation"),
            ("Camera exposure", "Per frame", "10.001 ms", "Stable in this run"),
            ("Rolling-shutter skew", "Per frame", "15.041 ms", "Used for row-specific orientation"),
            ("Online camera-IMU offset", "Evaluated conditionally", "0.0 ms", "Estimator found no sustained residual improvement"),
        ],
        [2700, 1600, 1600, 3460],
        numeric_columns=(1, 2),
    )
    add_source_note(doc, "Source: final run CSV. Sensor rates are estimated from monotonic source timestamps.")
    add_body(
        doc,
        "The inertial streams are not the limiting factor in this session. They are high-rate, regular, and gravity magnitude is stable. The observed weakness lies in visual translation and pipeline delay: gravity and gyroscope constrain rotation very well, but an IMU cannot directly recover low-frequency phone translation without drift. That translation must still be corrected visually.",
    )

    heading(doc, "4.5 Completed offline ablations", 2)
    add_table(
        doc,
        ["Replay configuration", "Valid frames", "Catastrophic steps", "Hold frames", "Decision"],
        [
            ("Unmasked baseline", "311", "6", "0", "Rejected: foreground motion causes jumps"),
            ("Dynamic-mask baseline", "299", "1", "0", "Retained: large reduction in catastrophic steps"),
            ("Weighted fit without safety gate", "386", "7", "0", "Rejected: coverage rises at the cost of unstable updates"),
            ("Reliability + gated safe hold", "319", "0", "78", "Retained: no catastrophic proxy events"),
            ("Gated hold + dormant track memory", "318", "0", "90", "Disabled: only two tracks reacquired; no net gain"),
        ],
        [3300, 1200, 1500, 1200, 2160],
        numeric_columns=(1, 2, 3),
    )
    add_source_note(doc, "Source: replay experiments on InkMARC_20260921_183433. A catastrophic step is the experiment's >100 px corner-displacement proxy, not ground-truth pose error. Configurations are not all directly comparable outside that replay script.")
    add_body(
        doc,
        "The ablation history supports the current conservative design. Dynamic masking reduced catastrophic proxy events from six to one, while the final reliability-and-gating configuration eliminated them in the replay at the cost of 78 held frames. An unguarded weighted fit increased valid output but also increased catastrophic events. Dormant track memory produced only two reacquisitions and slightly fewer valid frames, so it remains compiled but disabled.",
    )

    heading(doc, "4.6 Learned alternatives tested offline", 2)
    add_table(
        doc,
        ["Method", "Observed replay result", "Current disposition"],
        [
            ("CoTracker3 (CPU prototype)", "90 sampled frames; 51 valid; 2 catastrophic proxy steps; 2.81 s per sampled frame", "Not suitable for this phone's live path in tested form"),
            ("XFeat recovery", "28 attempts; 8 accepted (28.6%); median 1.50 s per attempt; no catastrophic corrections", "Potential asynchronous recovery research, not live deployment"),
            ("Dormant sparse-track memory", "2 tracks reacquired; 318 vs 319 valid frames; more hold frames", "Disabled by default"),
        ],
        [2600, 4400, 2360],
    )
    add_source_note(doc, "Source: completed offline experiments on InkMARC_20260921_183433. CPU timing is implementation- and hardware-specific.")

    doc.add_page_break()
    heading(doc, "5. Relationship to prior work", 1)
    add_body(
        doc,
        "The implementation is best understood as a deliberately narrow synthesis of planar tracking, robust visual-inertial estimation, online point tracking, and semantic foreground masking. It does not reproduce a complete VIO or SLAM system. Instead, it exploits the known horizontal working surface and the single-overlay task to obtain a smaller state, lower computational cost, and interpretable failure handling.",
    )

    add_table(
        doc,
        ["Research line", "Established idea", "InkMARC adaptation and extension"],
        [
            ("Weighted planar tracking", "WOFT weights optical-flow correspondences before fitting an 8-DOF homography [1].", "Reliability is built from classical LK, forward-backward, and IMU-prior residuals, then used in a 3-parameter gravity-constrained plane fit with effective-support and spatial-coverage gates."),
            ("Dynamic visual-inertial estimation", "RD-VIO, DynaVINS++, and IDY-VINS combine inertial priors with robust handling of dynamic correspondences and recovery [2-4].", "The same principle is specialized to a single horizontal surface: inertial rotation is treated as known, dynamic evidence affects track weights/masks, and weak vision triggers bounded hold rather than contaminating a global state."),
            ("Long-term point tracking", "LEAP-VO, CoTracker3, ReTracker, SPOT, and Track-On2 use temporal context, memory, or retracking to survive occlusion [5-9].", "A lightweight dormant-track memory and two learned alternatives were evaluated on the project's own recording. They were not adopted without evidence of benefit and viable latency; the live path keeps explicit bounded memory through reliability, mask persistence, relock candidates, and plane velocity."),
            ("Efficient learned matching", "XFeat targets resource-efficient sparse/semi-dense matching [10].", "XFeat was used only as an offline recovery oracle. Its good inlier support but 1.50 s median attempt time motivated a recovery-only, asynchronous role rather than replacing live LK."),
            ("Semantic hand tracking", "MediaPipe Hands provides on-device landmark detection [11].", "The landmark hull is expanded into a hand-plus-forearm exclusion mask and paired with an IMU-compensated residual mask so clipped hands and shadows can still be rejected."),
            ("Rolling-shutter VIO practice", "Modern VIO accounts for timestamp alignment and rolling exposure when high-rate motion matters.", "Per-row orientation, online camera-IMU offset testing, end-to-end latency telemetry, and display-time plane prediction are joined in the same mobile overlay pipeline."),
        ],
        [2100, 3300, 3960],
    )

    heading(doc, "6. Novel contribution", 1)
    add_callout(
        doc,
        "Contribution claim",
        "The defensible novelty is the system-level combination and specialization, not the invention of optical flow, gravity sensing, robust estimation, or semantic hand detection individually. In the literature reviewed through September 2026, no cited work presents this exact combination as an ARCore-independent, on-device tracker for a user-placed rectangle on a known horizontal writing surface.",
    )
    for text in [
        "A task-specific state reduction: device orientation and gravity remove rotation and plane-normal ambiguity, leaving a three-parameter visual translation-over-height solve rather than a free homography or full SLAM state.",
        "Cross-modal track reliability: each correspondence is graded using image evidence and its residual from the IMU-plane prediction, then influences both robust estimation and effective-support accounting.",
        "A paired occlusion strategy: semantic hand geometry and IMU-compensated photometric dynamics are combined so either can exclude contaminated tracks, with explicit evidence showing that the geometric-photometric layer rescued a weak semantic detector in the final run.",
        "Safety-gated continuity: support, spatial distribution, residual, innovation, and relock consistency decide whether vision is allowed to correct the plane; otherwise the system retains a bounded inertial/constant-velocity prediction.",
        "Timing-aware mobile implementation: rolling-shutter row times, camera-IMU alignment, output-age prediction, portrait coordinate mapping, and raw-video/CSV synchronization are treated as part of the estimator rather than as recording or UI details.",
        "Evidence-driven integration: offline recordings were used to compare masking, reliability weighting, safety gates, track memory, CoTracker3, and XFeat. Components that did not improve the project's replay evidence were left out of the live path.",
    ]:
        add_list_item(doc, text, number_id)

    heading(doc, "7. Limitations and threats to validity", 1)
    for text in [
        "No external ground truth. Continuity and low residual cannot establish absolute registration accuracy or long-term metric drift.",
        "Single final run and single phone. Results may change with camera intrinsics, sensor timestamp behavior, lighting, surface texture, and processing capacity.",
        "High and variable latency. Median total latency is 113 ms and p95 is 190 ms. The 140 ms predictor cap leaves the tail uncompensated, while constant-velocity translation is weakest at motion reversals.",
        "Heavy use of hold. A 100% visible overlay after placement includes 47.1% held rows and one 2.45 s hold. Availability must not be presented as 100% fresh tracking.",
        "Semantic hand detector weakness. Only one semantic-mask row was recorded despite obvious hand visibility. The dynamic mask is effective in this clip but may confuse lighting changes, specularities, or scene motion with occlusion.",
        "Horizontal-plane assumption. The model is intentionally invalid for non-horizontal or non-planar targets unless the plane normal is estimated by another method.",
        "Raw recording versus displayed overlay. The MP4 is the camera stream; the annotated video is reconstructed from CSV. This is synchronized evidence, but not a direct screen capture of every display refresh.",
        "Offline proxy definitions. The >100 px catastrophic-step criterion and valid-frame counts are useful within a replay, but they are not standard benchmark metrics and should not be compared directly with published dataset scores.",
    ]:
        add_list_item(doc, text, bullet_id)

    heading(doc, "8. Thesis-ready interpretation", 1)
    add_body(
        doc,
        "The study supports a thesis claim that a known horizontal working plane can be tracked on non-ARCore hardware by combining commodity inertial orientation with sparse visual correspondence, rather than reconstructing a general 3-D map. The implementation demonstrates why the constraint matters: rotation and perspective remain stable under phone tilt, while visual measurements are reserved for the translation components that gravity cannot observe.",
    )
    add_body(
        doc,
        "The final run also shows the value of uncertainty-aware failure handling. Dynamic foreground motion does not simply disappear; it reduces spatially distributed effective support. The tracker responds by withholding correction and propagating a bounded model until reliable table evidence returns. This is preferable to an apparently responsive but unconstrained homography that attaches to the hand. The evidence therefore supports robustness and continuity claims, while also showing that latency and prolonged prediction remain the principal barriers to precise stylus registration.",
    )
    add_body(
        doc,
        "Within the complete InkMARC thesis, this surface tracker should be framed as the reference-coordinate layer beneath stylus-tip inference. Its role is to maintain a stable mapping between image pixels and a local writing-surface frame. Stylus inference can then be evaluated separately for tip localisation, presence, and contact state. This separation makes both modules easier to diagnose and avoids attributing surface-motion error to the learned stylus model.",
    )

    heading(doc, "9. Conclusion", 1)
    add_body(
        doc,
        "InkMARC has progressed from a distorted, axis-misaligned ARCore fallback to an installable non-ARCore implementation with direct portrait preview, horizontal-plane visual-inertial tracking, explicit occlusion handling, and a synchronized evaluation pipeline. In the final 35.96 s recording, the post-placement overlay remained available throughout and survived the hand episode. Accepted fits were internally well-supported and low-residual. The result is nevertheless not equivalent to ground-truth-accurate registration: half of post-placement rows used a new constrained visual correction, 47.1% were held, and median total latency was 113.4 ms.",
    )
    add_body(
        doc,
        "The most defensible research contribution is the principled integration of a gravity-constrained plane model, image-and-IMU track reliability, dual occlusion evidence, safety-gated relocking, rolling-shutter timing, and synchronized diagnostics into a lightweight Android path that does not require ARCore. This is a credible thesis contribution as a specialized systems method and an evidence-led engineering study, provided the claims remain bounded to continuity, robustness, and internal consistency rather than unmeasured absolute accuracy.",
    )

    heading(doc, "References", 1)
    refs = [
        ("J. Serych and J. Matas, 'Planar Object Tracking via Weighted Optical Flow,' WACV, 2023.", "https://arxiv.org/abs/2301.10057"),
        ("J. Li et al., 'RD-VIO: Robust Visual-Inertial Odometry for Mobile Augmented Reality in Dynamic Environments,' IEEE TVCG, 2024.", "https://doi.org/10.1109/TVCG.2024.3353263"),
        ("S. Song, H. Lim, A. J. Lee, and H. Myung, 'DynaVINS++: Robust Visual-Inertial State Estimator in Dynamic Environments by Adaptive Truncated Least Squares and Stable State Recovery,' IEEE RA-L, 2024.", "https://doi.org/10.1109/LRA.2024.3455905"),
        ("W. Sun, Y. Zhang, and B. Wei, 'A Visual-Inertial Motion Prior SLAM for Dynamic Environments (IDY-VINS),' arXiv:2503.23429, 2025.", "https://arxiv.org/abs/2503.23429"),
        ("W. Chen, L. Chen, R. Wang, and M. Pollefeys, 'LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry,' CVPR, 2024.", "https://openaccess.thecvf.com/content/CVPR2024/html/Chen_LEAP-VO_Long-term_Effective_Any_Point_Tracking_for_Visual_Odometry_CVPR_2024_paper.html"),
        ("N. Karaev et al., 'CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos,' ICCV, 2025.", "https://openaccess.thecvf.com/content/ICCV2025/html/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.html"),
        ("D. Tan et al., 'ReTracker: Exploring Image Matching for Robust Online Any Point Tracking,' ICCV, 2025.", "https://openaccess.thecvf.com/content/ICCV2025/papers/Tan_ReTracker_Exploring_Image_Matching_for_Robust_Online_Any_Point_Tracking_ICCV_2025_paper.pdf"),
        ("Q. Dong and Y. Fu, 'Online Dense Point Tracking with Streaming Memory (SPOT),' ICCV, 2025.", "https://openaccess.thecvf.com/content/ICCV2025/html/Dong_Online_Dense_Point_Tracking_with_Streaming_Memory_ICCV_2025_paper.html"),
        ("G. Aydemir, W. Xie, and F. Guney, 'Track-On2: Enhancing Online Point Tracking with Memory,' IEEE TPAMI, 2026.", "https://arxiv.org/abs/2509.19115"),
        ("G. Potje, F. Cadar, A. Araujo, R. Martins, and E. R. Nascimento, 'XFeat: Accelerated Features for Lightweight Image Matching,' CVPR, 2024.", "https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html"),
        ("F. Zhang et al., 'MediaPipe Hands: On-device Real-time Hand Tracking,' arXiv:2006.10214, 2020.", "https://arxiv.org/abs/2006.10214"),
    ]
    for index, (citation, url) in enumerate(refs, start=1):
        add_reference(doc, index, citation, url)

    heading(doc, "Appendix A. Reproducibility record", 1)
    add_table(
        doc,
        ["Artifact", "Role"],
        [
            ("app/src/main/assets/InkMARC_20260922_122250.mp4", "Final raw camera recording"),
            ("app/src/main/assets/InkMARC_20260922_122250_sensors.csv", "Final synchronized sensor and tracker telemetry"),
            ("tools/annotate_session.py", "Reconstructs annotated video from synchronized quadrilateral coordinates"),
            ("tools/analyze_final_run.py", "Produces final-run metrics JSON and telemetry timeline without rerunning the tracker"),
            (".analysis_tracking/final_122250/metrics.json", "Machine-readable metrics used in this report"),
            (".analysis_tracking/stage0_baseline.json", "Mask ablation evidence"),
            (".analysis_tracking/stage4_memory_ablation.json", "Safety-gate and track-memory evidence"),
            (".analysis_tracking/cotracker3_hand_interval.json", "CoTracker3 offline result"),
            (".analysis_tracking/xfeat_recovery.json", "XFeat recovery result"),
        ],
        [4350, 5010],
    )

    heading(doc, "Appendix B. Key implementation files", 1)
    add_table(
        doc,
        ["File", "Responsibility"],
        [
            ("InkMarcActivity.kt", "Feature toggles and selection of ARCore versus standalone path"),
            ("StandaloneCameraController.kt", "Camera2 session, portrait mapping, capture timing, and synchronized outputs"),
            ("FallbackSurfaceTracker.kt", "Live orchestration, masks, seeding, recovery, telemetry, and overlay updates"),
            ("fallback/SurfaceTrackerCore.kt", "Sparse LK, consistency checks, feature lifecycle, and track reliability"),
            ("ImuPlaneConstraint.kt", "Horizontal-plane fusion, gating, relock, velocity prediction, and row timing"),
            ("PlaneMotionGeometry.kt", "Weighted robust three-parameter plane solve"),
            ("DynamicOcclusionMasker.kt", "IMU-compensated photometric motion and shadow mask"),
            ("MediaPipeHandOcclusionMasker.kt", "Asynchronous expanded hand/forearm landmark mask"),
            ("SessionVideoRecorder.kt / SessionSensorRecorder.kt", "Raw H.264 and schema-7 synchronized telemetry"),
        ],
        [4200, 5160],
    )

    # Metadata and deterministic core properties.
    props = doc.core_properties
    props.title = "InkMARC ARCore-Independent Horizontal-Surface Tracking"
    props.subject = "Implementation and final observational evaluation"
    props.author = "InkMARC project"
    props.keywords = "visual-inertial tracking, horizontal plane, Android, ARCore fallback, occlusion"
    props.comments = "Generated from synchronized run InkMARC_20260922_122250 and completed replay evidence."

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
