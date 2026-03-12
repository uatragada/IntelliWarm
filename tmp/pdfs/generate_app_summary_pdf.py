from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, PageTemplate, Paragraph, Spacer


ROOT = Path(r"D:\Uday Atragada\Projects\DA\IntelliWarm")
OUTPUT = ROOT / "output" / "pdf" / "intelliwarm_app_summary_one_page.pdf"


def bullet_list(items, style, bullet_color):
    color = bullet_color.hexval()[2:]
    return [
        Paragraph(
            f"<font color='#{color}'><b>-</b></font> {item}",
            style,
        )
        for item in items
    ]


def section(title, body, styles):
    parts = [Paragraph(title, styles["section"])]
    parts.extend(body)
    return parts


def build_pdf():
    page_size = landscape(letter)
    margin = 0.45 * inch
    gutter = 0.28 * inch
    column_width = (page_size[0] - (2 * margin) - gutter) / 2
    frame_height = page_size[1] - (2 * margin)

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=page_size,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )

    left = Frame(margin, margin, column_width, frame_height, id="left", showBoundary=0)
    right = Frame(margin + column_width + gutter, margin, column_width, frame_height, id="right", showBoundary=0)
    doc.addPageTemplates(PageTemplate(id="two_col", frames=[left, right]))

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=21,
            textColor=colors.HexColor("#183153"),
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#56657A"),
            spaceAfter=8,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=12,
            textColor=colors.HexColor("#0F4C5C"),
            spaceBefore=4,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=10.3,
            textColor=colors.black,
            spaceAfter=3,
        ),
        "feature": ParagraphStyle(
            "Feature",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.1,
            leading=9.7,
            textColor=colors.black,
            spaceAfter=1,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=9.2,
            textColor=colors.black,
            spaceAfter=2,
        ),
        "step": ParagraphStyle(
            "Step",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.0,
            leading=9.8,
            leftIndent=10,
            firstLineIndent=-10,
            spaceAfter=2,
        ),
    }

    title = [
        Paragraph("IntelliWarm", styles["title"]),
        Paragraph(
            "One-page repo-grounded summary generated from README, docs, configs, and code.",
            styles["subtitle"],
        ),
    ]

    what_it_is = section(
        "What It Is",
        [
            Paragraph(
                "IntelliWarm is a Flask-based HVAC optimization platform for hybrid heating systems in homes and offices. "
                "It is designed to minimize heating cost by combining room-level control, zone concepts, forecasting inputs, "
                "simulation, and hardware-ready integration boundaries.",
                styles["body"],
            ),
        ],
        styles,
    )

    who_its_for = section(
        "Who It's For",
        [
            Paragraph(
                "Primary user/persona: a home or small-building operator who configures rooms and zones, monitors temperatures "
                "and occupancy, and reviews heating decisions through the dashboard and APIs.",
                styles["body"],
            ),
            Paragraph(
                "Specific named persona: Not found in repo.",
                styles["small"],
            ),
        ],
        styles,
    )

    what_it_does = section(
        "What It Does",
        bullet_list(
            [
                "Bootstraps a Flask app with typed config loading, route registration, scheduler startup, and runtime services.",
                "Models rooms, zones, heating actions, simulation state, and hybrid-heating decision contracts in typed data models.",
                "Runs deterministic multi-room thermal simulation for offline validation and replayable testing.",
                "Builds aligned forecast bundles for occupancy, weather fallback, and energy pricing inputs.",
                "Uses an explainable baseline controller with discrete OFF, ECO, COMFORT, and PREHEAT actions.",
                "Includes a zone-level HybridController that compares electric-heater cost versus gas-furnace cost and explains the choice.",
                "Persists room data, logs, optimization runs, and reporting queries in SQLite; exposes dashboard and room APIs.",
            ],
            styles["feature"],
            colors.HexColor("#0F4C5C"),
        ),
        styles,
    )

    how_it_works = section(
        "How It Works",
        [
            Paragraph(
                "<b>Inputs and config:</b> <font name='Courier'>configs/config.yaml</font> defines system settings, energy prices, rooms, zones, devices, and database path. "
                "<font name='Courier'>SystemConfig</font> loads typed config and environment overrides.",
                styles["small"],
            ),
            Paragraph(
                "<b>Bootstrap:</b> <font name='Courier'>create_runtime_bootstrap()</font> wires <font name='Courier'>Database</font>, "
                "<font name='Courier'>SensorManager</font>, <font name='Courier'>DeviceController</font>, <font name='Courier'>EnergyPriceService</font>, "
                "<font name='Courier'>ForecastBundleService</font>, <font name='Courier'>SystemScheduler</font>, and <font name='Courier'>IntelliWarmRuntime</font>.",
                styles["small"],
            ),
            Paragraph(
                "<b>Runtime flow:</b> Flask routes call runtime service methods for dashboard data, room data, demo flows, and optimization cycles. "
                "The scheduler triggers <font name='Courier'>run_optimization_cycle()</font> on the configured poll interval.",
                styles["small"],
            ),
            Paragraph(
                "<b>Control path:</b> Baseline control is currently used in runtime optimization; repo docs mark HybridController runtime wiring as the highest-priority integration gap. "
                "HybridController already exists and compares per-room electric demand against zone furnace cost.",
                styles["small"],
            ),
            Paragraph(
                "<b>Persistence and UI:</b> SQLite stores rooms, readings, optimization history, and logs; dashboard templates and JSON endpoints surface current room state and reports.",
                styles["small"],
            ),
        ],
        styles,
    )

    how_to_run = section(
        "How To Run",
        [
            Paragraph("1. From the repo root, install dependencies: <font name='Courier'>pip install -r requirements.txt</font>", styles["step"]),
            Paragraph(
                "2. Optional quick verification: <font name='Courier'>pytest tests/test_simulation.py tests/test_runtime_service.py tests/test_modules.py</font>",
                styles["step"],
            ),
            Paragraph("3. Start the app: <font name='Courier'>python app.py</font>", styles["step"]),
            Paragraph("4. Open <font name='Courier'>http://localhost:5000</font> in a browser.", styles["step"]),
        ],
        styles,
    )

    evidence = section(
        "Repo Notes",
        [
            Paragraph(
                "Source basis: README, docs/architecture.md, docs/roadmap.md, docs/srs.md, app.py, intelliwarm/services/application.py, "
                "intelliwarm/services/runtime.py, intelliwarm/routes/dashboard.py, configs/config.yaml, requirements.txt.",
                styles["small"],
            )
        ],
        styles,
    )

    story = []
    story.extend(title)
    story.extend(what_it_is)
    story.extend(who_its_for)
    story.extend(what_it_does)
    story.append(FrameBreak())
    story.extend(how_it_works)
    story.append(Spacer(1, 0.06 * inch))
    story.extend(how_to_run)
    story.append(Spacer(1, 0.06 * inch))
    story.extend(evidence)

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
