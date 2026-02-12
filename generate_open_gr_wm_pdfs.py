from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


OUT_SYSTEM = Path("Open_GR_WM_System_Design.pdf")
OUT_SLIDES = Path("Open_GR_WM_Product_Features_Slides.pdf")


def c(hex_code: str) -> colors.Color:
    return colors.HexColor(hex_code)


PALETTE = {
    "bg0": c("#060B1A"),
    "bg1": c("#0D1733"),
    "panel": c("#111A36"),
    "panel2": c("#172347"),
    "text": c("#E8EEF9"),
    "muted": c("#A2B2CE"),
    "mint": c("#9FE7D3"),
    "sky": c("#9ED2FF"),
    "rose": c("#F6B4D5"),
    "peach": c("#F7C6A0"),
    "line": c("#26345E"),
}


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(a: colors.Color, b: colors.Color, t: float) -> colors.Color:
    return colors.Color(lerp(a.red, b.red, t), lerp(a.green, b.green, t), lerp(a.blue, b.blue, t))


def draw_gradient_bg(cv: canvas.Canvas, width: float, height: float) -> None:
    steps = 70
    for i in range(steps):
        t = i / float(steps - 1)
        cv.setFillColor(lerp_color(PALETTE["bg0"], PALETTE["bg1"], t))
        y = height * (i / float(steps))
        cv.rect(0, y, width, height / float(steps) + 1, stroke=0, fill=1)


def draw_orb(cv: canvas.Canvas, x: float, y: float, r: float, color: colors.Color) -> None:
    if hasattr(cv, "setFillAlpha"):
        for i in range(7):
            t = i / 6.0
            cv.saveState()
            cv.setFillAlpha(0.07 * (1.0 - t))
            cv.setFillColor(color)
            cv.circle(x, y, r * (1.0 + t * 2.4), stroke=0, fill=1)
            cv.restoreState()
    else:
        cv.setFillColor(color)
        cv.circle(x, y, r, stroke=0, fill=1)


def draw_brand(cv: canvas.Canvas, width: float, height: float, subtitle: str = "LOCAL GRAPH WORKBENCH") -> None:
    x = 36
    y = height - 42
    cv.setFillColor(PALETTE["panel2"])
    cv.roundRect(x - 8, y - 20, 230, 36, 10, stroke=0, fill=1)
    cv.setStrokeColor(PALETTE["line"])
    cv.roundRect(x - 8, y - 20, 230, 36, 10, stroke=1, fill=0)

    cv.setFillColor(PALETTE["mint"])
    cv.circle(x + 7, y - 2, 3, stroke=0, fill=1)
    cv.setFillColor(PALETTE["sky"])
    cv.circle(x + 14, y + 5, 3, stroke=0, fill=1)
    cv.setFillColor(PALETTE["rose"])
    cv.circle(x + 21, y - 1, 3, stroke=0, fill=1)

    cv.setFillColor(PALETTE["text"])
    cv.setFont("Helvetica-Bold", 12)
    cv.drawString(x + 34, y + 3, "Open GR --WM")
    cv.setFont("Helvetica", 8)
    cv.setFillColor(PALETTE["muted"])
    cv.drawString(x + 34, y - 9, subtitle)


def draw_wrapped(
    cv: canvas.Canvas,
    text: str,
    x: float,
    y_top: float,
    width: float,
    font: str = "Helvetica",
    size: int = 11,
    color: colors.Color = PALETTE["text"],
    leading: float = 14,
) -> float:
    cv.setFillColor(color)
    cv.setFont(font, size)
    lines = simpleSplit(text, font, size, width)
    y = y_top
    for line in lines:
        cv.drawString(x, y, line)
        y -= leading
    return y


def draw_panel(
    cv: canvas.Canvas,
    x: float,
    y_top: float,
    w: float,
    h: float,
    title: str,
    body: str,
    accent: colors.Color,
) -> None:
    y = y_top - h
    cv.setFillColor(PALETTE["panel"])
    cv.setStrokeColor(PALETTE["line"])
    cv.roundRect(x, y, w, h, 12, stroke=1, fill=1)

    cv.setFillColor(accent)
    cv.roundRect(x + 8, y + h - 28, 4, 20, 2, stroke=0, fill=1)

    cv.setFont("Helvetica-Bold", 12)
    cv.setFillColor(PALETTE["text"])
    cv.drawString(x + 18, y + h - 22, title)
    draw_wrapped(cv, body, x + 18, y + h - 40, w - 28, font="Helvetica", size=10, color=PALETTE["muted"], leading=13)


def draw_chip(cv: canvas.Canvas, x: float, y: float, text: str, accent: colors.Color) -> None:
    w = max(70, 10 + 6.4 * len(text))
    cv.setFillColor(PALETTE["panel2"])
    cv.setStrokeColor(accent)
    cv.roundRect(x, y, w, 20, 10, stroke=1, fill=1)
    cv.setFillColor(PALETTE["text"])
    cv.setFont("Helvetica", 9)
    cv.drawString(x + 8, y + 6, text)


def draw_arrow(cv: canvas.Canvas, x1: float, y1: float, x2: float, y2: float) -> None:
    cv.setStrokeColor(PALETTE["sky"])
    cv.setLineWidth(1.4)
    cv.line(x1, y1, x2, y2)
    cv.line(x2, y2, x2 - 6, y2 + 3)
    cv.line(x2, y2, x2 - 6, y2 - 3)


def make_system_design(path: Path) -> None:
    w, h = A4
    cv = canvas.Canvas(str(path), pagesize=A4)

    # Page 1
    draw_gradient_bg(cv, w, h)
    draw_orb(cv, w * 0.15, h * 0.82, 38, PALETTE["sky"])
    draw_orb(cv, w * 0.86, h * 0.18, 34, PALETTE["rose"])
    draw_brand(cv, w, h)

    cv.setFillColor(PALETTE["text"])
    cv.setFont("Helvetica-Bold", 30)
    cv.drawString(40, h - 120, "System Design")
    cv.setFont("Helvetica", 14)
    cv.setFillColor(PALETTE["mint"])
    cv.drawString(40, h - 145, "Open GR --WM")

    summary = (
        "A local-first Graph RAG product that ingests PDFs, routes extraction across text and vision models, "
        "builds a persistent knowledge graph, and serves grounded chat answers with source citations and quality checks."
    )
    draw_panel(cv, 40, h - 180, w - 80, 110, "Architecture Intent", summary, PALETTE["mint"])

    draw_panel(
        cv,
        40,
        h - 305,
        (w - 100) / 2,
        150,
        "Design Principles",
        "1) Fully local execution. 2) Model-specialized pipeline. 3) Persistent graph lifecycle. "
        "4) Fast retrieval before deep reasoning. 5) Explainability via chunk/graph citations.",
        PALETTE["sky"],
    )
    draw_panel(
        cv,
        60 + (w - 100) / 2,
        h - 305,
        (w - 100) / 2,
        150,
        "Model Allocation",
        "llama3.2 handles fast text extraction + judge checks, llama3.2-vision handles visual pages, "
        "and deepseek-r1:14b handles rich chat reasoning.",
        PALETTE["rose"],
    )

    cv.setFillColor(PALETTE["muted"])
    cv.setFont("Helvetica", 9)
    cv.drawRightString(w - 36, 24, "Page 1 | Executive Overview")
    cv.showPage()

    # Page 2: end-to-end map
    draw_gradient_bg(cv, w, h)
    draw_brand(cv, w, h, subtitle="ARCHITECTURE MAP")
    cv.setFillColor(PALETTE["text"])
    cv.setFont("Helvetica-Bold", 22)
    cv.drawString(40, h - 94, "End-to-End Flow")

    boxes: List[Tuple[str, float, float, float, float, colors.Color]] = [
        ("1. Upload PDF", 40, h - 150, 120, 40, PALETTE["peach"]),
        ("2. Page Router", 185, h - 150, 120, 40, PALETTE["mint"]),
        ("3A. Text Scanner", 330, h - 130, 120, 40, PALETTE["sky"]),
        ("3B. Vision Scanner", 330, h - 175, 120, 40, PALETTE["rose"]),
        ("4. Graph Builder", 475, h - 150, 120, 40, PALETTE["mint"]),
        ("5. Persist Graph", 40, h - 240, 130, 40, PALETTE["sky"]),
        ("6. Load/Unload", 190, h - 240, 130, 40, PALETTE["peach"]),
        ("7. Retrieve Context", 340, h - 240, 130, 40, PALETTE["mint"]),
        ("8. DeepSeek Answer", 490, h - 240, 130, 40, PALETTE["rose"]),
        ("9. Judge + Retry", 340, h - 300, 130, 40, PALETTE["sky"]),
        ("10. Render Chat", 490, h - 300, 130, 40, PALETTE["peach"]),
    ]
    for label, x, y, bw, bh, accent in boxes:
        cv.setFillColor(PALETTE["panel2"])
        cv.setStrokeColor(accent)
        cv.roundRect(x, y, bw, bh, 10, stroke=1, fill=1)
        cv.setFillColor(PALETTE["text"])
        cv.setFont("Helvetica", 9)
        cv.drawCentredString(x + bw / 2, y + bh / 2 - 3, label)

    draw_arrow(cv, 160, h - 130, 185, h - 130)
    draw_arrow(cv, 305, h - 130, 330, h - 110)
    draw_arrow(cv, 305, h - 130, 330, h - 155)
    draw_arrow(cv, 450, h - 110, 475, h - 130)
    draw_arrow(cv, 450, h - 155, 475, h - 140)
    draw_arrow(cv, 540, h - 150, 540, h - 200)
    draw_arrow(cv, 170, h - 220, 190, h - 220)
    draw_arrow(cv, 320, h - 220, 340, h - 220)
    draw_arrow(cv, 470, h - 220, 490, h - 220)
    draw_arrow(cv, 405, h - 260, 405, h - 300)
    draw_arrow(cv, 470, h - 300, 490, h - 300)

    draw_panel(
        cv,
        40,
        h - 370,
        w - 80,
        140,
        "Key Choice Callouts",
        "Page-level routing avoids expensive vision inference on text-heavy pages. Chunk retrieval narrows context before "
        "reasoning. LLM-as-judge catches unsupported claims and triggers one retry. Storage supports load/unload/delete and "
        "multi-PDF graph augmentation.",
        PALETTE["mint"],
    )

    cv.setFillColor(PALETTE["muted"])
    cv.setFont("Helvetica", 9)
    cv.drawRightString(w - 36, 24, "Page 2 | Architecture Flow")
    cv.showPage()

    # Page 3: component deep dive
    draw_gradient_bg(cv, w, h)
    draw_brand(cv, w, h, subtitle="COMPONENT DECISIONS")
    cv.setFillColor(PALETTE["text"])
    cv.setFont("Helvetica-Bold", 22)
    cv.drawString(40, h - 94, "Component Rationale")

    y = h - 130
    callouts = [
        (
            "Ingestion Layer",
            "PyPDF2 extracts text per page. Image-page detection and text-density checks enable selective vision inference.",
            PALETTE["sky"],
        ),
        (
            "Graph Extraction",
            "NetworkX MultiDiGraph stores entities and predicates. JSON parsing is resilient to model filler and malformed blocks.",
            PALETTE["mint"],
        ),
        (
            "Embeddings & Retrieval",
            "all-MiniLM-L6-v2 embeddings support fast cosine retrieval over chunks before LLM reasoning.",
            PALETTE["peach"],
        ),
        (
            "Reasoning & Quality",
            "deepseek-r1:14b produces the answer, while a fast local judge model flags weak grounding and requests a retry.",
            PALETTE["rose"],
        ),
        (
            "Persistence Lifecycle",
            "Graph, triples, chunks, embeddings, and metadata are saved per graph. Users can load, unload from RAM, augment, or delete.",
            PALETTE["sky"],
        ),
    ]
    for title, body, accent in callouts:
        draw_panel(cv, 40, y, w - 80, 92, title, body, accent)
        y -= 102

    cv.setFillColor(PALETTE["muted"])
    cv.setFont("Helvetica", 9)
    cv.drawRightString(w - 36, 24, "Page 3 | Component-Level Decisions")
    cv.showPage()

    # Page 4: operations, risks, mitigations
    draw_gradient_bg(cv, w, h)
    draw_brand(cv, w, h, subtitle="OPERATIONS + TRADEOFFS")
    cv.setFillColor(PALETTE["text"])
    cv.setFont("Helvetica-Bold", 22)
    cv.drawString(40, h - 94, "Operational Playbook")

    draw_panel(
        cv,
        40,
        h - 130,
        w - 80,
        105,
        "Performance Controls",
        "Tune build mode (Fast/Balanced/Thorough), max vision pages, and text batch size. Use augment mode to incrementally grow "
        "graphs instead of full rebuilds.",
        PALETTE["mint"],
    )
    draw_panel(
        cv,
        40,
        h - 245,
        (w - 100) / 2,
        170,
        "Known Risks",
        "1) Incomplete model JSON\n2) Long answer truncation\n3) Vision over-processing\n4) Context mismatch across chunks\n"
        "5) UI state resets during reruns",
        PALETTE["rose"],
    )
    draw_panel(
        cv,
        60 + (w - 100) / 2,
        h - 245,
        (w - 100) / 2,
        170,
        "Mitigations in App",
        "Robust JSON extraction + fallbacks, answer continuation pass, page-level routing, citation enforcement, judge-and-retry "
        "loop, and stateful workspace selection.",
        PALETTE["sky"],
    )

    draw_panel(
        cv,
        40,
        h - 430,
        w - 80,
        135,
        "Future Enhancements",
        "1) Background worker queue for non-blocking chat/build. 2) Citation-level confidence scoring. 3) Table-optimized OCR pass. "
        "4) Graph version snapshots and rollback. 5) Async streaming by token with smoother UI transitions.",
        PALETTE["peach"],
    )

    draw_chip(cv, 42, 42, "Local-only", PALETTE["mint"])
    draw_chip(cv, 145, 42, "Multi-model", PALETTE["sky"])
    draw_chip(cv, 255, 42, "Persistent Graphs", PALETTE["rose"])
    draw_chip(cv, 392, 42, "Grounded Chat", PALETTE["peach"])

    cv.setFillColor(PALETTE["muted"])
    cv.setFont("Helvetica", 9)
    cv.drawRightString(w - 36, 24, "Page 4 | Operations and Next Steps")
    cv.save()


def slide_size() -> Tuple[float, float]:
    return 13.333 * 72, 7.5 * 72


def draw_slide_header(cv: canvas.Canvas, w: float, h: float, title: str, subtitle: str, slide_idx: int, total: int) -> None:
    draw_gradient_bg(cv, w, h)
    draw_orb(cv, w * 0.1, h * 0.85, 42, PALETTE["sky"])
    draw_orb(cv, w * 0.9, h * 0.18, 34, PALETTE["rose"])

    cv.setFillColor(PALETTE["panel2"])
    cv.roundRect(30, h - 58, 180, 30, 10, stroke=0, fill=1)
    cv.setFont("Helvetica-Bold", 11)
    cv.setFillColor(PALETTE["text"])
    cv.drawString(42, h - 39, "Open GR --WM")

    cv.setFont("Helvetica-Bold", 34)
    cv.setFillColor(PALETTE["text"])
    cv.drawString(36, h - 112, title)
    cv.setFont("Helvetica", 16)
    cv.setFillColor(PALETTE["mint"])
    cv.drawString(38, h - 138, subtitle)

    cv.setFont("Helvetica", 10)
    cv.setFillColor(PALETTE["muted"])
    cv.drawRightString(w - 24, 22, f"{slide_idx}/{total}")


def draw_bullets(cv: canvas.Canvas, x: float, y_top: float, width: float, bullets: List[str]) -> None:
    y = y_top
    for bullet in bullets:
        cv.setFillColor(PALETTE["mint"])
        cv.circle(x + 4, y - 4, 2.4, stroke=0, fill=1)
        y = draw_wrapped(cv, bullet, x + 14, y, width - 14, font="Helvetica", size=13, color=PALETTE["text"], leading=18)
        y -= 6


def draw_stat_card(cv: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, value: str, note: str, accent: colors.Color) -> None:
    cv.setFillColor(PALETTE["panel"])
    cv.setStrokeColor(PALETTE["line"])
    cv.roundRect(x, y, w, h, 14, stroke=1, fill=1)
    cv.setFillColor(accent)
    cv.roundRect(x + 12, y + h - 18, 32, 6, 3, stroke=0, fill=1)
    cv.setFont("Helvetica", 11)
    cv.setFillColor(PALETTE["muted"])
    cv.drawString(x + 12, y + h - 34, title)
    cv.setFont("Helvetica-Bold", 22)
    cv.setFillColor(PALETTE["text"])
    cv.drawString(x + 12, y + h - 60, value)
    cv.setFont("Helvetica", 10)
    cv.setFillColor(PALETTE["mint"])
    cv.drawString(x + 12, y + 14, note)


def make_slide_deck(path: Path) -> None:
    w, h = slide_size()
    cv = canvas.Canvas(str(path), pagesize=(w, h))

    slides = []
    slides.append(
        {
            "title": "Your Local Graph RAG",
            "subtitle": "All brains. No cloud bill panic.",
            "bullets": [
                "Turns PDF chaos into a structured knowledge graph you can chat with.",
                "Runs completely on your Mac with specialized local models.",
                "Built for real docs: annual reports, charts, and table-heavy narratives.",
            ],
            "stats": [
                ("Privacy", "100% Local", "No external API calls", PALETTE["mint"]),
                ("Model Roles", "3 Specialists", "Right model, right job", PALETTE["sky"]),
            ],
        }
    )
    slides.append(
        {
            "title": "Feature 1: Smart Ingestion",
            "subtitle": "Upload PDFs, route pages, skip wasted compute.",
            "bullets": [
                "Per-page routing: every page gets text extraction.",
                "Vision model only runs on pages with images or very low text density.",
                "Build modes let you trade speed vs depth.",
                "Normie translation: your laptop sweats only where it matters.",
            ],
            "stats": [
                ("Routing", "Page-level", "No blanket vision burn", PALETTE["peach"]),
                ("Control", "Fast/Balanced/Thorough", "One-click tuning", PALETTE["rose"]),
            ],
        }
    )
    slides.append(
        {
            "title": "Feature 2: Three-Model Architecture",
            "subtitle": "Speed, sight, and reasoning in one pipeline.",
            "bullets": [
                "Scanner: llama3.2 extracts entities and relations from text.",
                "Eyes: llama3.2-vision reads charts/tables from selected pages.",
                "Brain: deepseek-r1:14b handles final multi-hop reasoning in chat.",
                "Judge: fast local check catches unsupported claims before final output.",
            ],
            "stats": [
                ("Scanner", "llama3.2", "Fast extraction", PALETTE["mint"]),
                ("Brain", "deepseek-r1:14b", "Deep reasoning", PALETTE["sky"]),
            ],
        }
    )
    slides.append(
        {
            "title": "Feature 3: Knowledge Graph Build",
            "subtitle": "Entities + predicates become a queryable map.",
            "bullets": [
                "NetworkX MultiDiGraph stores rich directed relationships.",
                "Triples from text and vision are composed into one graph.",
                "Progress logs and stop controls keep long builds manageable.",
                "If parsing gets weird, robust JSON fallback logic prevents crashes.",
            ],
            "stats": [
                ("Structure", "MultiDiGraph", "Direct + reverse neighbor lookup", PALETTE["rose"]),
                ("Resilience", "Robust Parser", "Handles model filler", PALETTE["peach"]),
            ],
        }
    )
    slides.append(
        {
            "title": "Feature 4: Persistent Graph Library",
            "subtitle": "Build once. Reuse anytime.",
            "bullets": [
                "Auto-save graph, triples, chunks, and embeddings to disk.",
                "Load/unload from RAM on demand.",
                "Delete from disk to reclaim space.",
                "Disk size and source-file count are visible per graph.",
            ],
            "stats": [
                ("Lifecycle", "Load / Unload / Delete", "Storage in your control", PALETTE["mint"]),
                ("Metadata", "Source-aware", "Tracks augmented files", PALETTE["sky"]),
            ],
        }
    )
    slides.append(
        {
            "title": "Feature 5: Graph Augmentation",
            "subtitle": "Add more PDFs to the same graph over time.",
            "bullets": [
                "Choose 'Augment loaded graph' to merge new sources into existing knowledge.",
                "Embeddings and triples are appended; graph identity remains stable.",
                "Perfect for evolving research sets and periodic financial updates.",
                "Normie translation: one brain, many documents, less chaos.",
            ],
            "stats": [
                ("Mode", "Incremental", "No full rebuild required", PALETTE["peach"]),
                ("Continuity", "Stable Graph ID", "Same source of truth", PALETTE["rose"]),
            ],
        }
    )
    slides.append(
        {
            "title": "Feature 6: Chat Studio",
            "subtitle": "Grounded answers, citations, and visual callouts.",
            "bullets": [
                "Retrieval combines keyword graph neighbors + embedding top chunks.",
                "Answers cite [Graph] and [Chunk N] to show where facts came from.",
                "Hero figures and optional charts render in the response.",
                "Live thinking panel shows progress while the model reasons.",
            ],
            "stats": [
                ("Grounding", "Dual Retrieval", "Graph + vector context", PALETTE["mint"]),
                ("UX", "Live Thinking", "No dead-air waiting", PALETTE["sky"]),
            ],
        }
    )
    slides.append(
        {
            "title": "Feature 7: Quality Guardrails",
            "subtitle": "LLM judge checks before answer delivery.",
            "bullets": [
                "Judge model classifies answer as pass, retry, or not_found.",
                "Retry path asks for correction when grounding looks weak.",
                "If data is absent in context, app says so clearly.",
                "Normie translation: less confident nonsense, more useful honesty.",
            ],
            "stats": [
                ("Judge", "Fast Local", "Low-latency validation", PALETTE["rose"]),
                ("Fallback", "Not Found", "No fabricated answers", PALETTE["peach"]),
            ],
        }
    )
    slides.append(
        {
            "title": "Feature 8: Aesthetic UX",
            "subtitle": "Dark theme, playful glow, focus-first interactions.",
            "bullets": [
                "Two workspaces: Ingest + Graph Studio and Chat Studio.",
                "Progress bars, status bubbles, and dynamic states guide the user.",
                "Passive elements stay muted, active context lights up.",
                "Because serious tools can still look like they had coffee and style.",
            ],
            "stats": [
                ("Design", "Pastel on Dark", "Readable and modern", PALETTE["mint"]),
                ("Flow", "Context-aware", "Less tab confusion", PALETTE["sky"]),
            ],
        }
    )
    slides.append(
        {
            "title": "How to Use It in 30 Seconds",
            "subtitle": "Short path from PDF to answer.",
            "bullets": [
                "1) Ingest tab: upload PDF and click Build (or Augment).",
                "2) Load your graph from library when needed.",
                "3) Chat tab: ask question, inspect citations and hero figures.",
                "4) Repeat with new PDFs to keep your graph current.",
                "5) Delete old graphs when your SSD starts side-eyeing you.",
            ],
            "stats": [
                ("Time to Value", "Minutes", "Not quarters", PALETTE["peach"]),
                ("Reality", "Local compute", "Expect warm laptop vibes", PALETTE["rose"]),
            ],
        }
    )

    total = len(slides)
    for idx, slide in enumerate(slides, start=1):
        draw_slide_header(cv, w, h, slide["title"], slide["subtitle"], idx, total)
        draw_bullets(cv, 46, h - 178, w * 0.58, slide["bullets"])

        card_w = w * 0.32
        card_x = w * 0.64
        draw_stat_card(cv, card_x, h - 280, card_w, 110, *slide["stats"][0])
        draw_stat_card(cv, card_x, h - 408, card_w, 110, *slide["stats"][1])

        cv.showPage()

    cv.save()


def main() -> None:
    make_system_design(OUT_SYSTEM)
    make_slide_deck(OUT_SLIDES)
    print(f"Generated: {OUT_SYSTEM.resolve()}")
    print(f"Generated: {OUT_SLIDES.resolve()}")


if __name__ == "__main__":
    main()
